from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np

from .ffmpeg_utils import require


def _global_range(depth: np.ndarray, norm_mode: str) -> tuple[float, float] | None:
    """Return (lo, hi) for global colormap mapping, or None for per-frame mode.

    "global"      → exact min/max across all frames.
    "global-p99"  → 1st/99th percentiles, robust to outlier frames.
    "per-frame"   → returns None (caller falls back to per-frame normalization).
    """
    if norm_mode == "per-frame":
        return None
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return None
    if norm_mode == "global":
        lo, hi = float(finite.min()), float(finite.max())
    elif norm_mode == "global-p99":
        lo = float(np.percentile(finite, 1.0))
        hi = float(np.percentile(finite, 99.0))
    else:
        raise ValueError(
            f"unknown colormap norm_mode: {norm_mode!r} "
            "(choose per-frame, global, global-p99)"
        )
    if not (hi > lo):
        return None
    return lo, hi


def _render_frame(d: np.ndarray, cmap: str, lo_hi: tuple[float, float] | None) -> np.ndarray:
    """Return (H, W, 3) uint8 RGB for one depth slice."""
    if lo_hi is None:
        # Per-frame mode: defer to DA3's visualize_depth (matches legacy
        # output exactly).
        from depth_anything_3.utils.visualize import visualize_depth
        return visualize_depth(d, cmap=cmap).astype(np.uint8)
    lo, hi = lo_hi
    import matplotlib
    cm = matplotlib.colormaps.get_cmap(cmap)
    d = np.asarray(d, dtype=np.float32)
    d = np.where(np.isfinite(d), d, lo)
    n = np.clip((d - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    rgba = cm(n)  # (H, W, 4) in [0, 1]
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def write_depth_video(
    depth: np.ndarray,
    out_path: Path,
    fps: float,
    cmap: str = "gray",
    norm_mode: str = "global-p99",
) -> Path:
    """Render an (N, H, W) depth array to a browser-playable H.264 mp4.

    Pipes raw RGB frames into ffmpeg → libx264 (yuv420p, faststart). Avoids
    imageio's silent codec fallback to mpeg4 when imageio_ffmpeg's bundled
    binary lacks x264 support.

    ``norm_mode`` controls how the depth values map to colormap range:
    "per-frame" rebases [0, 1] every frame (legacy behavior, causes brightness
    flicker when the scene's depth range shifts), "global" pins to the entire
    sequence's min/max, "global-p99" uses 1st/99th percentiles (default —
    robust to a single noisy frame).
    """
    depth = np.asarray(depth, dtype=np.float32)
    nan_frac = float(np.isnan(depth).mean())
    inf_frac = float(np.isinf(depth).mean())
    finite = depth[np.isfinite(depth)]
    if finite.size:
        finite_min = float(finite.min())
        finite_max = float(finite.max())
    else:
        finite_min = finite_max = float("nan")
    print(
        f"[v2d] depth stats: shape={depth.shape}, "
        f"nan={nan_frac:.2%}, inf={inf_frac:.2%}, "
        f"min={finite_min:.4f}, max={finite_max:.4f}"
    )
    if nan_frac > 0 or inf_frac > 0:
        print(
            "[v2d] WARNING: depth tensor contains NaN/Inf — model output likely broken. "
            "On Apple Silicon, try --device cpu or a smaller model. "
            "Replacing NaN/Inf with 0 so the video still encodes."
        )
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if finite_min == finite_max:
        print(
            "[v2d] WARNING: depth tensor is constant — output will be a flat color. "
            "Likely an MPS attention bug; try --device cpu or a smaller model."
        )

    if depth.shape[0] == 0:
        raise ValueError("depth tensor has zero frames")

    lo_hi = _global_range(depth, norm_mode)
    if lo_hi is not None:
        print(
            f"[v2d] colormap norm: {norm_mode}  range=[{lo_hi[0]:.4f}, {lo_hi[1]:.4f}]"
        )
    else:
        print(f"[v2d] colormap norm: per-frame")

    first = _render_frame(depth[0], cmap, lo_hi)
    if first.ndim != 3 or first.shape[2] != 3:
        raise RuntimeError(f"unexpected rendered frame shape {first.shape}")
    h, w = first.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    require("ffmpeg")
    # pad to even dims — yuv420p requires width/height divisible by 2.
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", f"{fps}",
            "-i", "pipe:",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "slow",
            "-crf", "18",
            "-movflags", "+faststart",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        try:
            proc.stdin.write(first.tobytes())
            for idx in range(1, depth.shape[0]):
                frame = _render_frame(depth[idx], cmap, lo_hi)
                proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass  # ffmpeg already errored; rc check below surfaces it.
    finally:
        if proc.stdin:
            proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with code {rc} writing {out_path}")
    return out_path


def _fit_scale(target: np.ndarray, source: np.ndarray, eps: float = 1e-8) -> float:
    """Scalar least-squares fit: find s minimising ||target - s*source||²."""
    t = np.asarray(target, dtype=np.float64).flatten()
    s = np.asarray(source, dtype=np.float64).flatten()
    valid = np.isfinite(t) & np.isfinite(s)
    if not valid.any():
        return 1.0
    t = t[valid]; s = s[valid]
    num = float(np.sum(t * s))
    den = float(np.sum(s * s)) + eps
    return num / den


def stitch_chunks(chunks: Iterable[tuple[int, int, np.ndarray]]) -> np.ndarray:
    """Concatenate overlapping depth chunks with scale alignment + crossfade.

    Each chunk is ``(start, end, depth)`` where ``depth`` is ``(end-start, H, W)``.
    The first chunk defines the reference scale; subsequent chunks are rescaled
    by a scalar fit on their overlap with the running result, then linearly
    crossfaded across the overlap region.
    """
    chunks = list(chunks)
    if not chunks:
        raise ValueError("no chunks to stitch")
    cur = chunks[0][2].astype(np.float32, copy=True)
    for i in range(1, len(chunks)):
        prev_end = chunks[i - 1][1]
        new_start, _new_end, new_depth = chunks[i]
        new_depth = new_depth.astype(np.float32, copy=False)
        ov = prev_end - new_start
        if ov <= 0:
            cur = np.concatenate([cur, new_depth], axis=0)
            continue
        scale = _fit_scale(cur[-ov:], new_depth[:ov])
        new_aligned = new_depth * scale
        alpha = np.linspace(0.0, 1.0, ov, dtype=np.float32).reshape(-1, 1, 1)
        cur[-ov:] = cur[-ov:] * (1.0 - alpha) + new_aligned[:ov] * alpha
        cur = np.concatenate([cur, new_aligned[ov:]], axis=0)
    return cur


def plan_chunks(n_frames: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Plan (start, end) frame indices for chunked inference."""
    if chunk_size <= 0 or n_frames <= chunk_size:
        return [(0, n_frames)]
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be in [0, chunk_size). chunk_size={chunk_size}")
    step = chunk_size - overlap
    plan: list[tuple[int, int]] = []
    i = 0
    while i < n_frames:
        end = min(i + chunk_size, n_frames)
        plan.append((i, end))
        if end == n_frames:
            break
        i += step
    return plan
