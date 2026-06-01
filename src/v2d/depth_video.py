from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio
import numpy as np


def write_depth_video(depth: np.ndarray, out_path: Path, fps: float) -> Path:
    """Render an (N, H, W) depth array to a colorised mp4."""
    from depth_anything_3.utils.visualize import visualize_depth

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8, macro_block_size=1,
    )
    try:
        for idx in range(depth.shape[0]):
            frame = visualize_depth(depth[idx]).astype(np.uint8)
            writer.append_data(frame)
    finally:
        writer.close()
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
