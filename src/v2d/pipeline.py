from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .depth_video import plan_chunks, stitch_chunks, write_depth_video
from .ffmpeg_utils import mux_audio, probe_fps
from .interpolate import RifeRunner, choose_multiplier


@dataclass
class PipelineConfig:
    input_video: Path
    output_video: Path
    sample_fps: float = 1.0
    target_fps: Optional[float] = None
    model_dir: str = "depth-anything/DA3-LARGE-1.1"
    device: str = "cuda"
    process_res: int = 504
    rife_dir: Optional[Path] = None
    rife_python: Optional[str] = None
    rife_verbose: bool = False
    interpolator: str = "rife"  # "rife" | "none"
    chunk_size: int = 32
    chunk_overlap: int = 8
    keep_audio: bool = True
    work_dir: Optional[Path] = None
    vram_check: bool = True
    vram_safety: float = 1.25


def run(cfg: PipelineConfig) -> Path:
    source_fps = probe_fps(cfg.input_video)
    target_fps = cfg.target_fps or source_fps

    tmp_root = cfg.work_dir or Path(tempfile.mkdtemp(prefix="v2d_"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    da3_dir = tmp_root / "da3"
    da3_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v2d] source fps={source_fps:.3f}, sample fps={cfg.sample_fps}, target fps={target_fps:.3f}")

    _run_da3(cfg, da3_dir)
    depth_low = da3_dir / "depth_video.mp4"
    if not depth_low.exists():
        raise FileNotFoundError(f"DA3 did not produce {depth_low}")

    if cfg.interpolator == "none":
        print("[v2d] interpolation disabled (--no-interpolate)")
        depth_interp = depth_low
    elif cfg.interpolator == "rife":
        if cfg.rife_dir is None:
            raise ValueError("interpolator='rife' requires cfg.rife_dir")
        import sys
        multi = choose_multiplier(target_fps, cfg.sample_fps)
        print(f"[v2d] RIFE interpolating x{multi}")
        rife = RifeRunner(
            cfg.rife_dir, cfg.rife_python or sys.executable, verbose=cfg.rife_verbose,
        )
        depth_interp = tmp_root / "depth_interp.mp4"
        rife.interpolate(depth_low, depth_interp, multi, log_path=tmp_root / "rife.log")
    else:
        raise ValueError(f"unknown interpolator: {cfg.interpolator!r}")

    cfg.output_video.parent.mkdir(parents=True, exist_ok=True)
    if cfg.keep_audio:
        print("[v2d] muxing audio from source")
        mux_audio(depth_interp, cfg.input_video, cfg.output_video)
    else:
        shutil.copyfile(depth_interp, cfg.output_video)

    if cfg.work_dir is None:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"[v2d] done → {cfg.output_video}")
    return cfg.output_video


def _depth_array(prediction) -> np.ndarray:
    d = prediction.depth
    if hasattr(d, "detach"):
        d = d.detach().cpu().numpy()
    return np.asarray(d, dtype=np.float32)


def _fmt_gib(n_bytes: float) -> str:
    return f"{n_bytes / (1024 ** 3):.2f} GiB"


def _probe_vram_or_die(
    *,
    model,
    image_files: list,
    device: str,
    process_res: int,
    max_chunk: int,
    safety: float,
) -> None:
    """Empirical VRAM check: probe a tiny forward pass, extrapolate to max chunk.

    Skips silently on non-CUDA devices. Aborts with a sized suggestion when
    estimated peak (with safety margin) exceeds free VRAM on the target GPU.
    """
    if not device.startswith("cuda"):
        return

    import torch

    if not torch.cuda.is_available():
        return

    dev = torch.device(device)
    probe_n = min(4, max_chunk, len(image_files))
    if probe_n <= 0:
        return

    # Baseline = model weights + any persistent allocations already on device.
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)

    print(f"[v2d] probing VRAM with a {probe_n}-frame forward pass…")
    try:
        with torch.inference_mode():
            _ = model.inference(
                image=image_files[:probe_n],
                export_dir=None,
                process_res=process_res,
            )
        torch.cuda.synchronize(dev)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"[v2d] VRAM probe OOM at {probe_n} frames @ process_res={process_res}. "
            f"Lower --process-res or pick a smaller model."
        ) from exc

    peak = torch.cuda.max_memory_allocated(dev)
    activations = max(peak - baseline, 0)
    per_frame = activations / probe_n
    estimated = baseline + per_frame * max_chunk
    needed = estimated * safety

    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(dev)

    print(
        f"[v2d] VRAM probe: baseline={_fmt_gib(baseline)}, "
        f"per-frame≈{_fmt_gib(per_frame)}, "
        f"est. peak for {max_chunk}-frame chunk={_fmt_gib(estimated)} "
        f"(×{safety:g} safety = {_fmt_gib(needed)}); "
        f"free={_fmt_gib(free)} / total={_fmt_gib(total)}"
    )

    if needed <= free:
        return

    # Suggest a chunk size that fits.
    headroom = free / safety - baseline
    if per_frame > 0 and headroom > 0:
        safe_chunk = int(headroom / per_frame)
    else:
        safe_chunk = 0

    msg = (
        f"[v2d] insufficient VRAM: need ≈{_fmt_gib(needed)} for "
        f"chunk-size={max_chunk} @ process_res={process_res}, "
        f"only {_fmt_gib(free)} free.\n"
    )
    if safe_chunk >= 1:
        msg += f"  → try --chunk-size {safe_chunk} (or lower --process-res)."
    else:
        msg += "  → lower --process-res or pick a smaller model; chunking alone won't fit."
    msg += "\n  (bypass this check with --no-vram-check)"
    raise RuntimeError(msg)


def _run_da3(cfg: PipelineConfig, export_dir: Path) -> None:
    # Suppress noise we cannot fix at the source.
    import logging, os, warnings
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    warnings.filterwarnings(
        "ignore", message=r"You are sending unauthenticated requests to the HF Hub.*"
    )
    logging.getLogger("depth_anything_3").setLevel(logging.ERROR)

    # Import lazily — torch + DA3 weights load is slow.
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.services.input_handlers import VideoHandler

    print(f"[v2d] loading model {cfg.model_dir} on {cfg.device}")
    model = DepthAnything3.from_pretrained(cfg.model_dir).to(cfg.device)

    print(f"[v2d] extracting frames at {cfg.sample_fps} fps")
    image_files = VideoHandler.process(str(cfg.input_video), str(export_dir), cfg.sample_fps)
    n = len(image_files)

    plan = plan_chunks(n, cfg.chunk_size, cfg.chunk_overlap)

    if cfg.vram_check:
        max_chunk = max(e - s for s, e in plan)
        _probe_vram_or_die(
            model=model,
            image_files=image_files,
            device=cfg.device,
            process_res=cfg.process_res,
            max_chunk=max_chunk,
            safety=cfg.vram_safety,
        )

    if len(plan) == 1:
        print(f"[v2d] inference on {n} frames in a single pass")
        pred = model.inference(image=image_files, export_dir=None, process_res=cfg.process_res)
        depth = _depth_array(pred)
    else:
        print(
            f"[v2d] chunked inference: {len(plan)} chunks (chunk-size={cfg.chunk_size}, "
            f"overlap={cfg.chunk_overlap}, total {n} frames)"
        )
        chunks = []
        for ci, (s, e) in enumerate(plan):
            print(f"[v2d]   chunk {ci + 1}/{len(plan)}: frames {s}–{e - 1} ({e - s} frames)")
            pred = model.inference(
                image=image_files[s:e], export_dir=None, process_res=cfg.process_res,
            )
            chunks.append((s, e, _depth_array(pred)))
        depth = stitch_chunks(chunks)
        if depth.shape[0] != n:
            raise RuntimeError(
                f"stitched depth has {depth.shape[0]} frames, expected {n}"
            )

    out = export_dir / "depth_video.mp4"
    print(f"[v2d] writing depth video to {out} at {cfg.sample_fps} fps")
    write_depth_video(depth, out, fps=cfg.sample_fps)
