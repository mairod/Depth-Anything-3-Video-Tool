from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ._vram import probe_vram_or_die
from .depth_video import plan_chunks, stitch_chunks, write_depth_video
from .ffmpeg_utils import extract_frames, finalize_output, probe_duration, probe_fps
from .interpolate import RifeRunner, choose_multiplier


@dataclass
class PipelineConfig:
    input_video: Path
    output_video: Path
    backend: str = "da3"  # "da3" | "vda"
    sample_fps: float = 1.0
    target_fps: Optional[float] = None
    model_dir: str = "depth-anything/DA3-LARGE-1.1"
    device: str = "cuda"
    process_res: int = 504
    rife_dir: Optional[Path] = None
    rife_python: Optional[str] = None
    rife_verbose: bool = False
    interpolator: str = "rife"  # "rife" | "none" (DA3 only — VDA always skips RIFE)
    chunk_size: int = 32
    chunk_overlap: int = 8
    keep_audio: bool = True
    work_dir: Optional[Path] = None
    colormap: str = "gray"
    colormap_norm: str = "global-p99"  # per-frame | global | global-p99
    vram_check: bool = True
    vram_safety: float = 1.25
    # VDA-specific
    vda_dir: Optional[Path] = None
    vda_encoder: str = "vitl"
    vda_input_size: int = 518
    vda_fp32: bool = False
    vda_max_len: int = -1
    vda_max_res: int = -1


def run(cfg: PipelineConfig) -> Path:
    source_fps = probe_fps(cfg.input_video)
    source_duration = probe_duration(cfg.input_video)
    target_fps = cfg.target_fps or source_fps

    tmp_root = cfg.work_dir or Path(tempfile.mkdtemp(prefix="v2d_"))
    tmp_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[v2d] backend={cfg.backend}, source fps={source_fps:.3f}, "
        f"target fps={target_fps:.3f}"
    )

    if cfg.backend == "da3":
        depth_video = _run_da3_pipeline(cfg, tmp_root, target_fps)
    elif cfg.backend == "vda":
        from .vda_backend import run_vda
        depth_video = tmp_root / "depth_video.mp4"
        run_vda(cfg, depth_video, target_fps)
    else:
        raise ValueError(f"unknown backend: {cfg.backend!r} (choose da3 or vda)")

    cfg.output_video.parent.mkdir(parents=True, exist_ok=True)
    depth_dur = probe_duration(depth_video)
    print(
        f"[v2d] retiming depth ({depth_dur:.3f}s) to match source ({source_duration:.3f}s); "
        f"{'muxing audio' if cfg.keep_audio else 'no audio'}"
    )
    finalize_output(
        depth_video,
        cfg.input_video,
        cfg.output_video,
        with_audio=cfg.keep_audio,
        target_duration=source_duration,
    )

    if cfg.work_dir is None:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"[v2d] done → {cfg.output_video}")
    return cfg.output_video


def _run_da3_pipeline(cfg: PipelineConfig, tmp_root: Path, target_fps: float) -> Path:
    da3_dir = tmp_root / "da3"
    da3_dir.mkdir(parents=True, exist_ok=True)
    print(f"[v2d] DA3 sample fps={cfg.sample_fps}")

    _run_da3(cfg, da3_dir)
    depth_low = da3_dir / "depth_video.mp4"
    if not depth_low.exists():
        raise FileNotFoundError(f"DA3 did not produce {depth_low}")

    if cfg.interpolator == "none":
        print("[v2d] interpolation disabled (--no-interpolate)")
        return depth_low
    if cfg.interpolator == "rife":
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
        return depth_interp
    raise ValueError(f"unknown interpolator: {cfg.interpolator!r}")


def _depth_array(prediction) -> np.ndarray:
    d = prediction.depth
    if hasattr(d, "detach"):
        d = d.detach().cpu().numpy()
    return np.asarray(d, dtype=np.float32)


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

    print(f"[v2d] loading model {cfg.model_dir} on {cfg.device}")
    model = DepthAnything3.from_pretrained(cfg.model_dir).to(cfg.device)

    # Cap extraction at process_res to avoid full-res PNGs from high-res inputs;
    # DA3 would downscale to process_res internally anyway.
    print(f"[v2d] extracting frames at {cfg.sample_fps} fps, capped at {cfg.process_res}px long edge")
    image_files = extract_frames(
        cfg.input_video,
        export_dir / "input_images",
        cfg.sample_fps,
        max_long_edge=cfg.process_res,
    )
    n = len(image_files)

    plan = plan_chunks(n, cfg.chunk_size, cfg.chunk_overlap)

    if cfg.vram_check:
        max_chunk = max(e - s for s, e in plan)
        def _probe(n: int) -> None:
            model.inference(
                image=image_files[:n],
                export_dir=None,
                process_res=cfg.process_res,
            )
        probe_vram_or_die(
            device=cfg.device,
            n_total=max_chunk,
            run_probe=_probe,
            safety=cfg.vram_safety,
            item_name="frame",
            oom_hint=f"Lower --process-res (currently {cfg.process_res}) or pick a smaller model.",
            suggestion_flag="--chunk-size",
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
    write_depth_video(depth, out, fps=cfg.sample_fps, cmap=cfg.colormap, norm_mode=cfg.colormap_norm)
