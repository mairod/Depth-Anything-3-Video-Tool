"""Video-Depth-Anything backend.

VDA isn't a pip package: it's a git repo with `video_depth_anything/` and
`utils/` Python packages. We clone it into the v2d cache (see bootstrap.py)
and add that directory to ``sys.path`` at runtime — no install step on the
package itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ._vram import probe_vram_or_die
from .depth_video import write_depth_video


# Matches VDA's run.py model_configs.
_VDA_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _add_vda_to_syspath(vda_dir: Path) -> None:
    p = str(vda_dir.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_model(vda_dir: Path, encoder: str, ckpt: Path, device: str):
    import torch
    from video_depth_anything.video_depth import VideoDepthAnything

    if encoder not in _VDA_MODEL_CONFIGS:
        raise ValueError(f"unknown VDA encoder: {encoder!r} (vits, vitb, vitl)")

    print(f"[v2d] loading VDA-{encoder} from {ckpt} on {device}")
    model = VideoDepthAnything(**_VDA_MODEL_CONFIGS[encoder])
    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def run_vda(cfg, out_video: Path, target_fps: float) -> None:
    """Read source frames, run VDA, write a depth video at the model's output fps.

    The depth video duration may differ from source by a frame or two; the
    pipeline's finalize_output() retimes to exact source duration during mux.
    """
    from .bootstrap import vda_ckpt_path, vda_dir_default, vda_is_installed

    vda_dir = cfg.vda_dir or vda_dir_default()
    if not vda_is_installed(vda_dir, encoder=cfg.vda_encoder):
        raise FileNotFoundError(
            f"VDA not installed at {vda_dir} for encoder={cfg.vda_encoder!r}. "
            f"Run `v2d setup-vda --encoder {cfg.vda_encoder}` first."
        )
    _add_vda_to_syspath(vda_dir)
    ckpt = vda_ckpt_path(vda_dir, cfg.vda_encoder)

    model = _load_model(vda_dir, cfg.vda_encoder, ckpt, cfg.device)

    from utils.dc_utils import read_video_frames

    print(
        f"[v2d] reading frames from {cfg.input_video} "
        f"(target_fps={target_fps}, max_len={cfg.vda_max_len}, max_res={cfg.vda_max_res})"
    )
    frames, used_fps = read_video_frames(
        str(cfg.input_video),
        cfg.vda_max_len,
        target_fps,
        cfg.vda_max_res,
    )
    n_frames = len(frames)
    if n_frames == 0:
        raise RuntimeError(f"read_video_frames returned 0 frames for {cfg.input_video}")
    print(f"[v2d] {n_frames} frames at {used_fps:.3f} fps; running VDA-{cfg.vda_encoder}")

    if cfg.vram_check:
        # Effective chunk: VDA processes up to max_len frames at a time
        # internally (default -1 = full video).
        chunk_n = n_frames if cfg.vda_max_len <= 0 else min(cfg.vda_max_len, n_frames)

        def _probe(n: int) -> None:
            model.infer_video_depth(
                frames[:n], used_fps,
                input_size=cfg.vda_input_size,
                device=cfg.device,
                fp32=cfg.vda_fp32,
            )

        probe_vram_or_die(
            device=cfg.device,
            n_total=chunk_n,
            run_probe=_probe,
            safety=cfg.vram_safety,
            item_name="frame",
            oom_hint=(
                f"Lower --vda-input-size (currently {cfg.vda_input_size}), "
                f"set --vda-max-len, or pick a smaller --vda-encoder."
            ),
            suggestion_flag="--vda-max-len",
        )

    depths, out_fps = model.infer_video_depth(
        frames, used_fps,
        input_size=cfg.vda_input_size,
        device=cfg.device,
        fp32=cfg.vda_fp32,
    )
    depths = np.asarray(depths, dtype=np.float32)

    print(f"[v2d] writing depth video to {out_video} at {out_fps:.3f} fps")
    write_depth_video(depths, out_video, fps=out_fps, cmap=cfg.colormap)
