from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .bootstrap import (
    ensure_rife,
    ensure_vda,
    rife_dir_default,
    rife_is_installed,
    vda_dir_default,
    vda_is_installed,
)
from .pipeline import PipelineConfig, run

app = typer.Typer(
    help="Convert any video to a depth-map video at the source frame rate.",
    add_completion=False,
    no_args_is_help=True,
)

# assets/examples/ sits at the repo root (works for editable installs).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLE = _REPO_ROOT / "assets" / "examples" / "robot_unitree.mp4"


@app.command()
def setup(
    rife_dir: Optional[Path] = typer.Option(None, help="Where to clone Practical-RIFE. Defaults to cache."),
):
    """Install RIFE into the v2d cache (needed for the DA3 backend)."""
    path = ensure_rife(rife_dir, install=True)
    typer.echo(f"[v2d] RIFE checkout ready at {path}")


@app.command("setup-vda")
def setup_vda(
    vda_dir: Optional[Path] = typer.Option(None, help="Where to clone Video-Depth-Anything. Defaults to cache."),
    encoder: str = typer.Option("vitl", help="Which encoder checkpoint to download: vits | vitb | vitl."),
):
    """Clone Video-Depth-Anything and download a checkpoint into the v2d cache."""
    path = ensure_vda(vda_dir, encoder=encoder, install=True)
    typer.echo(f"[v2d] VDA checkout ready at {path} (encoder={encoder})")


@app.command()
def convert(
    input_video: Optional[Path] = typer.Argument(
        None,
        help=f"Input video. Defaults to the bundled example ({DEFAULT_EXAMPLE.name}).",
    ),
    output_video: Path = typer.Option(Path("output.mp4"), "--output", "-o"),
    backend: str = typer.Option(
        "da3",
        help="Depth model backend: `da3` (Depth-Anything-3 + RIFE) or `vda` (Video-Depth-Anything, no RIFE).",
    ),
    sample_fps: float = typer.Option(1.0, help="[DA3 only] FPS sent to DA3. Lower = faster, less VRAM."),
    target_fps: Optional[float] = typer.Option(None, help="Output fps. Defaults to source fps."),
    model_dir: str = typer.Option(
        "depth-anything/DA3-LARGE-1.1",
        help="[DA3 only] DA3 model id or local path. See the README for the full model list.",
    ),
    device: str = typer.Option("cuda", help="cuda | mps | cpu"),
    process_res: int = typer.Option(504, help="[DA3 only] DA3 processing resolution."),
    rife_dir: Optional[Path] = typer.Option(
        None,
        help="[DA3 only] Path to a Practical-RIFE checkout. Defaults to the cached one from `v2d setup`.",
    ),
    rife_python: Optional[str] = typer.Option(None, help="[DA3 only] Python interpreter used to invoke RIFE."),
    rife_verbose: bool = typer.Option(False, "--rife-verbose", help="[DA3 only] Stream RIFE's ffmpeg output instead of logging to a file."),
    chunk_size: int = typer.Option(
        32,
        help="[DA3 only] DA3 chunk size (frames per forward pass). Use a large value (or 0) to disable chunking.",
    ),
    chunk_overlap: int = typer.Option(
        8,
        help="[DA3 only] Frames shared between consecutive chunks. Bigger = smoother seam, slower.",
    ),
    no_interpolate: bool = typer.Option(False, "--no-interpolate", help="[DA3 only] Skip frame interpolation. (VDA always skips RIFE.)"),
    keep_audio: bool = typer.Option(True, help="Mux audio from the source video into the output."),
    work_dir: Optional[Path] = typer.Option(None, help="Persist intermediates here for debugging."),
    colormap: str = typer.Option(
        "gray",
        help="Matplotlib colormap for depth rendering. e.g. gray, gray_r, Spectral, viridis, magma, inferno, plasma, turbo, jet.",
    ),
    vram_check: bool = typer.Option(
        True,
        "--vram-check/--no-vram-check",
        help="Probe VRAM with a tiny forward pass and abort if the configured chunk-size won't fit (CUDA only).",
    ),
    vram_safety: float = typer.Option(
        1.25,
        help="Safety multiplier on the estimated peak VRAM before comparing to free memory.",
    ),
    vda_dir: Optional[Path] = typer.Option(
        None,
        help="[VDA only] Path to a Video-Depth-Anything checkout. Defaults to the cached one from `v2d setup-vda`.",
    ),
    vda_encoder: str = typer.Option(
        "vitl",
        help="[VDA only] VDA encoder size: vits | vitb | vitl.",
    ),
    vda_input_size: int = typer.Option(
        518,
        help="[VDA only] VDA model input resolution (longest edge).",
    ),
    vda_fp32: bool = typer.Option(
        False,
        "--vda-fp32",
        help="[VDA only] Run VDA in fp32 instead of fp16. Slower, more VRAM.",
    ),
    vda_max_len: int = typer.Option(
        -1,
        help="[VDA only] Cap frames per internal VDA chunk (-1 = full video).",
    ),
    vda_max_res: int = typer.Option(
        -1,
        help="[VDA only] Cap frame read resolution (longest edge, -1 = no cap).",
    ),
):
    """Run the full backend → mux pipeline (default: DA3 + RIFE)."""
    if input_video is None:
        if not DEFAULT_EXAMPLE.exists():
            raise typer.BadParameter(
                f"No input_video given and bundled example not found at {DEFAULT_EXAMPLE}."
            )
        input_video = DEFAULT_EXAMPLE
        typer.echo(f"[v2d] using bundled example: {input_video}")

    backend = backend.lower()
    if backend not in {"da3", "vda"}:
        raise typer.BadParameter(f"unknown backend {backend!r} (choose da3 or vda)")

    resolved_rife: Optional[Path] = None
    interpolator_final = "none"
    resolved_vda: Optional[Path] = None

    if backend == "da3":
        if no_interpolate:
            interpolator_final = "none"
        else:
            candidate = rife_dir or rife_dir_default()
            if not rife_is_installed(candidate):
                raise typer.BadParameter(
                    f"RIFE not installed at {candidate}. Run `v2d setup` or pass --no-interpolate."
                )
            resolved_rife = candidate
            interpolator_final = "rife"
    else:  # vda
        candidate_vda = vda_dir or vda_dir_default()
        if not vda_is_installed(candidate_vda, encoder=vda_encoder):
            raise typer.BadParameter(
                f"VDA not installed at {candidate_vda} for encoder={vda_encoder!r}. "
                f"Run `v2d setup-vda --encoder {vda_encoder}`."
            )
        resolved_vda = candidate_vda

    cfg = PipelineConfig(
        input_video=input_video,
        output_video=output_video,
        backend=backend,
        sample_fps=sample_fps,
        target_fps=target_fps,
        model_dir=model_dir,
        device=device,
        process_res=process_res,
        rife_dir=resolved_rife,
        rife_python=rife_python,
        rife_verbose=rife_verbose,
        interpolator=interpolator_final,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_audio=keep_audio,
        work_dir=work_dir,
        colormap=colormap,
        vram_check=vram_check,
        vram_safety=vram_safety,
        vda_dir=resolved_vda,
        vda_encoder=vda_encoder,
        vda_input_size=vda_input_size,
        vda_fp32=vda_fp32,
        vda_max_len=vda_max_len,
        vda_max_res=vda_max_res,
    )
    run(cfg)


if __name__ == "__main__":
    app()
