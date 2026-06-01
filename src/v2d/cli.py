from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .bootstrap import (
    ensure_film,
    ensure_rife,
    film_is_installed,
    film_path_default,
    rife_dir_default,
    rife_is_installed,
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
    interpolator: str = typer.Option("rife", help="Which interpolator(s) to install: rife | film | both."),
    rife_dir: Optional[Path] = typer.Option(None, help="Where to clone Practical-RIFE. Defaults to cache."),
    film_model: Optional[Path] = typer.Option(None, help="Where to save the FILM .pt file. Defaults to cache."),
):
    """Install RIFE and/or FILM into the v2d cache."""
    interpolator = interpolator.lower()
    if interpolator not in {"rife", "film", "both"}:
        raise typer.BadParameter("interpolator must be one of: rife, film, both")
    if interpolator in {"rife", "both"}:
        path = ensure_rife(rife_dir, install=True)
        typer.echo(f"[v2d] RIFE checkout ready at {path}")
    if interpolator in {"film", "both"}:
        path = ensure_film(film_model, install=True)
        typer.echo(f"[v2d] FILM weights ready at {path}")


@app.command()
def convert(
    input_video: Optional[Path] = typer.Argument(
        None,
        help=f"Input video. Defaults to the bundled example ({DEFAULT_EXAMPLE.name}).",
    ),
    output_video: Path = typer.Option(Path("output.mp4"), "--output", "-o"),
    sample_fps: float = typer.Option(1.0, help="FPS sent to DA3. Lower = faster, less VRAM."),
    target_fps: Optional[float] = typer.Option(None, help="Output fps. Defaults to source fps."),
    model_dir: str = typer.Option(
        "depth-anything/DA3-LARGE-1.1",
        help="DA3 model id or local path. See the README for the full model list.",
    ),
    device: str = typer.Option("cuda", help="cuda | mps | cpu"),
    process_res: int = typer.Option(504, help="DA3 processing resolution."),
    interpolator: str = typer.Option(
        "rife",
        help="Frame interpolator: rife (default, fast, occasional hallucination) | film (better at large motion, slower).",
    ),
    rife_dir: Optional[Path] = typer.Option(
        None,
        help="Path to a Practical-RIFE checkout. Defaults to the cached one from `v2d setup`.",
    ),
    rife_python: Optional[str] = typer.Option(None, help="Python interpreter used to invoke RIFE."),
    rife_verbose: bool = typer.Option(False, "--rife-verbose", help="Stream RIFE's ffmpeg output instead of logging to a file."),
    film_model: Optional[Path] = typer.Option(
        None,
        help="Path to the FILM .pt file. Defaults to the cached one from `v2d setup --interpolator film`.",
    ),
    chunk_size: int = typer.Option(
        32,
        help="DA3 chunk size (frames per forward pass). Use a large value (or 0) to disable chunking.",
    ),
    chunk_overlap: int = typer.Option(
        8,
        help="Frames shared between consecutive chunks. Bigger = smoother seam, slower.",
    ),
    no_interpolate: bool = typer.Option(False, "--no-interpolate", help="Skip frame interpolation."),
    keep_audio: bool = typer.Option(True, help="Mux audio from the source video into the output."),
    work_dir: Optional[Path] = typer.Option(None, help="Persist intermediates here for debugging."),
):
    """Run the full DA3 → RIFE → mux pipeline."""
    if input_video is None:
        if not DEFAULT_EXAMPLE.exists():
            raise typer.BadParameter(
                f"No input_video given and bundled example not found at {DEFAULT_EXAMPLE}."
            )
        input_video = DEFAULT_EXAMPLE
        typer.echo(f"[v2d] using bundled example: {input_video}")

    interpolator = interpolator.lower()
    if interpolator not in {"rife", "film"}:
        raise typer.BadParameter("--interpolator must be one of: rife, film")

    resolved_rife: Optional[Path] = None
    resolved_film: Optional[Path] = None
    if no_interpolate:
        interpolator_final = "none"
    elif interpolator == "film":
        candidate = film_model or film_path_default()
        if not film_is_installed(candidate):
            raise typer.BadParameter(
                f"FILM weights not installed at {candidate}. "
                "Run `v2d setup --interpolator film` or pass --no-interpolate."
            )
        resolved_film = candidate
        interpolator_final = "film"
    else:  # rife
        candidate = rife_dir or rife_dir_default()
        if not rife_is_installed(candidate):
            raise typer.BadParameter(
                f"RIFE not installed at {candidate}. Run `v2d setup` or pass --no-interpolate."
            )
        resolved_rife = candidate
        interpolator_final = "rife"

    cfg = PipelineConfig(
        input_video=input_video,
        output_video=output_video,
        sample_fps=sample_fps,
        target_fps=target_fps,
        model_dir=model_dir,
        device=device,
        process_res=process_res,
        rife_dir=resolved_rife,
        rife_python=rife_python,
        rife_verbose=rife_verbose,
        interpolator=interpolator_final,
        film_model=resolved_film,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_audio=keep_audio,
        work_dir=work_dir,
    )
    run(cfg)


if __name__ == "__main__":
    app()
