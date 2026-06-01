from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path


def require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise RuntimeError(f"{tool!r} not found in PATH")
    return path


def probe_fps(video: Path) -> float:
    require("ffprobe")
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,avg_frame_rate",
            "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)["streams"][0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        val = data.get(key, "0/0")
        try:
            f = float(Fraction(val))
            if f > 0:
                return f
        except (ZeroDivisionError, ValueError):
            continue
    raise RuntimeError(f"could not determine fps for {video}")


def extract_frames(
    video: Path,
    out_dir: Path,
    sample_fps: float,
    max_long_edge: int | None = None,
) -> list[str]:
    """Extract PNG frames at sample_fps, optionally capping the longest edge.

    Replaces DA3's VideoHandler.process — that one writes full-resolution PNGs
    even when DA3 immediately resizes them to process_res. Capping at extraction
    time avoids the disk/RAM blowup on high-res inputs.

    Returns sorted list of frame paths.
    """
    require("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)

    vf = [f"fps={sample_fps}"]
    if max_long_edge:
        # Scale longest edge to max_long_edge only when input is larger.
        # `-2` keeps aspect ratio and forces even dimensions.
        vf.append(
            f"scale='if(gt(iw,ih),min(iw\\,{max_long_edge}),-2)'"
            f":'if(gt(iw,ih),-2,min(ih\\,{max_long_edge}))'"
        )

    pattern = str(out_dir / "%06d.png")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(video),
            "-vf", ",".join(vf),
            "-start_number", "0",
            pattern,
        ],
        check=True,
    )
    frames = sorted(str(p) for p in out_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"ffmpeg extracted no frames from {video}")
    return frames


def mux_audio(silent_video: Path, audio_source: Path, out: Path) -> None:
    require("ffmpeg")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(silent_video),
            "-i", str(audio_source),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(out),
        ],
        check=True,
    )
