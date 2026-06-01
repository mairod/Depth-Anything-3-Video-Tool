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
