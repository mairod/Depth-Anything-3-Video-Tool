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


def probe_duration(video: Path) -> float:
    """Container duration in seconds via ffprobe."""
    require("ffprobe")
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            str(video),
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError as exc:
        raise RuntimeError(f"could not parse duration for {video}: {out!r}") from exc


def finalize_output(
    depth_video: Path,
    source_video: Path,
    out: Path,
    *,
    with_audio: bool,
    target_duration: float,
) -> None:
    """Retime depth_video to exactly target_duration (via setpts) and optionally
    mux the source audio track. Re-encodes video to H.264/yuv420p.

    Sampling at sample_fps quantizes the depth video's duration to the nearest
    1/sample_fps; RIFE then rounds again. Without retiming, depth drifts a few
    hundred ms from source, breaking sync. setpts scales every frame's PTS so
    the depth playhead matches the source playhead at every instant.
    """
    require("ffmpeg")
    in_dur = probe_duration(depth_video)
    if in_dur <= 0:
        raise RuntimeError(f"depth video {depth_video} has non-positive duration")
    vf = f"setpts=PTS*{target_duration}/{in_dur}"

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(depth_video)]
    if with_audio:
        cmd += ["-i", str(source_video)]
    cmd += [
        "-filter:v", vf,
        # Preserve setpts-modified timestamps exactly; without this ffmpeg
        # re-quantizes to a fixed output fps and we lose sub-frame precision
        # (drifts ~1 frame ≈ 30 ms at 30 fps).
        "-fps_mode", "passthrough", "-vsync", "passthrough",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "slow", "-crf", "18", "-movflags", "+faststart",
    ]
    if with_audio:
        cmd += ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-an"]
    cmd.append(str(out))
    subprocess.run(cmd, check=True)


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
