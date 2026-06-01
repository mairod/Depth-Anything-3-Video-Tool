from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path


class RifeRunner:
    """Wrap a Practical-RIFE checkout (https://github.com/hzwer/Practical-RIFE)."""

    def __init__(self, rife_dir: Path, python: str = "python", verbose: bool = False):
        self.rife_dir = rife_dir
        self.script = rife_dir / "inference_video.py"
        if not self.script.exists():
            raise FileNotFoundError(
                f"{self.script} not found. Clone Practical-RIFE and pass --rife-dir."
            )
        self.python = python
        self.verbose = verbose

    def interpolate(self, src: Path, dst: Path, multi: int, log_path: Path | None = None) -> None:
        if multi < 2:
            shutil.copyfile(src, dst)
            return
        cmd = [
            self.python, str(self.script),
            "--multi", str(multi),
            "--video", str(src.resolve()),
        ]
        if self.verbose:
            subprocess.run(cmd, check=True, cwd=self.rife_dir)
        else:
            log_path = log_path or (src.parent / "rife.log")
            with open(log_path, "wb") as logf:
                result = subprocess.run(
                    cmd, cwd=self.rife_dir, stdout=logf, stderr=subprocess.STDOUT
                )
            if result.returncode != 0:
                # Surface tail of the log so the user can diagnose.
                tail = log_path.read_text(errors="replace").splitlines()[-30:]
                raise subprocess.CalledProcessError(
                    result.returncode, cmd,
                    output=("\n".join(tail) + f"\n(full log at {log_path})"),
                )
        produced = self._find_output(src, multi)
        produced.replace(dst)

    def _find_output(self, src: Path, multi: int) -> Path:
        # RIFE writes files like ``{stem}_{multi}X_{newfps}fps.mp4`` and, in
        # recent versions, ``{stem}_{multi}X_{newfps}fps_noaudio.mp4`` when the
        # source has no audio track (true for our depth-only intermediate).
        stem = src.stem
        pattern = f"{stem}_{multi}X_*"
        candidates = [
            p for p in src.parent.glob(pattern) if p != src and p.suffix.lower() in {".mp4", ".mov", ".mkv"}
        ]
        if not candidates:
            raise FileNotFoundError(
                f"RIFE produced no output for {src} (looked for {pattern})"
            )
        return max(candidates, key=lambda p: p.stat().st_mtime)


def choose_multiplier(source_fps: float, sample_fps: float) -> int:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")
    return max(1, math.ceil(source_fps / sample_fps))
