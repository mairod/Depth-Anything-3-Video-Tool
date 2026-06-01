from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

RIFE_REPO = "https://github.com/hzwer/Practical-RIFE.git"
RIFE_WEIGHTS_GDRIVE_ID = "1gViYvvQrtETBgU1w8axZSsr7YUuw31uy"  # v4.26 standard
RIFE_WEIGHTS_VERSION = "v4.26"


def cache_root() -> Path:
    root = os.environ.get("V2D_CACHE_DIR")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".cache" / "v2d"


def rife_dir_default() -> Path:
    return cache_root() / "Practical-RIFE"


def rife_is_installed(path: Path) -> bool:
    return (path / "inference_video.py").exists() and (path / "train_log").is_dir()


def clone_rife(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if (target / ".git").exists():
        print(f"[v2d] updating existing RIFE checkout at {target}")
        subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=True)
    else:
        print(f"[v2d] cloning Practical-RIFE into {target}")
        subprocess.run(["git", "clone", "--depth=1", RIFE_REPO, str(target)], check=True)


def print_weights_instructions(target: Path) -> None:
    print()
    print("=" * 70)
    print("Automatic RIFE weights download failed. Grab them manually:")
    print()
    print("  1. Open https://github.com/hzwer/Practical-RIFE#usage")
    print(f"  2. Download the {RIFE_WEIGHTS_VERSION} (or newer) zip.")
    print(f"  3. Unzip the `train_log/` folder into:")
    print(f"        {target}/train_log/")
    print("=" * 70)


def download_rife_weights(target: Path) -> bool:
    try:
        import gdown
    except ImportError:
        print("[v2d] gdown not installed; skipping automatic weights download.")
        return False
    zip_path = target / f"rife_{RIFE_WEIGHTS_VERSION}.zip"
    url = f"https://drive.google.com/uc?id={RIFE_WEIGHTS_GDRIVE_ID}"
    print(f"[v2d] downloading RIFE {RIFE_WEIGHTS_VERSION} weights from Google Drive")
    try:
        gdown.download(url, str(zip_path), quiet=False)
    except Exception as e:  # network / quota errors
        print(f"[v2d] gdown failed: {e}")
        return False
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        return False
    print(f"[v2d] extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    zip_path.unlink(missing_ok=True)
    # Some zips ship a __MACOSX/ sidecar — clean it.
    macosx = target / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx, ignore_errors=True)
    return (target / "train_log").is_dir()


def ensure_rife(target: Path | None = None, *, install: bool = False) -> Path:
    path = target or rife_dir_default()
    if rife_is_installed(path):
        return path
    if not install:
        raise FileNotFoundError(
            f"RIFE not found at {path}. Run `v2d setup` first or pass --rife-dir."
        )
    clone_rife(path)
    if not (path / "train_log").is_dir():
        if not download_rife_weights(path):
            print_weights_instructions(path)
    return path
