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


# ---------------------------------------------------------------------------
# Video-Depth-Anything (VDA) — second backend, parallel to RIFE above.
# ---------------------------------------------------------------------------

VDA_REPO = "https://github.com/DepthAnything/Video-Depth-Anything.git"
VDA_HF_REPO = {
    "vits": "depth-anything/Video-Depth-Anything-Small",
    "vitb": "depth-anything/Video-Depth-Anything-Base",
    "vitl": "depth-anything/Video-Depth-Anything-Large",
}
VDA_CKPT_FILENAME = {
    "vits": "video_depth_anything_vits.pth",
    "vitb": "video_depth_anything_vitb.pth",
    "vitl": "video_depth_anything_vitl.pth",
}


def vda_dir_default() -> Path:
    return cache_root() / "Video-Depth-Anything"


def vda_ckpt_path(target: Path, encoder: str) -> Path:
    return target / "checkpoints" / VDA_CKPT_FILENAME[encoder]


def vda_repo_is_cloned(target: Path) -> bool:
    return (target / "video_depth_anything" / "__init__.py").exists()


def vda_is_installed(target: Path, encoder: str | None = None) -> bool:
    """True iff the repo is cloned and at least one checkpoint exists.

    When ``encoder`` is given, require that specific checkpoint. Otherwise
    accept any of the three.
    """
    if not vda_repo_is_cloned(target):
        return False
    if encoder is not None:
        return vda_ckpt_path(target, encoder).exists()
    return any(vda_ckpt_path(target, e).exists() for e in VDA_CKPT_FILENAME)


def clone_vda(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if (target / ".git").exists():
        print(f"[v2d] updating existing VDA checkout at {target}")
        subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=True)
    else:
        print(f"[v2d] cloning Video-Depth-Anything into {target}")
        subprocess.run(["git", "clone", "--depth=1", VDA_REPO, str(target)], check=True)


def download_vda_weights(target: Path, encoder: str) -> Path:
    if encoder not in VDA_HF_REPO:
        raise ValueError(f"unknown VDA encoder: {encoder!r} (choose vits, vitb, vitl)")
    from huggingface_hub import hf_hub_download

    ckpt_dir = target / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    filename = VDA_CKPT_FILENAME[encoder]
    print(f"[v2d] downloading {filename} from {VDA_HF_REPO[encoder]}")
    fetched = hf_hub_download(
        repo_id=VDA_HF_REPO[encoder],
        filename=filename,
        local_dir=str(ckpt_dir),
    )
    return Path(fetched)


def ensure_vda(
    target: Path | None = None,
    *,
    encoder: str = "vitl",
    install: bool = False,
) -> Path:
    path = target or vda_dir_default()
    if vda_is_installed(path, encoder=encoder):
        return path
    if not install:
        raise FileNotFoundError(
            f"VDA not found at {path} (or checkpoint for encoder={encoder!r} missing). "
            f"Run `v2d setup-vda --encoder {encoder}` or pass --vda-dir."
        )
    if not vda_repo_is_cloned(path):
        clone_vda(path)
    if not vda_ckpt_path(path, encoder).exists():
        download_vda_weights(path, encoder)
    return path
