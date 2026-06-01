from __future__ import annotations

import bisect
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import imageio
import numpy as np
import torch

# FILM uses grid_sample(padding_mode='border'), which MPS does not implement.
# Enable the standard CPU-fallback so the call doesn't crash on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


FILM_ALIGN = 64  # FILM's encoder requires HxW divisible by 64


def _pad_to_align(img: np.ndarray, align: int = FILM_ALIGN) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    pad_h = (align - h % align) % align
    pad_w = (align - w % align) % align
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="edge")
    crop = (top, left, top + h, left + w)
    return padded, crop


class FilmRunner:
    """Frame-Interpolation-for-Large-Motion via dajes' torchscript checkpoint.

    The checkpoint is a single ``film_net_fp{16,32}.pt`` file. We import nothing
    from the dajes repo — the JIT graph holds the full model.
    """

    def __init__(self, model_path: Path, device: str = "cpu", half: bool = False, verbose: bool = False):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"FILM checkpoint not found at {self.model_path}. Run `v2d setup --interpolator film`."
            )
        # FILM uses grid_sample(padding_mode='border') which MPS still does not
        # implement (and the standard CPU fallback can't catch it because the
        # call lives inside the TorchScript graph). Silently downgrade to CPU.
        if device == "mps":
            print("[v2d] FILM falls back to CPU on Apple Silicon (MPS lacks grid_sample border padding).")
            device = "cpu"
        self.device = device
        self.half = half
        self.verbose = verbose
        self._model = None

    def _load(self):
        if self._model is None:
            m = torch.jit.load(str(self.model_path), map_location="cpu")
            m.eval()
            if self.half and self.device != "cpu":
                m.half()
            else:
                m.float()
            self._model = m.to(self.device)
        return self._model

    def interpolate(self, src: Path, dst: Path, multi: int, log_path: Path | None = None) -> None:
        if multi < 2:
            shutil.copyfile(src, dst)
            return

        reader = imageio.get_reader(str(src))
        meta = reader.get_meta_data()
        in_fps = float(meta.get("fps", 24))
        frames = [np.asarray(f) for f in reader]
        reader.close()
        if len(frames) < 2:
            raise RuntimeError(f"need >= 2 frames in {src} for interpolation, got {len(frames)}")

        if self.verbose:
            print(f"[v2d] FILM: {len(frames)} frames @ {in_fps:.2f} fps → multi={multi} → {len(frames) * multi - (multi - 1)} frames @ {in_fps * multi:.2f} fps")

        n_inter = multi - 1
        out_frames: List[np.ndarray] = [frames[0]]
        for i in range(len(frames) - 1):
            inter = self._interpolate_pair(frames[i], frames[i + 1], n_inter)
            # `inter` includes the start+end frames; we already pushed frames[i]
            # via the previous step, and we don't want to duplicate it.
            out_frames.extend(inter[1:])

        out_fps = in_fps * multi
        dst.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(dst), fps=out_fps, codec="libx264", quality=8, macro_block_size=1,
        )
        try:
            for f in out_frames:
                writer.append_data(f)
        finally:
            writer.close()

    def _interpolate_pair(self, f0: np.ndarray, f1: np.ndarray, n_inter: int) -> List[np.ndarray]:
        model = self._load()
        f0p, crop = _pad_to_align(f0)
        f1p, _ = _pad_to_align(f1)
        y1, x1c, y2, x2c = crop

        def to_tensor(img: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            return t.to(self.device, dtype=torch.float16 if (self.half and self.device != "cpu") else torch.float32)

        results = [to_tensor(f0p), to_tensor(f1p)]
        idxes = [0, n_inter + 1]
        remains = list(range(1, n_inter + 1))
        splits = torch.linspace(0, 1, n_inter + 2)

        for _ in range(n_inter):
            starts = splits[idxes[:-1]]
            ends = splits[idxes[1:]]
            distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - 0.5).abs()
            mat = int(torch.argmin(distances).item())
            start_i, step = np.unravel_index(mat, distances.shape)
            end_i = start_i + 1
            x0 = results[start_i]
            x1 = results[end_i]
            dt_val = float(
                (splits[remains[step]] - splits[idxes[start_i]])
                / (splits[idxes[end_i]] - splits[idxes[start_i]])
            )
            dt = x0.new_full((1, 1), dt_val)
            with torch.no_grad():
                pred = model(x0, x1, dt).clamp(0, 1)
            ins = bisect.bisect_left(idxes, remains[step])
            idxes.insert(ins, remains[step])
            results.insert(ins, pred)
            del remains[step]

        out: List[np.ndarray] = []
        for t in results:
            arr = (t.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            out.append(arr[y1:y2, x1c:x2c].copy())
        return out
