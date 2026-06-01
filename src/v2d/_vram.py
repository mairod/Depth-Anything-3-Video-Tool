"""Shared CUDA VRAM probe used by every inference backend.

Runs a tiny forward pass with the caller-supplied closure, measures peak
allocator usage, extrapolates linearly to the planned full-run size, and
aborts (with a sized suggestion) if it would exceed free VRAM.

Skips silently on non-CUDA devices so MPS/CPU users see no overhead.
"""
from __future__ import annotations

from typing import Callable


def fmt_gib(n_bytes: float) -> str:
    return f"{n_bytes / (1024 ** 3):.2f} GiB"


def probe_vram_or_die(
    *,
    device: str,
    n_total: int,
    run_probe: Callable[[int], None],
    safety: float = 1.25,
    item_name: str = "frame",
    oom_hint: str = "",
    suggestion_flag: str = "--chunk-size",
    bypass_flag: str = "--no-vram-check",
) -> None:
    if not device.startswith("cuda"):
        return

    import torch

    if not torch.cuda.is_available():
        return

    dev = torch.device(device)
    probe_n = min(4, n_total)
    if probe_n <= 0:
        return

    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)

    print(f"[v2d] probing VRAM with a {probe_n}-{item_name} forward pass…")
    try:
        with torch.inference_mode():
            run_probe(probe_n)
        torch.cuda.synchronize(dev)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        msg = f"[v2d] VRAM probe OOM at {probe_n} {item_name}s."
        if oom_hint:
            msg += " " + oom_hint
        raise RuntimeError(msg) from exc

    peak = torch.cuda.max_memory_allocated(dev)
    activations = max(peak - baseline, 0)
    per_item = activations / probe_n
    estimated = baseline + per_item * n_total
    needed = estimated * safety

    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(dev)

    print(
        f"[v2d] VRAM probe: baseline={fmt_gib(baseline)}, "
        f"per-{item_name}≈{fmt_gib(per_item)}, "
        f"est. peak for {n_total}-{item_name} run={fmt_gib(estimated)} "
        f"(×{safety:g} safety = {fmt_gib(needed)}); "
        f"free={fmt_gib(free)} / total={fmt_gib(total)}"
    )

    if needed <= free:
        return

    headroom = free / safety - baseline
    if per_item > 0 and headroom > 0:
        safe_n = int(headroom / per_item)
    else:
        safe_n = 0

    msg = (
        f"[v2d] insufficient VRAM: need ≈{fmt_gib(needed)} for {n_total} {item_name}s, "
        f"only {fmt_gib(free)} free.\n"
    )
    if safe_n >= 1:
        msg += f"  → try {suggestion_flag} {safe_n} (or use a smaller model / lower resolution)."
    else:
        msg += "  → use a smaller model or lower the processing resolution."
    msg += f"\n  (bypass this check with {bypass_flag})"
    raise RuntimeError(msg)
