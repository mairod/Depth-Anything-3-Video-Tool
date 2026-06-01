# v2d — Video to Depth-Map Video

Convert any video into a depth-map video at the source frame rate.
Standalone wrapper around [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
(depth estimation) and [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
(frame interpolation).

```
input.mp4 ──► DA3 (sample at low fps) ──► depth_video.mp4 (low fps)
                                                │
                                                ▼
                                        RIFE interpolation
                                                │
                                                ▼
                              depth_video.mp4 at source fps + audio
```

Running DA3 at low `--sample-fps` (default 1) keeps inference cheap; RIFE
interpolates the depth maps back up to the source fps. Audio from the
source is muxed back in by default.

## Demo

Input ([`assets/examples/robot_unitree.mp4`](assets/examples/robot_unitree.mp4))
→ depth output ([`assets/examples/demo_robot_unitree.mp4`](assets/examples/demo_robot_unitree.mp4)),
produced with the default settings (`DA3-LARGE-1.1`, `--sample-fps 1`,
`--colormap gray`, RIFE interpolation back to source fps).

<table>
  <tr>
    <th>Input</th>
    <th>Depth output</th>
  </tr>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/9a5d7443-fd84-4240-a983-510114e83b04" controls width="400"></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/6778e7c6-63c4-4b1b-91b4-75fd9e561d27" controls width="400"></video>
    </td>
  </tr>
</table>

Reproduce:

```bash
v2d convert assets/examples/robot_unitree.mp4 -o demo_robot_unitree.mp4 --device mps
```

## Requirements

- Python 3.10 – 3.12
- `ffmpeg` and `ffprobe` on `PATH`
- A CUDA / MPS / CPU torch install (whatever DA3 supports on your hardware)

DA3 itself is pulled in automatically via `pip install`. Practical-RIFE is
cloned on demand by `v2d setup`.

## Install

```bash
git clone <this-repo>
cd 2026_Depth-Anything-3-Video-Tool
make install PY=python3.12          # creates .venv, installs v2d + DA3
source .venv/bin/activate

# one-time: clone Practical-RIFE and download its weights into the v2d cache
v2d setup
# (weights are pulled from Google Drive via gdown; if that fails the command
# prints a manual download link)
```

The cache lives at `~/.cache/v2d/Practical-RIFE` (override with
`V2D_CACHE_DIR`).

> **Why a Makefile and not just `pip install -e .`?** Upstream DA3 lists
> `xformers` and `open3d` as required dependencies; neither ships pre-built
> wheels for macOS arm64 and both fail to compile from source on Apple
> Silicon. Both are optional at runtime for the inference path we use
> (`xformers` is wrapped in a try/except SwiGLU fallback, `open3d` is only
> imported by the benchmarking code). `make install` therefore installs
> v2d's listed deps first, then installs DA3 itself with `--no-deps`. On
> Linux/CUDA you can additionally `pip install xformers open3d` afterwards
> if you want them.

## Usage

Quick test with the bundled example (`assets/examples/robot_unitree.mp4`):

```bash
v2d convert -o depth.mp4 --device mps   # or cuda / cpu
```

Your own video:

```bash
v2d convert input.mp4 -o depth.mp4 --sample-fps 1 --device cuda
```

> **macOS note:** if you hit `OMP: Error #15: Initializing libomp.dylib,
> but found libomp.dylib already initialized.`, prefix the command with
> `KMP_DUPLICATE_LIB_OK=TRUE`, e.g.
> `KMP_DUPLICATE_LIB_OK=TRUE v2d convert -o depth.mp4 --device mps`.
> This happens when torch and another OpenMP-linked library coexist in
> the same venv (common with the DA3 + ffmpeg combo on Apple Silicon).
> Setting it permanently in your shell rc is fine.

Apple Silicon:

```bash
v2d convert input.mp4 -o depth.mp4 --device mps
```

Without interpolation (depth video stays at `--sample-fps`):

```bash
v2d convert input.mp4 -o depth.mp4 --no-interpolate
```

Use a smaller, permissively-licensed model:

```bash
v2d convert input.mp4 -o depth.mp4 --model-dir depth-anything/DA3-BASE
```

## Chunked inference (default)

DA3 stacks all sampled frames into a single forward pass; its cross-view
attention is roughly quadratic in frame count, so long videos at moderate
`--sample-fps` OOM quickly and the largest models compound the problem.

`v2d` therefore chunks the frame sequence by default:

- `--chunk-size 32` frames per DA3 forward pass
- `--chunk-overlap 8` shared frames between consecutive chunks
- A scalar least-squares fit on the overlap rescales each new chunk to the
  running depth scale (handles per-chunk scale drift, which can happen even
  with Nested models because the metric scale_factor is fit per inference
  call)
- A linear crossfade across the overlap removes the seam

When the total frame count is `<= chunk_size`, chunking is a no-op.

The win: you can push `--sample-fps` up (less work for RIFE and far
fewer hallucinations) and handle long videos without OOM. Pass `--chunk-size 0`
or a value larger than the total frame count to force single-pass inference.

> **Note**: chunk boundaries break DA3's cross-view attention at the seam,
> so very narrow `--chunk-overlap` can produce subtle flicker at the join.
> The default of 8 hides this in most cases; bump it to 12-16 for picky
> content or fast motion.

## Frame interpolator

[Practical-RIFE](https://github.com/hzwer/Practical-RIFE) is used to
upsample DA3's low-fps depth output to the target frame rate. It is fast,
GPU-friendly on all platforms, and low memory. At very high multipliers
(when `--sample-fps` is much lower than the source) it can hallucinate.

Install with:

```bash
v2d setup   # clones Practical-RIFE + downloads v4.26 weights
```

## Options (convert)

| Flag | Default | Notes |
|------|---------|-------|
| `--output / -o` | `output.mp4` | Final muxed file |
| `--sample-fps` | `1.0` | Frame rate sent to DA3 |
| `--target-fps` | source fps | Output fps after interpolation |
| `--model-dir` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | Any DA3 model id / local path |
| `--device` | `cuda` | `cuda`, `mps`, or `cpu` |
| `--process-res` | `504` | DA3 processing resolution. Also caps frame extraction (longest edge) and the depth output resolution. Must be a multiple of 14 (DA3 patch size). |
| `--chunk-size` | `32` | Frames per DA3 forward pass (0 = single-shot) |
| `--chunk-overlap` | `8` | Frames shared between consecutive chunks |
| `--rife-dir` | cache | Override the Practical-RIFE checkout location |
| `--rife-python` | current interpreter | Python used to invoke RIFE |
| `--no-interpolate` | off | Skip interpolation, output at `--sample-fps` |
| `--keep-audio / --no-keep-audio` | keep | Mux source audio into output |
| `--work-dir` | tempdir | Keep intermediates here (extracted frames, low-fps depth) |
| `--colormap` | `gray` | Matplotlib colormap for depth rendering (see below) |

## Processing resolution

`--process-res` controls three things at once:

1. **Frame extraction**: ffmpeg downscales the longest edge of each extracted
   frame to `process_res` (no-op if the input is already smaller). Avoids
   writing full-resolution PNGs to disk only for DA3 to immediately resize
   them.
2. **DA3 inference**: longest edge clamped to `process_res`, then rounded to a
   multiple of 14 (DA3's ViT patch size).
3. **Output depth video**: written at the same processed resolution. The
   depth video does **not** match the input resolution unless `process_res`
   ≥ the input's longest edge.

Useful values (all multiples of 14):

| `--process-res` | Long edge | When |
|----------------:|-----------|------|
| `252` | 252 px | Toy / very fast smoke tests |
| `392` | 392 px | Fast, some loss on thin structures |
| **`504`** | 504 px | DA3 default — best quality/VRAM trade-off |
| `700` | 700 px | Sharper edges, ~2× VRAM |
| `1008` | 1008 px | Max practical, ~4× VRAM |

DA3 attention is roughly quadratic in resolution, so doubling `process_res`
quadruples VRAM use. Depth maps are smooth, so going above `504` rarely buys
visible quality — bump it only for inputs with very thin/small structures.

## Colormaps

Pass any [matplotlib colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
name to `--colormap`. DA3 inverts depth internally (1/z), so **bright = near, dark = far**
(unless you use a reversed `_r` variant, which flips that).

The most useful ones for depth videos:

| Colormap | Type | Look | When to use |
|----------|------|------|-------------|
| **`gray`** (default) | sequential, grayscale | black far → white near | Cleanest, no chroma noise; best for downstream ML (e.g. ControlNet depth, geometry passes) or stereo/3D tools that expect luminance depth |
| `gray_r` | sequential, grayscale | white far → black near | Reversed — matches the convention used by some depth-from-stereo tools |
| `Spectral` | diverging, rainbow | blue far → yellow → red near | DA3's original default; high contrast, eye-friendly for human inspection |
| `viridis` | sequential, perceptual | dark purple → green → yellow | Perceptually uniform, colorblind-safe, good for quantitative review |
| `magma` | sequential, perceptual | black → purple → orange → white | Similar to viridis but warmer; great for dark scenes |
| `inferno` | sequential, perceptual | black → red → yellow | Higher contrast than magma; pops on dark backgrounds |
| `plasma` | sequential, perceptual | purple → pink → yellow | Vibrant, perceptually uniform alternative to viridis |
| `turbo` | sequential, rainbow | blue → green → red | High-contrast rainbow; popular in robotics/SLAM viewers |
| `jet` | sequential, rainbow | blue → cyan → yellow → red | Classic MATLAB rainbow; perceptually non-uniform — avoid for analysis but familiar to many |

Append `_r` to any name to reverse it (`viridis_r`, `magma_r`, `turbo_r`, …).

```bash
v2d convert input.mp4 -o depth.mp4 --colormap viridis
v2d convert input.mp4 -o depth.mp4 --colormap gray_r
```

> **Tip:** if you're feeding the depth video into another model (ControlNet,
> NeRF preprocessing, stereo conversion tools), keep the default `gray` —
> color colormaps embed three near-identical channels and waste bitrate
> while distorting the underlying scalar.

## DA3 models

Pass any of these to `--model-dir`. The default is **`DA3-LARGE-1.1`** —
good quality, moderate VRAM, no Mac-arm64 setup pain.

### Any-view series (relative depth + camera poses)

| Model | Params | License | Notes |
|-------|--------|---------|-------|
| `depth-anything/DA3-SMALL` | 0.08 B | Apache 2.0 | Fastest, commercial-friendly |
| `depth-anything/DA3-BASE` | 0.12 B | Apache 2.0 | Light, commercial-friendly |
| `depth-anything/DA3-LARGE` | 0.35 B | CC BY-NC 4.0 | Deprecated, prefer `-1.1` |
| `depth-anything/DA3-LARGE-1.1` | 0.35 B | CC BY-NC 4.0 | **Default**, balanced |
| `depth-anything/DA3-GIANT` | 1.15 B | CC BY-NC 4.0 | Deprecated, prefer `-1.1` |
| `depth-anything/DA3-GIANT-1.1` | 1.15 B | CC BY-NC 4.0 | Best any-view depth + pose |

### Nested series (any-view + metric scaling → meters)

| Model | Params | License | Notes |
|-------|--------|---------|-------|
| `depth-anything/DA3NESTED-GIANT-LARGE` | 1.40 B | CC BY-NC 4.0 | Deprecated, prefer `-1.1` |
| `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | 1.40 B | CC BY-NC 4.0 | Outputs metric depth in meters |

### Monocular-only variants

| Model | Params | License | Notes |
|-------|--------|---------|-------|
| `depth-anything/DA3METRIC-LARGE` | 0.35 B | Apache 2.0 | Mono metric depth, sky segmentation |
| `depth-anything/DA3MONO-LARGE` | 0.35 B | Apache 2.0 | Mono relative depth (no camera, no multi-view) |

`-1.1` suffix = retrained after a training bug fix; always prefer those
over the non-suffixed versions.

> **License caveat:** anything CC BY-NC 4.0 is **non-commercial only**.
> For commercial work, stick to `DA3-SMALL`, `DA3-BASE`, `DA3METRIC-LARGE`,
> or `DA3MONO-LARGE`.

> **VRAM:** DA3's cross-view attention is quadratic in frame count, so
> larger models also dominate memory at high `--sample-fps`. Rough ballpark
> on Apple Silicon at `--process-res 504`, 20 frames: SMALL ≈ 2 GB,
> BASE ≈ 3 GB, LARGE ≈ 6 GB, GIANT ≈ 14 GB, NESTED-GIANT-LARGE ≈ 18 GB.

## macOS / MPS notes

- **torch is pinned to `<2.7` on macOS** because torch 2.7 – 2.12 silently
  break DA3-LARGE (and presumably larger) on MPS: attention layers
  collapse and depth comes out a single flat color. torch 2.6.x is the
  last version that produces correct depth on Apple Silicon. Linux/CUDA
  installs are unaffected and pull the latest torch.
- v2d prints depth-tensor stats after inference. If you see
  `min == max` (constant tensor) or non-zero `nan`/`inf` percentages,
  the model output is bad — usually a backend-specific bug. Try
  `--device cpu` (slow but always correct) or switch to a smaller model
  (`DA3-BASE` / `DA3-SMALL`).

## Known limitations

- **VRAM**: DA3's cross-view attention is quadratic in frame count. Long
  videos at high `--process-res` will OOM. Drop `--process-res`, lower
  `--sample-fps`, or split the input.
- **Depth-edge ghosting**: RIFE is trained on RGB; on hard depth
  discontinuities it can produce mild ghosting.
- **No chunking / resume**: re-running re-extracts everything. Pass
  `--work-dir` to keep intermediates between runs.

## License

Wrapper code is Apache-2.0. DA3 and Practical-RIFE keep their own
licenses — see their repos.
