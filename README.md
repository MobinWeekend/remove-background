# rembg (native)

Local, native background removal in Rust: a headless engine, a CLI and a native GUI.

## Layout

| crate | what |
|---|---|
| `crates/core` (`rembg-core`) | headless engine: model registry, download/cache (`~/.cache/rembg/models`, sha256-verified), pre/post-processing, ONNX Runtime inference (CPU, optional CUDA), compositing. No UI deps. |
| `crates/gui` (`rembg-gui`) | egui/eframe app: drop/paste → cut-out, preview backgrounds, clipboard, compare mode, model manager. |
| `crates/cli` (`rembg` binary) | `remove`, `compare`, `models`, `gui`. |

## Spec

- **Remove**: drop / open / paste (Ctrl+V) an image → background removed on a worker thread (UI never blocks).
- **Preview backgrounds**: checkerboard, dark, light, custom colour; zoom & pan.
- **Output**: copy result to clipboard (Ctrl+C, no file needed), save PNG/WebP, mask-only, or flattened onto a colour.
- **Compare**: run the image through every installed model → grid with model name + inference time, synced zoom/pan, cut-out ⇄ mask toggle; pick two for a side-by-side slider; pick a winner (copy/save/set default); save the whole grid as one image (`rembg compare` does the same headless).
- **Model manager**: all registry models with family, input size, download size, licence, installed state; download (with progress) / delete / set default.
- **Models**: U²-Net (u2net, u2netp, u2net_human_seg, silueta), ISNet (general, anime), BiRefNet (general, lite, portrait, DIS, HRSOD, massive), BRIA RMBG 1.4 / 2.0 (non-commercial licence — only downloaded on explicit request).
- **GPU**: `--features cuda` enables the CUDA execution provider; falls back to CPU if CUDA libs are missing.

## Build

```sh
make release          # CPU build -> target/release/rembg (ONNX Runtime linked statically, single file)
make cuda             # CUDA build + bundles the CUDA 13 runtime (cudart/cuBLAS/cuDNN/cuFFT, ~2 GB) in target/release/lib
make check            # fmt + clippy + tests
make bench IMG=x.jpg  # time every installed model
```

### GPU (CUDA)

- Built against ONNX Runtime 1.31 / CUDA 13 → needs an NVIDIA driver ≥ 580 and an Ampere-or-newer GPU.
- The binary re-execs itself with `<exe dir>/lib` on `LD_LIBRARY_PATH`, so ship `rembg`,
  `libonnxruntime_providers_{cuda,shared}.so` and `lib/` together (`make pull` on the client does this).
- `--device auto` (default) uses CUDA when it can and falls back to CPU if the CUDA libs are missing.
  If a model fails on the GPU at run time (e.g. out of VRAM), it is retried on the GPU after freeing other
  sessions, then on CPU. BiRefNet / BRIA 2.0 at 1024² need roughly 8 GB+ of VRAM.

| model (girl-1.jpg, 600×399) | 24-thread CPU | RTX 5070 Ti |
|---|---:|---:|
| isnet-general-use | 216 ms | 74 ms |
| birefnet-general-lite | 2920 ms | 139 ms |
| birefnet-general | 4653 ms | 287 ms |
| bria-rmbg-2.0 | 4536 ms | 267 ms |

## GUI

`rembg gui [image] [--tab remove|compare|models]`

| | |
|---|---|
| drop / Ctrl+V / Ctrl+O | load an image (paste accepts image data or a copied file path) |
| Ctrl+C / Ctrl+S | copy / save the transparent result |
| scroll · drag · double-click | zoom about cursor · pan · fit |
| hold Space | show the original |
| M / B | toggle mask view / cycle background (checker → dark → light → custom) |
| Ctrl+1/2/3 | Remove / Compare / Models tabs |

**Compare**: tiles share zoom/pan; click two tiles → *Side by side* slider; per tile *Use* (open in Remove),
📋 copy, ★ set default; *Save grid…* exports the labelled grid with the current background.

**Models**: install state, size, input resolution, licence (non-commercial highlighted), download with
progress / cancel / delete / sha256 verify, default model, device selection.

## CLI

```sh
rembg remove photo.jpg -o out/            # -> out/photo.png
rembg remove *.jpg -m birefnet-general --mask
rembg remove photo.jpg --bg '#ffffff' -o flat.jpg
rembg remove photo.jpg --copy             # result to clipboard only (no file)
rembg compare photo.jpg -o grid.png
rembg models list | pull <id>|--all | rm <id>
rembg gui [image]
```
