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
make release          # CPU build -> target/release/rembg
make cuda             # CUDA build + bundles CUDA runtime libs in target/release/lib
make check            # fmt + clippy + tests
```

## CLI

```sh
rembg remove photo.jpg -o out/            # -> out/photo.png
rembg remove *.jpg -m birefnet-general --mask
rembg remove photo.jpg --bg '#ffffff' -o flat.jpg
rembg remove photo.jpg --copy             # result to clipboard
rembg compare photo.jpg -o grid.png
rembg models list | pull <id>|--all | rm <id>
rembg gui [image]
```
