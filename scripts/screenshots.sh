#!/usr/bin/env bash
# Regenerate docs/screenshots using the GUI's --screenshot mode.
# Needs a running display and these models installed (rembg models pull ...):
#   u2netp u2net isnet-general-use bria-rmbg-1.4 birefnet-general-lite birefnet-general
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="${BIN:-target/release/rembg}"
OUT=docs/screenshots
IMG="${HOME}/.cache/rembg/docimg"
mkdir -p "$OUT" "$IMG"
# Public domain / CC0 test photos from scikit-image
for f in astronaut chelsea coffee; do
  [ -f "$IMG/$f.png" ] || curl -sfL -o "$IMG/$f.png" \
    "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/$f.png"
done
shot() { local name=$1; shift; "$BIN" gui "$@" --size 1280x820 --screenshot "$OUT/$name.png"; }
shot remove      "$IMG/astronaut.png" -m birefnet-general
shot remove-dark "$IMG/coffee.png"    -m birefnet-general --bg dark
shot mask        "$IMG/astronaut.png" -m birefnet-general --mask
shot compare     "$IMG/astronaut.png" --tab compare \
                 --models u2netp,u2net,isnet-general-use,bria-rmbg-1.4,birefnet-general-lite,birefnet-general
shot split       "$IMG/astronaut.png" --tab compare --split u2netp,birefnet-general --bg dark
shot models      --tab models
if command -v pngquant >/dev/null; then pngquant --force --skip-if-larger --quality 80-95 --ext .png "$OUT"/*.png; fi
ls -lh "$OUT"
