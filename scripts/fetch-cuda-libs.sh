#!/usr/bin/env bash
# Bundle the CUDA 13 runtime libraries needed by ONNX Runtime's CUDA EP into DEST
# (default target/release/lib), taken from NVIDIA's redistributable PyPI wheels.
#
# CUDA_SERIES pins toolkit-versioned packages to a minor series compatible with
# the oldest driver you target (13.0 => driver >= 580). Wheels are cached in
# ~/.cache/rembg/wheels.
set -euo pipefail
DEST="${1:-target/release/lib}"
CUDA_SERIES="${CUDA_SERIES:-13.0}"
CACHE="${HOME}/.cache/rembg/wheels"
mkdir -p "$DEST" "$CACHE"

python3 -I - "$DEST" "$CACHE" "$CUDA_SERIES" <<'PY'
import json, os, sys, urllib.request, zipfile, hashlib
dest, cache, series = sys.argv[1:4]
# (package, version prefix or None for latest)
pkgs = [
    ("nvidia-cuda-runtime", series),
    ("nvidia-cublas", series),
    ("nvidia-cufft", None),      # cuFFT 12.x ships with CUDA 13
    ("nvidia-curand", None),
    ("nvidia-cuda-nvrtc", series),
    ("nvidia-cudnn-cu13", None),
]
def vkey(v):
    return [int(x) if x.isdigit() else 0 for x in v.split(".")]
for name, prefix in pkgs:
    meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json"))
    versions = sorted((v for v, files in meta["releases"].items() if files and (prefix is None or v.startswith(prefix + "."))), key=vkey)
    if not versions:
        sys.exit(f"no {name} release matching {prefix}")
    ver = versions[-1]
    whl = next(f for f in meta["releases"][ver]
               if f["filename"].endswith(".whl") and "x86_64" in f["filename"] and "manylinux" in f["filename"])
    path = os.path.join(cache, whl["filename"])
    if not os.path.exists(path):
        print(f"  downloading {whl['filename']} ({whl['size']/1e6:.0f} MB)", flush=True)
        urllib.request.urlretrieve(whl["url"], path + ".part")
        h = hashlib.sha256(open(path + ".part", "rb").read()).hexdigest()
        if h != whl["digests"]["sha256"]:
            sys.exit(f"sha256 mismatch for {whl['filename']}")
        os.rename(path + ".part", path)
    n = 0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            base = os.path.basename(info.filename)
            if "/lib/" in info.filename and ".so" in base:
                with z.open(info) as src, open(os.path.join(dest, base), "wb") as out:
                    out.write(src.read())
                n += 1
    print(f"{name:22} {ver:14} {n} libs")
PY
du -sh "$DEST"
