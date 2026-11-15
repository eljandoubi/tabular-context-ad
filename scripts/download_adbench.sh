#!/bin/sh
# Download the eight ADBench tables used in the study (about 0.6 MB) into data/adbench.
set -eu

COMMIT=3dac8221081e190f157d78e93bfa8867f90d0965
BASE="https://raw.githubusercontent.com/Minqi824/ADBench/$COMMIT/adbench/datasets/Classical"
DEST="${1:-data/adbench}"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$DEST"
for name in 6_cardio 38_thyroid 30_satellite 32_shuttle 29_Pima 42_WBC 40_vowels 4_breastw; do
    if [ ! -s "$DEST/$name.npz" ]; then
        curl -fsSL "$BASE/$name.npz" -o "$DEST/$name.npz"
    fi
done

cd "$DEST"
shasum -a 256 -c "$HERE/adbench.sha256"
