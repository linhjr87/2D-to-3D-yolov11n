#!/usr/bin/env bash
# Download the fpcad_v0 dataset from the private Hugging Face dataset repo and install it into data/.
#
#   scripts/fetch_dataset.sh [HF_REPO]        (default: linhjr87/2D-to-3D-yolov11n)
#
# Steps: hf download -> sha256sum -c -> tar -xf -> rewrite `path:` in D/W data.yaml -> check manifests.
# Environment:
#   FORCE=1         replace an existing data/fpcad_v0/{D,W}; without it the script refuses
#   HF_LOCAL_DIR    directory that already holds fpcad_v0.tar and fpcad_v0.tar.sha256: skip the download
#                   (used for offline copies and for testing the script)
#   HF_TOKEN        access token for the private repo (or run `hf auth login` once)
set -euo pipefail

HF_REPO="${1:-linhjr87/2D-to-3D-yolov11n}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_DIR/data"
OUT="$DATA_DIR/fpcad_v0"
DL_DIR="${HF_LOCAL_DIR:-$DATA_DIR/_hf}"
TAR="fpcad_v0.tar"

die() { echo "fetch_dataset: $*" >&2; exit 1; }

if { [ -e "$OUT/D" ] || [ -e "$OUT/W" ]; } && [ "${FORCE:-0}" != 1 ]; then
    die "$OUT/D or $OUT/W already exists; run with FORCE=1 to replace it"
fi

# 1. download (resumable; hf skips files that are already complete)
if [ -z "${HF_LOCAL_DIR:-}" ]; then
    if command -v hf >/dev/null 2>&1; then
        HF=(hf download "$HF_REPO" --repo-type dataset --local-dir "$DL_DIR")
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF=(huggingface-cli download "$HF_REPO" --repo-type dataset --local-dir "$DL_DIR")
    else
        die "the Hugging Face CLI is missing: pip install -U huggingface_hub (then 'hf auth login' for the private repo)"
    fi
    mkdir -p "$DL_DIR"
    "${HF[@]}"
fi
[ -f "$DL_DIR/$TAR" ] || die "$DL_DIR/$TAR not found"
[ -f "$DL_DIR/$TAR.sha256" ] || die "$DL_DIR/$TAR.sha256 not found"

# 2. checksum (the .sha256 file has the form '<hash>  fpcad_v0.tar', relative to its own directory)
(cd "$DL_DIR" && sha256sum -c "$TAR.sha256")

# 3. extract next to the final location (same filesystem, so the D/W hardlinks survive the move)
mkdir -p "$DATA_DIR"
TMP="$(mktemp -d "$DATA_DIR/.fetch_tmp.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
tar -xf "$DL_DIR/$TAR" -C "$TMP"
for d in D W manifests; do
    [ -d "$TMP/fpcad_v0/$d" ] || die "archive has no fpcad_v0/$d"
done

# 4. manifests: the committed copies must be byte-identical to the archive's (split reproducibility)
for f in "$TMP"/fpcad_v0/manifests/*; do
    name="$(basename "$f")"
    committed="$OUT/manifests/$name"
    if [ -f "$committed" ]; then
        cmp -s "$f" "$committed" || die "manifests/$name differs from the committed copy in the git checkout"
    fi
done
[ -f "$OUT/manifests/tiles.csv" ] || die "$OUT/manifests/tiles.csv is missing: is this a full git checkout?"

# 5. install (D and W only; the committed manifests stay as they are)
mkdir -p "$OUT"
for d in D W; do
    rm -rf "${OUT:?}/$d"
    mv "$TMP/fpcad_v0/$d" "$OUT/$d"
done

# 6. data.yaml carries the build machine's absolute path: point it at this checkout
for d in D W; do
    yaml="$OUT/$d/data.yaml"
    [ -f "$yaml" ] || die "$yaml missing"
    NEWPATH="$OUT/$d" awk '
        BEGIN { q = sprintf("%c", 39) }
        /^path:/ { p = ENVIRON["NEWPATH"]; gsub(q, q q, p); print "path: " q p q; next }
        { print }
    ' "$yaml" > "$yaml.tmp"
    mv "$yaml.tmp" "$yaml"
    grep -q '^path:' "$yaml" || die "no 'path:' line in $yaml"
done

echo "fetch_dataset: ok, dataset installed in $OUT"
