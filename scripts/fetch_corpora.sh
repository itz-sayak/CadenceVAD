#!/usr/bin/env bash
# Fetch the public corpora used to retrain CadenceVAD.
#
# Speech, noise, and room impulse responses are downloaded from their canonical
# OpenSLR locations and extracted in place. Every archive is resumable, so the
# script can be re-run after an interruption without re-downloading what is
# already present.
#
# AVA-Speech is deliberately NOT fetched here: it is evaluation-only data and is
# prepared separately by scripts/prepare_ava_speech.py.
set -uo pipefail

ROOT="${1:-/home/seema/VAD-data}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"

mkdir -p "$ROOT/archives"

# name|url|destination directory
CORPORA=(
  "rirs|https://www.openslr.org/resources/28/rirs_noises.zip|rirs"
  "musan|https://www.openslr.org/resources/17/musan.tar.gz|musan"
  "librispeech-train-clean-100|https://www.openslr.org/resources/12/train-clean-100.tar.gz|librispeech"
  "librispeech-dev-clean|https://www.openslr.org/resources/12/dev-clean.tar.gz|librispeech"
  "librispeech-train-clean-360|https://www.openslr.org/resources/12/train-clean-360.tar.gz|librispeech"
)

free_gb() { df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'; }

for entry in "${CORPORA[@]}"; do
  IFS='|' read -r name url dest <<<"$entry"
  archive="$ROOT/archives/$(basename "$url")"
  marker="$ROOT/$dest/.$name.extracted"

  if [ -f "$marker" ]; then
    echo "[skip] $name already extracted"
    continue
  fi
  if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
    echo "[stop] only $(free_gb)G free, below the ${MIN_FREE_GB}G floor; skipping $name" >&2
    continue
  fi

  echo "[get ] $name -> $archive"
  if ! curl -fL --retry 8 --retry-delay 5 --retry-all-errors -C - -o "$archive" "$url"; then
    echo "[fail] download $name" >&2
    continue
  fi

  echo "[open] $name -> $ROOT/$dest"
  mkdir -p "$ROOT/$dest"
  case "$archive" in
    *.tar.gz) tar -xzf "$archive" -C "$ROOT/$dest" --strip-components=0 ;;
    *.zip)    unzip -q -o "$archive" -d "$ROOT/$dest" ;;
    *)        echo "[fail] unknown archive type $archive" >&2; continue ;;
  esac || { echo "[fail] extract $name" >&2; continue; }

  touch "$marker"
  rm -f "$archive"
  echo "[done] $name  ($(free_gb)G free)"
done

echo "[end ] corpora present under $ROOT:"
du -sh "$ROOT"/*/ 2>/dev/null
