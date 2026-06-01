#!/usr/bin/env bash
set -euo pipefail

# Merge directories named:
#   *_twfe, *_twyfe, *_twfe_*, *_twyfe_*
# into the corresponding directory name with the tag removed.

find . -mindepth 1 -type d | while IFS= read -r src; do
  base=$(basename "$src")
  parent=$(dirname "$src")

  # Remove _twfe or _twyfe as a path-name tag
  dest_base=$(printf '%s\n' "$base" | sed -E 's/_tw(y)?fe//g')

  # Skip if name did not change
  [ "$base" = "$dest_base" ] && continue

  dest="$parent/$dest_base"

  mkdir -p "$dest"

  # Merge contents, preserving files and subdirs.
  # Existing files with the same name are not overwritten.
  shopt -s dotglob nullglob
  for item in "$src"/*; do
    mv -n "$item" "$dest"/
  done
  shopt -u dotglob nullglob

  rmdir "$src" 2>/dev/null || true
done