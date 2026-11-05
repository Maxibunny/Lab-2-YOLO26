#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! test -d .git; then git init; fi
if ! test -f external/ultralytics/pyproject.toml; then git submodule add https://github.com/ultralytics/ultralytics.git external/ultralytics; fi
if ! test -f external/mobile-yolo/README.md; then git submodule add https://github.com/surendramaran/YOLO.git external/mobile-yolo; fi
git submodule update --init --recursive
git add .gitmodules external/ultralytics external/mobile-yolo
