@echo off
cd /d "%~dp0"
where git || (echo ERROR: Git is not installed & exit /b 1)
if not exist .git git init
if not exist external\ultralytics\pyproject.toml (
 git submodule add https://github.com/ultralytics/ultralytics.git external/ultralytics || exit /b 1
)
if not exist external\mobile-yolo\README.md (
 git submodule add https://github.com/surendramaran/YOLO.git external/mobile-yolo || exit /b 1
)
git submodule update --init --recursive
git add .gitmodules external/ultralytics external/mobile-yolo
@echo READY. Commit both submodules with: git commit -m "Add YOLO and Android source references"
