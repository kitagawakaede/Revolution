@echo off
REM Windows 用 labelme ランチャ
REM 使い方: label.bat <image_dir> [labels_file]

setlocal enabledelayedexpansion

if "%~1"=="" (
  echo Usage: %~nx0 ^<image_dir^> [labels_file]
  echo   image_dir:    ラベル対象の画像ディレクトリ
  echo   labels_file:  ラベル定義 .txt^(省略時は labels_sample_detection.txt^)
  exit /b 1
)

set "SCRIPT_DIR=%~dp0"
set "IMAGE_DIR=%~f1"
if "%~2"=="" (
  set "LABELS_FILE=%SCRIPT_DIR%labels_sample_detection.txt"
) else (
  set "LABELS_FILE=%~f2"
)

if not exist "%IMAGE_DIR%" (
  echo Error: image_dir not found: %IMAGE_DIR%
  exit /b 1
)
if not exist "%LABELS_FILE%" (
  echo Error: labels_file not found: %LABELS_FILE%
  exit /b 1
)

echo image_dir:  %IMAGE_DIR%
echo labels:     %LABELS_FILE%
echo.

cd /d "%SCRIPT_DIR%"
pixi run labelme "%IMAGE_DIR%" --labels "%LABELS_FILE%" --output "%IMAGE_DIR%" --nodata --autosave
