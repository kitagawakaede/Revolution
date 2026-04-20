#!/usr/bin/env bash
# クロスプラットフォーム対応 labelme ランチャ (Linux / macOS)
#
# 使い方:
#   ./label.sh <image_dir> [labels_file]
#
# 例:
#   ./label.sh data/tiles
#   ./label.sh data/tiles labels_sample_detection.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <image_dir> [labels_file]"
  echo "  image_dir:    ラベル対象の画像ディレクトリ"
  echo "  labels_file:  ラベル定義 .txt（省略時は labels_sample_detection.txt）"
  exit 1
fi

IMAGE_DIR="$(cd "$1" && pwd)"       # macOS の realpath 非標準を避ける
LABELS_FILE="${2:-${SCRIPT_DIR}/labels_sample_detection.txt}"

if [ ! -d "$IMAGE_DIR" ]; then
  echo "Error: image_dir not found: $IMAGE_DIR"; exit 1
fi
if [ ! -f "$LABELS_FILE" ]; then
  echo "Error: labels_file not found: $LABELS_FILE"; exit 1
fi

echo "image_dir:  $IMAGE_DIR"
echo "labels:     $LABELS_FILE"
echo

cd "$SCRIPT_DIR"
CONFIG_ARGS=()
if [ -f "$SCRIPT_DIR/.labelmerc" ]; then
  CONFIG_ARGS=(--config "$SCRIPT_DIR/.labelmerc")
fi
pixi run labelme \
  "$IMAGE_DIR" \
  --labels "$LABELS_FILE" \
  --output "$IMAGE_DIR" \
  --nodata \
  --autosave \
  "${CONFIG_ARGS[@]}"
