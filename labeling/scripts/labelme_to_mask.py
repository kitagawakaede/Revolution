"""labelme JSON → (N+1) クラスマスク画像

対応 shape:
  - linestrip: 中心線 → cv2.polylines で幅 `rasterize_width_px` に太らせる
  - polygon:   cv2.fillPoly で領域塗りつぶし
  - rectangle: fillPoly で矩形塗りつぶし

クラス ID は config の classes の順（0=background）。

使い方:
  pixi run python scripts/labelme_to_mask.py \\
    --config configs/default.yaml \\
    --labeled-dir data/labeled \\
    --output-dir data/dataset
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from _common import load_config


def build_class_map(class_names: list[str]) -> dict[str, int]:
    """name → class id。0 は background"""
    return {n: i + 1 for i, n in enumerate(class_names)}


def render_mask(data: dict, class_map: dict[str, int], stroke_width: int,
                treat_polygon_as_linestrip: bool = False) -> np.ndarray:
    H, W = data["imageHeight"], data["imageWidth"]
    mask = np.zeros((H, W), dtype=np.uint8)
    # class id 昇順で描画（後に書いたものが上書き）
    sorted_labels = sorted(class_map.keys(), key=lambda n: class_map[n])
    for label in sorted_labels:
        cid = class_map[label]
        for s in data["shapes"]:
            if s["label"] != label:
                continue
            pts = np.array(s["points"], dtype=np.int32)
            if len(pts) < 2:
                continue
            shape_type = s.get("shape_type", "polygon")
            if shape_type == "linestrip" or shape_type == "line" or \
               (treat_polygon_as_linestrip and shape_type == "polygon"):
                cv2.polylines(mask, [pts], isClosed=False, color=cid,
                              thickness=stroke_width, lineType=cv2.LINE_8)
            elif shape_type == "polygon":
                cv2.fillPoly(mask, [pts], color=cid)
            elif shape_type == "rectangle":
                (x0, y0), (x1, y1) = s["points"]
                rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
                cv2.fillPoly(mask, [rect], color=cid)
    return mask


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    PALETTE = [
        (255, 255, 255),  # 0: bg = 白
        (255, 0, 0), (0, 100, 255), (0, 200, 0), (255, 128, 0),
        (200, 0, 200), (0, 200, 200), (200, 100, 0),
    ]
    vis = image.copy()
    for cid in range(1, mask.max() + 1):
        color = PALETTE[cid % len(PALETTE)]
        sel = mask == cid
        vis[sel] = (vis[sel] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--labeled-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    class_names = cfg["classes"]
    stroke_width = cfg.get("label", {}).get("rasterize_width_px", 3)
    treat_polygon_as_linestrip = cfg.get("label", {}).get("treat_polygon_as_linestrip", False)
    class_map = build_class_map(class_names)

    (args.output_dir / "images").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "overlay").mkdir(parents=True, exist_ok=True)

    total = 0
    per_class = {n: 0 for n in class_names}
    for jp in sorted(args.labeled_dir.glob("*.json")):
        data = json.loads(jp.read_text())
        if not data.get("shapes"):
            continue
        img_path = args.labeled_dir / (jp.stem + ".png")
        if not img_path.exists():
            print(f"  (skip: no image) {jp.stem}")
            continue
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = render_mask(data, class_map, stroke_width, treat_polygon_as_linestrip)

        Image.fromarray(image).save(args.output_dir / "images" / f"{jp.stem}.png")
        Image.fromarray(mask).save(args.output_dir / "masks" / f"{jp.stem}.png")
        Image.fromarray(overlay(image, mask)).save(args.output_dir / "overlay" / f"{jp.stem}.png")

        pix = {n: int((mask == class_map[n]).sum()) for n in class_names}
        for n, p in pix.items():
            per_class[n] += p
        print(f"  {jp.stem}: {pix}")
        total += 1

    print(f"\n変換タイル数: {total}")
    print(f"クラス別ピクセル合計: {per_class}")
    print(f"出力: {args.output_dir}/(images|masks|overlay)/")


if __name__ == "__main__":
    main()
