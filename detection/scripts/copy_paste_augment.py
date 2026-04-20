"""Copy-paste augmentation でラベル済みタイルから学習データを増幅

前提（v02b kawasaki で実証済み）:
  - 対象物の形はタイル間で概ね固定
  - 変動は背景とノイズ
  → Copy-paste + ノイズ注入で rare class を増幅できる

入力:
  - labeled_dir/ 配下の画像 + labelme JSON（rectangle shape）
  - 同 dir の未アノテ画像を背景プールに
出力:
  - output_dir/tiles/*.png        合成タイル
  - output_dir/annotations.json   COCO 形式
  - output_dir/vis/*.png          冒頭 N 枚の可視化

使い方:
  pixi run python scripts/copy_paste_augment.py \\
    --config configs/default.yaml \\
    --labeled-dir data/train/images \\
    --output-dir data/train/synth
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _common import load_config


def load_manual_annotations(tiles_dir: Path, class_names: list[str]) -> tuple[dict, list[Path]]:
    patches = defaultdict(list)
    labeled = set()
    for jf in sorted(tiles_dir.glob("*.json")):
        data = json.loads(jf.read_text())
        tile_path = tiles_dir / data["imagePath"]
        if not tile_path.exists():
            continue
        labeled.add(tile_path)
        for sh in data.get("shapes", []):
            if sh.get("shape_type") != "rectangle":
                continue
            cls = sh["label"]
            if cls not in class_names:
                continue
            (x0, y0), (x1, y1) = sh["points"]
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])
            patches[cls].append({"tile": tile_path, "bbox": [x0, y0, x1, y1]})
    return patches, sorted(labeled)


def extract_patch(info: dict, margin: int) -> Image.Image:
    img = Image.open(info["tile"]).convert("L")
    x0, y0, x1, y1 = info["bbox"]
    x0 = max(0, int(x0) - margin)
    y0 = max(0, int(y0) - margin)
    x1 = min(img.width, int(x1) + margin)
    y1 = min(img.height, int(y1) + margin)
    return img.crop((x0, y0, x1, y1))


def paste_multiply(bg: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    h, w = patch.shape
    np.minimum(bg[y:y + h, x:x + w], patch, out=bg[y:y + h, x:x + w])


def bbox_iou(a, b) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def try_place(w: int, h: int, placed: list, tile_size: int, iou_margin: int,
              max_tries: int = 30) -> tuple[int, int] | None:
    for _ in range(max_tries):
        x = random.randint(0, tile_size - w)
        y = random.randint(0, tile_size - h)
        candidate = (x, y, x + w, y + h)
        exp = (candidate[0] - iou_margin, candidate[1] - iou_margin,
               candidate[2] + iou_margin, candidate[3] + iou_margin)
        if all(bbox_iou(exp, p) <= 0.0 for p in placed):
            return x, y
    return None


def inject_noise(canvas: Image.Image, font, tile_size: int, cfg_noise: dict) -> None:
    draw = ImageDraw.Draw(canvas)
    if cfg_noise.get("add_random_lines", True):
        for _ in range(random.randint(3, 10)):
            x0 = random.randint(0, tile_size); y0 = random.randint(0, tile_size)
            angle = random.uniform(0, 2 * math.pi)
            length = random.randint(40, 300)
            x1 = int(x0 + math.cos(angle) * length)
            y1 = int(y0 + math.sin(angle) * length)
            draw.line([(x0, y0), (x1, y1)], fill=random.randint(80, 180), width=1)
    if cfg_noise.get("add_small_text", True) and font is not None:
        texts = cfg_noise.get("candidate_texts", ["D", "R", "50", "80"])
        for _ in range(random.randint(5, 15)):
            x = random.randint(10, tile_size - 40)
            y = random.randint(10, tile_size - 20)
            draw.text((x, y), random.choice(texts), fill=random.randint(40, 120), font=font)


def build_synth_tile(backgrounds, patches, class_names, target_per_class,
                     class_count, font, tile_size, cfg):
    bg_path = random.choice(backgrounds)
    bg_img = Image.open(bg_path).convert("L")
    canvas_arr = np.array(Image.new("L", (tile_size, tile_size), 255), dtype=np.uint8)
    bg_arr = np.array(bg_img.resize((tile_size, tile_size)), dtype=np.uint8)
    blended = 255 - (255 - bg_arr) * 0.4
    canvas_arr = np.minimum(canvas_arr, blended.astype(np.uint8))

    canvas = Image.fromarray(canvas_arr, mode="L")
    inject_noise(canvas, font, tile_size, cfg.get("noise", {}))
    canvas_arr = np.array(canvas, dtype=np.uint8)

    placed, annotations = [], []
    order = list(class_names); random.shuffle(order)
    for cls in order:
        if not patches.get(cls):
            continue
        deficit = max(0, target_per_class - class_count[cls])
        if deficit == 0 and random.random() > 0.3:
            continue
        n_place = random.randint(1, 3) if deficit > 0 else 1
        for _ in range(n_place):
            info = random.choice(patches[cls])
            patch = extract_patch(info, margin=2)
            pw, ph = patch.size
            pos = try_place(pw, ph, placed, tile_size, cfg.get("iou_margin_px", 4))
            if pos is None:
                continue
            x, y = pos
            paste_multiply(canvas_arr, np.array(patch, dtype=np.uint8), x, y)
            placed.append((x, y, x + pw, y + ph))
            annotations.append({"category_name": cls, "bbox": [x, y, pw, ph]})
            class_count[cls] += 1

    if not annotations:
        return None
    return Image.fromarray(canvas_arr, mode="L"), annotations


def save_vis(tile: Image.Image, annots: list, out: Path, class_names: list[str]) -> None:
    rgb = tile.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    palette = [(0, 200, 0), (255, 128, 0), (200, 0, 200), (0, 128, 255),
               (255, 0, 0), (0, 200, 200), (200, 100, 0)]
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(class_names)}
    for a in annots:
        x, y, w, h = a["bbox"]
        c = color_map.get(a["category_name"], (0, 0, 0))
        draw.rectangle([x, y, x + w, y + h], outline=c, width=2)
        draw.text((x, max(0, y - 14)), a["category_name"], fill=c, font=font)
    rgb.save(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--labeled-dir", type=Path, required=True,
                        help="labelme JSON + 画像があるディレクトリ")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-synth", type=int, default=80)
    parser.add_argument("--target-per-class", type=int, default=100)
    parser.add_argument("--vis-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    class_names = cfg["classes"]
    tile_size = cfg.get("tile", {}).get("size", 1024)
    cp_cfg = cfg.get("copy_paste", {})

    random.seed(args.seed); np.random.seed(args.seed)
    synth_tiles = args.output_dir / "tiles"
    synth_vis = args.output_dir / "vis"
    synth_tiles.mkdir(parents=True, exist_ok=True)
    synth_vis.mkdir(parents=True, exist_ok=True)

    patches, labeled = load_manual_annotations(args.labeled_dir, class_names)
    all_tiles = sorted(args.labeled_dir.glob("*.png"))
    unlabeled = [p for p in all_tiles if p not in labeled]
    bg_pool = unlabeled if unlabeled else all_tiles

    print(f"classes: {class_names}")
    print(f"patches: {[(c, len(ps)) for c, ps in patches.items()]}")
    print(f"labeled: {len(labeled)}, bg_pool: {len(bg_pool)}")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except OSError:
        font = None

    class_count = Counter()
    images, annotations = [], []
    ann_id = 1
    for i in range(args.n_synth):
        result = build_synth_tile(
            bg_pool, patches, class_names,
            args.target_per_class, class_count, font, tile_size, cp_cfg,
        )
        if result is None:
            continue
        tile, annots = result
        img_id = i + 1
        fname = f"synth_{img_id:05d}.png"
        tile.save(synth_tiles / fname)
        images.append({"id": img_id, "file_name": fname,
                       "width": tile_size, "height": tile_size})
        for a in annots:
            cat_id = class_names.index(a["category_name"])
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": cat_id, "category_name": a["category_name"],
                "bbox": a["bbox"], "area": a["bbox"][2] * a["bbox"][3],
                "iscrowd": 0,
            })
            ann_id += 1
        if i < args.vis_count:
            save_vis(tile, annots, synth_vis / fname, class_names)

    coco = {
        "info": {"description": "synthetic (copy-paste augmentation)"},
        "images": images, "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in enumerate(class_names)],
    }
    (args.output_dir / "annotations.json").write_text(
        json.dumps(coco, ensure_ascii=False, indent=2))

    print(f"\n合成タイル: {len(images)}")
    print(f"合成アノテ: {len(annotations)}")
    print(f"クラス別: {dict(class_count)}")
    print(f"出力: {args.output_dir}/")


if __name__ == "__main__":
    main()
