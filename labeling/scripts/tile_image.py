"""画像をタイルに分割（物体検出用、overlap あり）

使い方:
  pixi run python scripts/tile_image.py --image in.png --out-dir tiles/
  pixi run python scripts/tile_image.py --image-dir raw/ --out-dir tiles/ --config configs/default.yaml
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from _common import load_config


def tile_grid(W: int, H: int, tile: int, overlap: int) -> list[tuple[int, int, int, int, int, int]]:
    """オーバーラップ付きタイル grid。(row, col, x0, y0, x1, y1)"""
    stride = max(1, tile - overlap)
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if not xs or xs[-1] + tile < W:
        xs.append(max(W - tile, 0))
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    if not ys or ys[-1] + tile < H:
        ys.append(max(H - tile, 0))
    tiles = []
    for r, ty in enumerate(ys):
        for c, tx in enumerate(xs):
            tiles.append((r, c, tx, ty, tx + tile, ty + tile))
    return tiles


def empty_labelme(image_filename: str, W: int, H: int) -> dict:
    return {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W,
    }


def process_image(image_path: Path, out_dir: Path, tile_size: int, overlap: int,
                  emit_json: bool) -> int:
    stem = image_path.stem
    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    tiles = tile_grid(W, H, tile_size, overlap)
    for r, c, x0, y0, x1, y1 in tiles:
        name = f"{stem}_r{r}c{c}"
        crop = image.crop((x0, y0, x1, y1))
        # 端タイルも tile_size に揃える（白 pad）
        padded = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
        padded.paste(crop, (0, 0))
        padded.save(out_dir / f"{name}.png")
        if emit_json:
            lj = empty_labelme(f"{name}.png", tile_size, tile_size)
            (out_dir / f"{name}.json").write_text(
                json.dumps(lj, ensure_ascii=False, indent=2)
            )
    return len(tiles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="単一画像パス")
    parser.add_argument("--image-dir", type=Path, help="ディレクトリ指定（画像複数）")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config（tile.size / tile.overlap を読む）")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--no-json", action="store_true",
                        help="空の labelme JSON を出さない")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        args.tile_size = cfg.get("tile", {}).get("size", args.tile_size)
        args.overlap = cfg.get("tile", {}).get("overlap", args.overlap)

    if not args.image and not args.image_dir:
        parser.error("--image または --image-dir のどちらかが必要")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = [args.image] if args.image else sorted(args.image_dir.glob("*.png")) + \
                                              sorted(args.image_dir.glob("*.jpg"))
    total = 0
    for p in targets:
        n = process_image(p, args.out_dir, args.tile_size, args.overlap,
                          emit_json=not args.no_json)
        print(f"[{p.name}] tiles={n}")
        total += n
    print(f"合計: {total} tiles → {args.out_dir}/")


if __name__ == "__main__":
    main()
