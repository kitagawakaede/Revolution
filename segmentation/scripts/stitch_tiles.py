"""タイル単位のマスク予測を元画像サイズに復元（統合）

tile_image.py で分割したタイルのアノテや予測マスクを、
{stem}_r{r}c{c}.png の命名規約からページ単位に縫い合わせる。

使い方:
  pixi run python scripts/stitch_tiles.py \\
    --tile-dir runs/segmentation/predict_tiles \\
    --original input.png \\
    --output full_mask.png
"""

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

TILE_PATTERN = re.compile(r"^(.+?)_r(\d+)c(\d+)$")


def stitch(tile_dir: Path, stem: str, tile_size: int, W: int, H: int,
           suffix: str = ".png") -> np.ndarray:
    """タイルから原サイズのマスクを復元"""
    full = np.zeros((H, W), dtype=np.uint8)
    for p in sorted(tile_dir.glob(f"{stem}_r*c*{suffix}")):
        m = TILE_PATTERN.match(p.stem)
        if not m:
            continue
        _, r, c = m.groups()
        r, c = int(r), int(c)
        x0 = c * tile_size; y0 = r * tile_size
        x1 = min(x0 + tile_size, W); y1 = min(y0 + tile_size, H)
        tile = np.array(Image.open(p))
        full[y0:y1, x0:x1] = tile[:y1 - y0, :x1 - x0]
    return full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-dir", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True,
                        help="元画像（サイズ取得用）")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stem", type=str, default=None,
                        help="タイル prefix（省略時は original の stem）")
    args = parser.parse_args()

    orig = Image.open(args.original)
    W, H = orig.size
    stem = args.stem or args.original.stem
    full = stitch(args.tile_dir, stem, args.tile_size, W, H)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(full).save(args.output)
    print(f"stitched ({W}x{H}) → {args.output}")


if __name__ == "__main__":
    main()
