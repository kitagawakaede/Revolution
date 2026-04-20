"""labelme JSON → COCO 形式変換

labelme は画像1枚ごとに <basename>.json を吐く。これを COCO 1つにまとめる。

使い方:
  pixi run python scripts/labelme_to_coco.py <labelme_dir> --output coco.json
"""

import argparse
import json
from pathlib import Path


def labelme_to_coco(labelme_dir: Path, output: Path) -> None:
    json_files = sorted(labelme_dir.glob("*.json"))
    if not json_files:
        print(f"No .json found in {labelme_dir}")
        return

    # ラベル収集
    label_set: set[str] = set()
    for jf in json_files:
        data = json.loads(jf.read_text())
        for sh in data.get("shapes", []):
            label_set.add(sh["label"])
    categories = sorted(label_set)
    cat_to_id = {c: i for i, c in enumerate(categories)}

    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1
    for img_id, jf in enumerate(json_files, start=1):
        data = json.loads(jf.read_text())
        img_path = data.get("imagePath", jf.stem + ".png")
        images.append({
            "id": img_id,
            "file_name": img_path,
            "width": data.get("imageWidth"),
            "height": data.get("imageHeight"),
        })
        for sh in data.get("shapes", []):
            if sh.get("shape_type") != "rectangle":
                continue
            (x0, y0), (x1, y1) = sh["points"]
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])
            bw, bh = x1 - x0, y1 - y0
            if bw <= 0 or bh <= 0:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_to_id[sh["label"]],
                "category_name": sh["label"],
                "bbox": [x0, y0, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": {"description": f"converted from labelme ({labelme_dir})"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": c} for c, i in cat_to_id.items()],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coco, ensure_ascii=False, indent=2))

    from collections import Counter
    cnt = Counter(a["category_name"] for a in annotations)
    print(f"images: {len(images)}  annotations: {len(annotations)}")
    print(f"per class: {dict(cnt)}")
    print(f"saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labelme_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("coco.json"))
    args = parser.parse_args()
    labelme_to_coco(args.labelme_dir, args.output)


if __name__ == "__main__":
    main()
