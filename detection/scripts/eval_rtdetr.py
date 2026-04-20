"""COCO mAP で RT-DETR 学習済みモデルを評価

使い方:
  pixi run python scripts/eval_rtdetr.py \\
    --config configs/default.yaml \\
    --ckpt runs/detection/best \\
    --images data/val/images --coco data/val/coco.json
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from _common import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--ckpt", type=Path, required=True, help="best/last ディレクトリ")
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--coco", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    class_names = cfg["classes"]
    images_dir = args.images or Path(cfg["data"]["val_images"])
    gt_coco = args.coco or Path(cfg["data"]["val_coco"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.ckpt)
    model = AutoModelForObjectDetection.from_pretrained(args.ckpt).to(device).eval()

    coco_gt = COCO(str(gt_coco))
    # class name → GT の category id 逆引き
    name_to_gtid = {c["name"]: c["id"] for c in coco_gt.loadCats(coco_gt.getCatIds())}

    results = []
    for img_id in coco_gt.getImgIds():
        im_info = coco_gt.loadImgs(img_id)[0]
        img = Image.open(images_dir / im_info["file_name"]).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]]).to(device)
        post = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=args.threshold,
        )[0]
        for score, label, box in zip(post["scores"], post["labels"], post["boxes"]):
            cls_name = class_names[label.item()]
            if cls_name not in name_to_gtid:
                continue
            x0, y0, x1, y1 = [v.item() for v in box]
            results.append({
                "image_id": img_id,
                "category_id": name_to_gtid[cls_name],
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": score.item(),
            })

    if not results:
        print("No predictions above threshold.")
        return

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"saved predictions: {args.output}")

    coco_dt = coco_gt.loadRes(results)
    e = COCOeval(coco_gt, coco_dt, iouType="bbox")
    e.evaluate()
    e.accumulate()
    e.summarize()


if __name__ == "__main__":
    main()
