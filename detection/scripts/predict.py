"""学習済み RT-DETR で単画像推論 + bbox オーバーレイ

使い方:
  pixi run python scripts/predict.py --ckpt runs/detection/best --img test.png
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection


PALETTE = [
    (0, 200, 0), (255, 128, 0), (200, 0, 200), (0, 128, 255),
    (255, 0, 0), (0, 200, 200), (200, 100, 0), (100, 100, 200),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--img", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--output", type=Path, default=None,
                        help="overlay PNG 出力（省略時は <img>_pred.png）")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.ckpt)
    model = AutoModelForObjectDetection.from_pretrained(args.ckpt).to(device).eval()
    id2label = model.config.id2label

    img = Image.open(args.img).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([img.size[::-1]]).to(device)
    post = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=args.threshold,
    )[0]

    detections = []
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for score, label, box in zip(post["scores"], post["labels"], post["boxes"]):
        lbl = id2label[label.item()]
        x0, y0, x1, y1 = [v.item() for v in box]
        s = score.item()
        color = PALETTE[label.item() % len(PALETTE)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0, max(0, y0 - 18)), f"{lbl} {s:.2f}", fill=color, font=font)
        detections.append({"label": lbl, "score": round(s, 4),
                           "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]})

    out = args.output or args.img.with_name(args.img.stem + "_pred.png")
    vis.save(out)
    print(f"{len(detections)} detections → {out}")

    if args.json_out:
        args.json_out.write_text(json.dumps(detections, indent=2, ensure_ascii=False))
        print(f"JSON → {args.json_out}")


if __name__ == "__main__":
    main()
