"""学習済みモデルで画像 1 枚を予測（タイル grid 推論 + 統合）

使い方:
  pixi run python scripts/predict.py --ckpt runs/segmentation/best.pt --img test.png
"""

import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

PALETTE = [
    (255, 255, 255),
    (255, 0, 0), (0, 100, 255), (0, 200, 0), (255, 128, 0),
    (200, 0, 200), (0, 200, 200), (200, 100, 0),
]


def predict_full(model, image: np.ndarray, device: str, crop: int) -> np.ndarray:
    H, W = image.shape[:2]
    pred = np.zeros((H, W), dtype=np.uint8)
    norm = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    for ty in range(0, H, crop):
        for tx in range(0, W, crop):
            y1 = min(ty + crop, H); x1 = min(tx + crop, W)
            patch = np.full((crop, crop, 3), 255, dtype=np.uint8)
            ph = y1 - ty; pw = x1 - tx
            patch[:ph, :pw] = image[ty:y1, tx:x1]
            x = norm(image=patch)["image"].unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
            p = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            pred[ty:y1, tx:x1] = p[:ph, :pw]
    return pred


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    vis = image.copy()
    for cid in range(1, mask.max() + 1):
        color = PALETTE[cid % len(PALETTE)]
        sel = mask == cid
        vis[sel] = (vis[sel] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--img", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mask-out", type=Path, default=None,
                        help="クラス ID マスクを別途保存")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    class_names = ckpt["classes"]
    model = smp.Unet(
        encoder_name=ckpt["encoder"],
        encoder_weights=None,
        in_channels=3,
        classes=len(class_names) + 1,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[ckpt] epoch={ckpt['epoch']}, classes={class_names}")

    image = np.array(Image.open(args.img).convert("RGB"))
    mask = predict_full(model, image, device, args.crop_size)
    vis = overlay(image, mask)

    out = args.output or args.img.with_name(args.img.stem + "_pred.png")
    Image.fromarray(vis).save(out)
    print(f"overlay → {out}")
    if args.mask_out:
        Image.fromarray(mask).save(args.mask_out)
        print(f"mask → {args.mask_out}")

    for i, n in enumerate(class_names, start=1):
        print(f"  {n}: {int((mask == i).sum())} px")


if __name__ == "__main__":
    main()
