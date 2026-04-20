"""U-Net (SMP) 多クラスセグメンテーション FT

特徴:
  - encoder 選択可 (config): mit_b0〜b4, resnet34, efficientnet-b0 など SMP 対応全般
  - 損失: CE + Dice + soft-clDice（線状構造向け、config で重み調整）
  - FP16, CosineLR, ランダム 512 crop + flip/rot90

使い方:
  pixi run python scripts/train_unet.py --config configs/default.yaml
"""

import argparse
import json
import random
import time
from pathlib import Path

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from _common import load_config


# --------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------
class TileDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_p, mask_p = self.items[idx]
        image = np.array(Image.open(img_p).convert("RGB"))
        mask = np.array(Image.open(mask_p))
        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].long()


def build_transforms(train: bool, crop_size: int) -> A.Compose:
    if train:
        return A.Compose([
            A.RandomCrop(crop_size, crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# --------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------
def soft_skeletonize(mask: torch.Tensor, n_iter: int = 10) -> torch.Tensor:
    for _ in range(n_iter):
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        opening = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)
        residual = F.relu(mask - opening)
        mask = F.relu(mask - residual)
    return mask


def soft_cldice_loss(pred, target_onehot, smooth=1.0, n_iter=10):
    total = 0.0
    n = pred.shape[1] - 1
    for c in range(1, pred.shape[1]):
        p = pred[:, c:c + 1]; t = target_onehot[:, c:c + 1]
        skel_p = soft_skeletonize(p, n_iter)
        skel_t = soft_skeletonize(t, n_iter)
        tprec = ((skel_p * t).sum() + smooth) / (skel_p.sum() + smooth)
        tsens = ((skel_t * p).sum() + smooth) / (skel_t.sum() + smooth)
        total = total + (1.0 - 2.0 * tprec * tsens / (tprec + tsens))
    return total / max(1, n)


def dice_loss(pred, target_onehot, smooth=1.0):
    total = 0.0
    n = pred.shape[1] - 1
    for c in range(1, pred.shape[1]):
        p = pred[:, c]; t = target_onehot[:, c]
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        total = total + (1.0 - (2.0 * inter + smooth) / (denom + smooth))
    return total / max(1, n)


def combined_loss(logits, target, num_classes, w_ce=1.0, w_dice=1.0, w_cldice=1.0):
    ce = F.cross_entropy(logits, target)
    pred = F.softmax(logits, dim=1)
    oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dice = dice_loss(pred, oh)
    cldice = soft_cldice_loss(pred, oh) if w_cldice > 0 else torch.tensor(0.0, device=logits.device)
    total = w_ce * ce + w_dice * dice + w_cldice * cldice
    return {"total": total, "ce": ce.detach(), "dice": dice.detach(), "cldice": cldice.detach()}


# --------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, val_items, device, num_classes, crop_size):
    model.eval()
    norm = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    total_inter = [0] * (num_classes - 1)
    total_union = [0] * (num_classes - 1)
    for img_p, mask_p in val_items:
        image = np.array(Image.open(img_p).convert("RGB"))
        mask = np.array(Image.open(mask_p))
        H, W = mask.shape
        pred_full = np.zeros((H, W), dtype=np.uint8)
        for ty in range(0, H, crop_size):
            for tx in range(0, W, crop_size):
                y1 = min(ty + crop_size, H); x1 = min(tx + crop_size, W)
                patch = np.full((crop_size, crop_size, 3), 255, dtype=np.uint8)
                ph = y1 - ty; pw = x1 - tx
                patch[:ph, :pw] = image[ty:y1, tx:x1]
                x = norm(image=patch)["image"].unsqueeze(0).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                p = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                pred_full[ty:y1, tx:x1] = p[:ph, :pw]
        for c in range(1, num_classes):
            inter = ((pred_full == c) & (mask == c)).sum()
            union = ((pred_full == c) | (mask == c)).sum()
            total_inter[c - 1] += int(inter)
            total_union[c - 1] += int(union)
    ious = [total_inter[i] / max(total_union[i], 1) for i in range(num_classes - 1)]
    return {"per_class_iou": ious, "miou": sum(ious) / max(1, len(ious))}


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    class_names = cfg["classes"]
    num_classes = len(class_names) + 1  # +1 for background
    t = cfg["train"]
    loss_cfg = t.get("loss", {})

    output_dir = Path(t.get("output_dir", "runs/segmentation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = t.get("seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_root = Path(cfg["data"]["dataset_root"])
    all_items = sorted([
        (ds_root / "images" / p.name, p)
        for p in (ds_root / "masks").glob("*.png")
        if (ds_root / "images" / p.name).exists()
    ])
    if not all_items:
        raise RuntimeError(f"No paired tiles in {ds_root}/(images|masks)")

    val_tiles = set(cfg["data"].get("val_tiles") or [])
    if val_tiles:
        train_items = [it for it in all_items if it[0].stem not in val_tiles]
        val_items = [it for it in all_items if it[0].stem in val_tiles]
    else:
        random.shuffle(all_items)
        n_val = max(1, int(len(all_items) * t.get("val_ratio", 0.15)))
        val_items = all_items[:n_val]
        train_items = all_items[n_val:]
    print(f"[data] train={len(train_items)}, val={len(val_items)}")
    print(f"       classes (+bg): {['background'] + class_names}")

    crop_size = t.get("crop_size", 512)
    crops_per = t.get("crops_per_tile_per_epoch", 16)
    train_ds = TileDataset(train_items * crops_per, build_transforms(True, crop_size))
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                              num_workers=2, drop_last=True)
    print(f"       1 epoch = {len(train_ds)} crops, {len(train_loader)} iter")

    model = smp.Unet(
        encoder_name=t["encoder"],
        encoder_weights=t.get("encoder_weights", "imagenet"),
        in_channels=3,
        classes=num_classes,
    ).to(device)
    print(f"[model] U-Net + {t['encoder']}, "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    scaler = torch.amp.GradScaler("cuda") if t.get("fp16", True) else None

    best_miou = -1.0
    history = []
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        t0 = time.time()
        ls = {"total": 0.0, "ce": 0.0, "dice": 0.0, "cldice": 0.0}
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    ld = combined_loss(logits, y, num_classes,
                                       loss_cfg.get("ce_weight", 1.0),
                                       loss_cfg.get("dice_weight", 1.0),
                                       loss_cfg.get("cldice_weight", 1.0))
                scaler.scale(ld["total"]).backward()
                scaler.step(optimizer); scaler.update()
            else:
                logits = model(x)
                ld = combined_loss(logits, y, num_classes,
                                   loss_cfg.get("ce_weight", 1.0),
                                   loss_cfg.get("dice_weight", 1.0),
                                   loss_cfg.get("cldice_weight", 1.0))
                ld["total"].backward(); optimizer.step()
            for k in ls:
                ls[k] += float(ld[k])
        for k in ls:
            ls[k] /= max(1, len(train_loader))
        scheduler.step()

        metrics = evaluate(model, val_items, device, num_classes, crop_size)
        dur = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        history.append({
            "epoch": epoch, "loss": ls, "val": metrics,
            "dur_s": round(dur, 1), "vram_mb": round(vram, 0),
        })
        iou_str = " ".join(f"{n}={v:.3f}" for n, v in zip(class_names, metrics["per_class_iou"]))
        print(f"[ep {epoch:3d}/{t['epochs']}] loss={ls['total']:.4f} "
              f"(ce={ls['ce']:.3f} dice={ls['dice']:.3f} cldice={ls['cldice']:.3f}) | "
              f"IoU {iou_str} mIoU={metrics['miou']:.3f} | {dur:.1f}s VRAM={vram:.0f}MB")

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch, "metrics": metrics,
                "encoder": t["encoder"], "classes": class_names,
            }, output_dir / "best.pt")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\n最良 mIoU: {best_miou:.3f}")
    print(f"→ {output_dir}/best.pt")


if __name__ == "__main__":
    main()
