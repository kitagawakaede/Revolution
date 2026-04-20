"""RT-DETR Fine-tuning（汎用版、v02b 由来）

特徴:
  - Differential LR（backbone / encoder-decoder / head で別 LR）
  - 線形 warmup → cosine decay
  - val_loss ベース Early Stop
  - クラス名・パス・ハイパラは config から

使い方:
  pixi run python scripts/train_rtdetr.py --config configs/default.yaml
"""

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from _common import load_config


class CocoDetDataset(Dataset):
    """labelme → COCO 変換済みの JSON を学習用に読み込む"""

    def __init__(self, images_dir: Path, coco_json: Path, processor, class_names: list[str]):
        self.images_dir = images_dir
        self.processor = processor
        self.class_names = class_names
        data = json.loads(coco_json.read_text())
        self.images = {im["id"]: im for im in data["images"]}
        cat_map = {c["name"]: c["id"] for c in data["categories"]}
        self.cat_remap = {
            cat_map[n]: i for i, n in enumerate(class_names) if n in cat_map
        }
        self.per_image = {im_id: [] for im_id in self.images}
        for a in data["annotations"]:
            if a["category_id"] not in self.cat_remap:
                continue
            self.per_image[a["image_id"]].append(a)
        self.image_ids = [i for i, anns in self.per_image.items() if anns]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        im_id = self.image_ids[idx]
        im_info = self.images[im_id]
        img = Image.open(self.images_dir / im_info["file_name"]).convert("RGB")
        anns = self.per_image[im_id]
        annotations = [{
            "image_id": im_id,
            "category_id": self.cat_remap[a["category_id"]],
            "bbox": a["bbox"], "area": a["area"],
            "iscrowd": a.get("iscrowd", 0),
        } for a in anns]
        enc = self.processor(
            images=img, annotations={"image_id": im_id, "annotations": annotations},
            return_tensors="pt",
        )
        return {"pixel_values": enc["pixel_values"][0], "labels": enc["labels"][0]}


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": [b["labels"] for b in batch],
    }


def build_differential_lr_groups(model, lr_backbone, lr_mid, lr_head):
    head_keywords = ("class_embed", "enc_score_head", "denoising_class_embed")
    backbone_keywords = ("backbone",)
    head_params, backbone_params, mid_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in head_keywords):
            head_params.append(p)
        elif any(k in name for k in backbone_keywords):
            backbone_params.append(p)
        else:
            mid_params.append(p)
    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": mid_params, "lr": lr_mid},
        {"params": head_params, "lr": lr_head},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    class_names = cfg["classes"]
    data = cfg["data"]
    t = cfg["train"]

    output_dir = Path(t.get("output_dir", "runs/detection"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, classes: {class_names}")

    id2label = {i: n for i, n in enumerate(class_names)}
    label2id = {n: i for i, n in enumerate(class_names)}

    processor = AutoImageProcessor.from_pretrained(
        t["model_id"],
        size={"width": t["image_size"], "height": t["image_size"]},
    )
    model = AutoModelForObjectDetection.from_pretrained(
        t["model_id"], id2label=id2label, label2id=label2id,
        num_labels=len(class_names), ignore_mismatched_sizes=True,
        anchor_image_size=[t["image_size"], t["image_size"]],
    ).to(device)

    train_ds = CocoDetDataset(Path(data["train_images"]), Path(data["train_coco"]),
                              processor, class_names)
    val_ds = CocoDetDataset(Path(data["val_images"]), Path(data["val_coco"]),
                            processor, class_names)
    print(f"train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                              collate_fn=collate_fn, num_workers=t.get("num_workers", 2))
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False,
                            collate_fn=collate_fn, num_workers=t.get("num_workers", 2))

    groups = build_differential_lr_groups(
        model, t["lr_backbone"], t["lr_encoder_decoder"], t["lr_head"],
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=t["weight_decay"])

    steps_per_epoch = max(1, len(train_ds) // t["batch_size"])
    total_steps = t["epochs"] * steps_per_epoch
    warmup_steps = min(t["warmup_steps"], total_steps // 4)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log = []
    best_val = float("inf")
    patience = 0
    global_step = 0
    for epoch in range(t["epochs"]):
        model.train()
        sum_train = 0.0; n_b = 0
        for batch in train_loader:
            pv = batch["pixel_values"].to(device)
            lbls = [{k: v.to(device) for k, v in x.items()} for x in batch["labels"]]
            out = model(pixel_values=pv, labels=lbls)
            optimizer.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            sum_train += out.loss.item(); n_b += 1
        train_loss = sum_train / max(1, n_b)

        model.eval()
        sum_val = 0.0; n_b = 0
        with torch.no_grad():
            for batch in val_loader:
                pv = batch["pixel_values"].to(device)
                lbls = [{k: v.to(device) for k, v in x.items()} for x in batch["labels"]]
                out = model(pixel_values=pv, labels=lbls)
                sum_val += out.loss.item(); n_b += 1
        val_loss = sum_val / max(1, n_b)

        lrs = scheduler.get_last_lr()
        log.append({"epoch": epoch + 1, "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "lr_backbone": lrs[0], "lr_mid": lrs[1], "lr_head": lrs[2]})
        print(f"ep {epoch+1:3d}/{t['epochs']} train={train_loss:.4f} val={val_loss:.4f} "
              f"lr(bb/mid/hd)={lrs[0]:.1e}/{lrs[1]:.1e}/{lrs[2]:.1e}")

        if val_loss < best_val - 1e-3:
            best_val = val_loss; patience = 0
            model.save_pretrained(output_dir / "best")
            processor.save_pretrained(output_dir / "best")
        else:
            patience += 1
            if patience >= t["early_stop_patience"]:
                print(f"  early stop @ {patience} epochs no improvement")
                break

    model.save_pretrained(output_dir / "last")
    processor.save_pretrained(output_dir / "last")
    (output_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"\n完了: best val={best_val:.4f}  → {output_dir}/(best|last)")


if __name__ == "__main__":
    main()
