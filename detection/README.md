# provide/detection — 物体検出ツールキット (RT-DETR, Linux + CUDA)

学習・推論専用。Linux + CUDA 環境で動かす。
ラベリング・前処理は [provide/labeling/](../labeling/) で行ってください。

## 特徴

- **モデル**: RT-DETR v2 (R18vd, Apache 2.0, 20M params)
- **学習**: Differential LR + Warmup + Early Stop
- **データ増幅**: Copy-paste augmentation (対象物が固定形状な場合に有効)
- **VRAM**: 3060 Ti 8GB 動作確認済 (v02b kawasaki PoC で AP50 43.9%)
- **実行**: pixi (ローカル) / Docker (サーバー)

## 📁 配置

```
detection/
├── pixi.toml                   # Linux + CUDA 環境
├── Dockerfile / .dockerignore  # サーバーデプロイ用
├── configs/default.yaml        # クラス・パス・ハイパラ（コピーして使う）
├── scripts/                    # 学習・評価・推論
└── data/                       # ← git 管理外、ラベリング側から転送
    ├── train/
    │   ├── images/             # labelme 済みタイル
    │   └── coco.json           # labelme → COCO 変換済み
    ├── val/
    │   ├── images/
    │   └── coco.json
    └── train/synth/            # (任意) copy_paste_augment の出力
        ├── tiles/
        └── annotations.json

runs/                           # 学習出力（ckpt / 学習ログ）、git 管理外
└── detection/
    ├── best/                   # best val_loss の ckpt
    ├── last/                   # 最終 epoch
    └── train_log.json
```

ラベリング側で作成した `coco.json` と画像を `data/` に転送して使う想定。

## ▶️ 起動

### 前提: アノテ済みデータの受け渡し

`provide/labeling/` で以下を済ませておく:
1. タイル化
2. labelme でアノテ
3. `pixi run to-coco data/tiles --output data/coco.json`

転送例（Mac → サーバ）:
```bash
rsync -av provide/labeling/data/out/ user@server:~/provide/detection/data/train/
```

### ローカル (pixi)

```bash
cd provide/detection
pixi install                                          # 初回のみ
# 任意: 合成データ増幅
pixi run augment --config configs/default.yaml \
  --labeled-dir data/train/images \
  --output-dir data/train/synth
# 学習
pixi run train --config configs/default.yaml
# 評価
pixi run eval --config configs/default.yaml --ckpt runs/detection/best
# 推論
pixi run predict --ckpt runs/detection/best --img test.png
```

### Docker (サーバー)

**動作確認済み**: Docker 29.1.3 / Ubuntu 24.04 WSL2 で `docker build` 成功（所要 ~8 分、image 5.52GB）。
`docker run --rm provide-detection predict --help` で ENTRYPOINT + pixi task + argparse 通過確認済み。

```bash
# 1) Build
docker build -t provide-detection -f Dockerfile .

# 2) 学習
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/configs:/app/configs \
  provide-detection train --config configs/default.yaml

# 3) 評価
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/configs:/app/configs \
  provide-detection eval --config configs/default.yaml --ckpt /app/runs/detection/best

# 4) 推論
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  provide-detection predict --ckpt /app/runs/detection/best --img /app/data/test.png
```

## ⚙️ 変更可能な設定

| 項目 | 場所 | 例 | 備考 |
|---|---|---|---|
| クラス名 | `configs/default.yaml` `classes` | `[my_class_1, my_class_2]` | **必ず書き換え** |
| データパス | `configs/default.yaml` `data.*` | `data/my_project/...` | 絶対パスでも相対でも OK |
| モデル | `configs/default.yaml` `train.model_id` | `PekingU/rtdetr_v2_r50vd` | R50 は精度↑VRAM↑ |
| Image size | `configs/default.yaml` `train.image_size` | 640 → 800 | 精度↑VRAM↑ |
| Batch size | `configs/default.yaml` `train.batch_size` | 4 → 2 / 8 | OOM 対策は下げる |
| LR | `configs/default.yaml` `train.lr_*` | backbone/mid/head 3 段 | v02b 実証値がデフォ |
| Warmup / EarlyStop | `configs/default.yaml` `train.*` | — | データ量で調整 |
| CUDA バージョン | `Dockerfile` 1 行目 | 12.1 → 12.4 | torch wheel と整合必須 |
| Volume mount | `docker run -v ...` | `/app/data`, `/app/runs`, `/app/configs` | 3 つ必須 |

## 実績（kawasaki v02b）

- 手付け 9 タイル / 97 instances (7 クラス) + 合成 74 タイル / 761 instances
- **AP50 43.9%** (RT-DETR v2 R18vd, 30 epoch, 12 分 on 3060 Ti 8GB)
- Differential LR で "フル FT" 比 +13pt 改善

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| CUDA OOM | `batch_size` を下げる、`image_size` を 512 に |
| numpy 2.x エラー | `numpy==1.26.4` 固定済、`pixi install` やり直し |
| `docker run` で GPU 見えない | `--gpus all` を付け忘れ、nvidia-container-toolkit インストール確認 |
| val_loss が下がらない | `lr_head` 上げる (5e-4)、Copy-paste aug を有効化、データ量増やす |
| `pixi install --locked` が失敗 (Docker) | `pixi.lock` を最新化: ローカルで `pixi install` して lockfile を commit し直す |
