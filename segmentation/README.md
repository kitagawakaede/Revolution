# provide/segmentation — セグメンテーションツールキット (U-Net + MiT-B2, Linux + CUDA)

学習・推論専用。Linux + CUDA 環境で動かす。
ラベリング・前処理・マスク変換は [provide/labeling/](../labeling/) で行ってください。

## 特徴

- **モデル**: SMP U-Net + **MiT-B2** encoder (MIT, 27.5M params)
- **損失**: CE + Dice + **soft-clDice**（細線の連結性保持）
- **Aug**: 512×512 ランダムクロップ + flip/rot90 + 同タイル繰り返しサンプリング
- **VRAM**: 3060 Ti 8GB 動作確認済（v04 kawasaki PoC で mIoU 0.209）
- **実行**: pixi (ローカル) / Docker (サーバー)

## 📁 配置

```
segmentation/
├── pixi.toml                   # Linux + CUDA 環境
├── Dockerfile / .dockerignore  # サーバーデプロイ用
├── configs/default.yaml
├── scripts/                    # 学習・推論
└── data/                       # ← git 管理外、ラベリング側から転送
    └── dataset/
        ├── images/             # アノテ済みタイル画像
        ├── masks/              # クラス ID マスク（uint8, 0=bg）
        └── overlay/            # 目視確認用（任意）

runs/                           # 学習出力（git 管理外）
└── segmentation/
    ├── best.pt
    └── history.json
```

ラベリング側で作成した `dataset/` フォルダをそのまま `data/` 配下に配置して使う想定。

## ▶️ 起動

### 前提: マスク化済みデータの受け渡し

`provide/labeling/` で以下を済ませておく:
1. タイル化 → labelme でアノテ → マスク変換
2. `pixi run to-mask --labeled-dir data/tiles --output-dir data/dataset`

転送例:
```bash
rsync -av provide/labeling/data/out/dataset/ user@server:~/provide/segmentation/data/dataset/
```

### ローカル (pixi)

```bash
cd provide/segmentation
pixi install                                   # 初回のみ
pixi run train --config configs/default.yaml
pixi run predict --ckpt runs/segmentation/best.pt --img test.png
# タイル予測の統合（タイル単位に予測を保存した場合）
pixi run stitch --tile-dir runs/seg/tiles --original input.png --output out.png
```

### Docker (サーバー)

**動作確認済み**: Docker 29.1.3 / Ubuntu 24.04 WSL2 で `docker build` 成功（所要 ~5 分、image 5.50GB）。
`docker run --rm provide-segmentation predict --help` で ENTRYPOINT + pixi task + argparse 通過確認済み。

```bash
# 1) Build
docker build -t provide-segmentation -f Dockerfile .

# 2) 学習
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/configs:/app/configs \
  provide-segmentation train --config configs/default.yaml

# 3) 推論
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  provide-segmentation predict --ckpt /app/runs/segmentation/best.pt --img /app/data/test.png
```

## ⚙️ 変更可能な設定

| 項目 | 場所 | 例 | 備考 |
|---|---|---|---|
| クラス名 | `configs/default.yaml` `classes` | `[pipe, wall]` | **必ず書き換え**。`_background_` は自動で 0 番 |
| データパス | `configs/default.yaml` `data.*` | `data/my_dataset/...` | labeling 側と合わせる |
| val 分割 | `configs/default.yaml` `data.val_tiles` | `["img1_r2c1"]` | 空なら `val_ratio` でランダム分割 |
| Encoder | `configs/default.yaml` `train.encoder` | `mit_b2` → `mit_b3` / `resnet34` | SMP 対応全種 |
| Crop size | `configs/default.yaml` `train.crop_size` | 512 → 384 / 256 | OOM 時に下げる |
| Epochs | `configs/default.yaml` `train.epochs` | 50 → 100 | 過学習と精度のバランス |
| clDice 重み | `configs/default.yaml` `train.loss.cldice_weight` | 1.0 → 2.0 | 細線で途切れが気になる時 ↑、polygon 領域なら 0 に |
| 線の太さ (ラベリング側) | `provide/labeling/.../config.yaml` | `rasterize_width_px` | マスク生成時。学習側では変えられない |
| CUDA バージョン | `Dockerfile` 1 行目 | 12.1 → 12.4 | torch wheel と整合必須 |
| Volume mount | `docker run -v ...` | `/app/data`, `/app/runs`, `/app/configs` | 3 つ必須 |

## 実績（kawasaki v04）

- 8 タイル / 435 shapes (linestrip, 2 クラス) 手動アノテ
- U-Net + MiT-B2 + clDice, 50 epoch, 4 分
- **val mIoU 0.209** @ 3060 Ti 8GB (VRAM 3.6GB)
- 未学習タイルで汎化確認

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| CUDA OOM | `crop_size` を 384/256、`batch_size` を 2 に |
| cv2 エラー (`_ARRAY_API not found`) | `numpy==1.26.4` 固定済、`pixi install` やり直し |
| val mIoU 低い | **アノテ拡充が第一**。次に `cldice_weight` を 2.0〜3.0 |
| 線が途切れて検出される | ラベリング側の `rasterize_width_px` を 4〜5 に |
| encoder 重み DL 失敗 | `~/.cache/torch/hub/` を確認、プロキシなら設定見直し |
| Docker build が失敗 (pixi install locked) | ローカルで `pixi install` して `pixi.lock` を commit し直す |
