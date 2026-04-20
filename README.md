# provide — 再利用可能な検出・セグメンテーション ツールキット

他プロジェクトでも使い回せるよう、kawasaki PoC で実証済みのパイプラインを
**3 つの独立したツールキット**として整備したもの。

## 構成（3 分割）

| Toolkit | 用途 | プラットフォーム | GPU | Docker |
|---|---|---|---|---|
| [`labeling/`](labeling/) | アノテ・前処理・変換 | **Mac / Win / Linux** | 不要 | 不要 |
| [`detection/`](detection/) | RT-DETR 学習・推論 | Linux | 必須 (CUDA 12.1) | あり |
| [`segmentation/`](segmentation/) | U-Net + MiT-B2 学習・推論 | Linux | 必須 (CUDA 12.1) | あり |

**分割の理由**:
- アノテは Mac / Win の作業者が行うことがある → クロスプラットフォーム必要
- 学習は GPU サーバーで行う → GUI 不要・Docker で再現性担保
- 3 つに分けることで各環境のインストール量・依存衝突が最小化

## ワークフロー全体像

```
[Mac/Win/Linux + labeling/]         [Linux サーバー + detection/ or segmentation/]
  raw 画像                                    │
    ↓ pixi run tile                          │
  1024×1024 タイル                           │
    ↓ ./label.sh (labelme GUI)               │
  labelme JSON                               │
    ↓ pixi run to-coco / to-mask             │
  COCO / マスク     ────── rsync / git ──→   data/
                                             ↓ pixi run train  /  docker run ... train
                                            ckpt
                                             ↓ pixi run predict / docker run ... predict
                                          予測画像
```

## Quick Start（代表例）

### セグメンテーション（配管・細線など）

**アノテ側（Mac 等）:**
```bash
cd provide/labeling
pixi install
pixi run tile --image raw/input.png --out-dir data/tiles
./label.sh data/tiles labels_sample_segmentation.txt
pixi run to-mask --config <your-config>.yaml \
  --labeled-dir data/tiles --output-dir data/out/dataset
```

**学習側（Linux サーバ）:**
```bash
cd provide/segmentation
# ローカル pixi
pixi install
pixi run train --config configs/default.yaml

# or Docker
docker build -t provide-segmentation -f Dockerfile .
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data -v $(pwd)/runs:/app/runs -v $(pwd)/configs:/app/configs \
  provide-segmentation train --config configs/default.yaml
```

### 物体検出（bbox 記号・部品など）

同じ流れで `labeling/` → `detection/` に流す。変換は `pixi run to-coco`。

## どちらを選ぶか

| タスク例 | 推奨 |
|---|---|
| 記号・マーク・部品（bbox で囲える） | `detection/` |
| 配管・ケーブル・細線（中心線をトレース） | `segmentation/` |
| 壁・領域（polygon で塗る） | `segmentation/` |

## 📁 配置（推奨）

各 toolkit は**独立したディレクトリ**として使う。依存衝突を避けるため別環境。

```
provide/
├── labeling/      ← Mac / Win / Linux で配布（作業者ごと）
├── detection/     ← Linux サーバーに配布
└── segmentation/  ← Linux サーバーに配布
```

提供時はフォルダまるごと ZIP / git clone で渡せば OK。

## kawasaki PoC での実績（参考値）

| Toolkit | データ | 精度 | 参考元 |
|---|---|---|---|
| detection | 9 タイル / 97 instances (7 クラス) | **AP50 43.9%** | `vlm/kawasaki_l2_drawing_poc/v02b_symbol_detection_ft` |
| segmentation | 8 タイル / 435 shapes (2 クラス) | **mIoU 0.209** | `vlm/kawasaki_l2_drawing_poc/v04_pipe_extraction` |

アノテ量を増やすと精度が伸びる段階（両 PoC とも）。スケーリングの目安として。

## 設計方針

1. **3 分割**: アノテ (CPU/cross-platform) と 学習 (GPU/Linux) を分けた
2. **pixi ベース**: lockfile 含めバージョン再現性を担保
3. **config ファースト**: クラス名・ハイパラ・パスは YAML で指定
4. **Docker 対応**: サーバーでは pixi を Docker に持ち込んで同じ pin を使用
5. **最小スクリプト**: 探しやすさ優先

## ライセンス構成

| コンポーネント | ライセンス | 用途 |
|---|---|---|
| transformers | Apache 2.0 | RT-DETR |
| segmentation-models-pytorch | MIT | U-Net + MiT |
| timm | Apache 2.0 | MiT 事前学習重みロード |
| albumentations | MIT | データ増幅 |
| pycocotools | BSD | COCO mAP |
| labelme | GPL | ラベリング（成果物は別ライセンス） |

学習済みモデル・推論結果は labelme を含まないため、商用配布可能な OSS ライセンス
組み合わせになっている（labelme は開発フェーズのみ使用）。
