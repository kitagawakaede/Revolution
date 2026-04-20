# provide/labeling — アノテ・前処理 ツールキット（クロスプラットフォーム）

画像のタイル化・labelme によるラベル付け・学習用フォーマットへの変換を
**Mac / Windows / Linux のどこでも動く** ように用意したもの。

学習・推論は別 toolkit（`provide/detection/`, `provide/segmentation/`）で行います。

## 対応プラットフォーム

- **Linux** (x86_64)
- **macOS** (Apple Silicon / Intel)
- **Windows** (x86_64)

GPU・CUDA 不要。CPU のみで完結。

## 📁 配置（どこに何を置くか）

```
labeling/
├── pixi.toml              # 環境定義（このディレクトリごと提供用コピー）
├── label.sh / label.bat   # labelme 起動ラッパ
├── labels_sample_*.txt    # ラベル定義の雛形（プロジェクト用にコピー改変）
├── scripts/               # 変換スクリプト
└── data/                  # ← 作業用に作って使う（.gitignore 済）
    ├── raw/                   # 元画像（大きいまま置く）
    ├── tiles/                 # tile_image.py の出力（画像 + 空 labelme JSON）
    └── out/                   # labelme_to_coco / labelme_to_mask の出力
```

## ▶️ 起動（主要コマンド）

### 初回セットアップ

```bash
cd provide/labeling
pixi install                     # 初回のみ。OS ごとに数分
```

### 1. 画像タイル化（大きい画像を 1024×1024 に分割）

```bash
pixi run tile --image data/raw/input.png --out-dir data/tiles/
# or ディレクトリ一括
pixi run tile --image-dir data/raw/ --out-dir data/tiles/
```

### 2. labelme で手動ラベリング

**Linux / macOS**:
```bash
./label.sh data/tiles labels_sample_detection.txt
```

**Windows**:
```cmd
label.bat data\tiles labels_sample_detection.txt
```

`--autosave` で自動保存、`D` 次画像 / `A` 前画像。shapes は:
- **rectangle** — bbox（Detection 用）
- **linestrip** — 中心線トレース（Segmentation 細線用）
- **polygon** — 領域（Segmentation 面用）

### 3. 変換

**Detection 用（labelme JSON → COCO）**:
```bash
pixi run to-coco data/tiles --output data/out/coco.json
```

**Segmentation 用（labelme JSON → クラス ID マスク）**:
```bash
pixi run to-mask --config config.yaml \
  --labeled-dir data/tiles --output-dir data/out/dataset/
```

### 4. 学習側への受け渡し

```bash
# サーバに転送する例（rsync）
rsync -av data/out/ user@server:~/provide/detection/data/
rsync -av data/out/dataset/ user@server:~/provide/segmentation/data/dataset/
```

## ⚙️ 変更可能な設定

| 項目 | どこを変更 | 例 | 備考 |
|---|---|---|---|
| ラベル名 | `labels_sample_*.txt` | `class_a` → 自プロジェクト用 | コピーして自前 .txt を作る |
| タイルサイズ | `--tile-size` or config | 1024 → 512 / 2048 | 小さいほどアノテ件数増 |
| タイル overlap | `--overlap` | 128 → 0 | Detection=128 推奨, Segmentation=0 推奨 |
| linestrip の太さ | `to-mask` の config `label.rasterize_width_px` | 3 → 5 | 太線が対象なら増やす |
| labelme ショートカット | labelme GUI 設定 | — | `D`/`A`=前後、`Ctrl+R`=矩形 etc |

## labelme 使い方 Tips

- **Create Rectangle** (`Ctrl+R`) — bbox
- **Create LineStrip** (`Ctrl+L`) — 多点線（中心線用）
- **Create Polygon** (`Ctrl+N`) — 多角形
- **削除** — 右クリックメニュー or `Delete` キー
- **複製** — `Ctrl+D`（bbox は位置微調整しやすい）
- **次/前画像** — `D` / `A`

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| Mac で labelme 起動時に Qt エラー | `pixi install` をやり直し、conda-forge の PyQt がフェッチされるのを確認 |
| Windows で .bat が閉じる | コマンドプロンプトから実行してエラーを確認 |
| 変換スクリプトで numpy エラー | `numpy==1.26.4` 固定済、`pixi install` で再構築 |
| JSON が学習側で読めない | 学習 toolkit の `classes:` と labelme ラベル名が一致しているか確認 |

## 想定ワークフロー（全体）

```
[このツールキット / Mac] raw画像 → tile → labelme → COCO/mask
                ↓ rsync / git push
[detection/segmentation / Linux サーバ] 学習 → 推論
```
