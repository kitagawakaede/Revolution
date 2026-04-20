# provide/detection サーバー学習・推論用 Dockerfile
#
# ビルド:
#   docker build -t provide-detection -f Dockerfile .
#
# 実行 (GPU + データ/出力を volume マウント):
#   docker run --rm --gpus all \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/runs:/app/runs \
#     -v $(pwd)/configs:/app/configs \
#     provide-detection train --config configs/default.yaml
#
# 変更可能: CUDA バージョンは base image 1 行目。torch wheel の cu121 と整合させること。

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIXI_HOME=/root/.pixi \
    PATH=/root/.pixi/bin:$PATH \
    PYTHONUNBUFFERED=1

# 最小限の OS パッケージ + pixi
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://pixi.sh/install.sh | bash

WORKDIR /app

# 依存解決（コードより先にコピーしてキャッシュ活用）
COPY pixi.toml pixi.lock ./
RUN pixi install --locked

# スクリプト・設定
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# data/ と runs/ は volume mount 前提（image に含めない）
VOLUME ["/app/data", "/app/runs"]

# pixi task を引数で指定して実行する想定
ENTRYPOINT ["pixi", "run"]
CMD ["train", "--config", "configs/default.yaml"]
