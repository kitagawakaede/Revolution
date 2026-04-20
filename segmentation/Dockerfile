# provide/segmentation サーバー学習・推論用 Dockerfile
#
# ビルド:
#   docker build -t provide-segmentation -f Dockerfile .
#
# 実行例 (学習):
#   docker run --rm --gpus all \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/runs:/app/runs \
#     -v $(pwd)/configs:/app/configs \
#     provide-segmentation train --config configs/default.yaml
#
# 変更可能: CUDA バージョンは base image 1 行目。torch cu121 wheel と整合必須。

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIXI_HOME=/root/.pixi \
    PATH=/root/.pixi/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://pixi.sh/install.sh | bash

WORKDIR /app

COPY pixi.toml pixi.lock ./
RUN pixi install --locked

COPY scripts/ ./scripts/
COPY configs/ ./configs/

VOLUME ["/app/data", "/app/runs"]

ENTRYPOINT ["pixi", "run"]
CMD ["train", "--config", "configs/default.yaml"]
