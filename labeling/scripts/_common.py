"""共通ユーティリティ（config 読み込み等）"""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """YAML config を読み込む"""
    with open(path) as f:
        return yaml.safe_load(f)
