from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def slugify(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower())
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "item"


def normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("_", " ").replace("-", " ")).strip().upper()


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_dataframe(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def read_dataframe(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

