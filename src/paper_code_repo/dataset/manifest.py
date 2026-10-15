from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils.files import ensure_dir, write_dataframe, write_json
from .discovery import PatientRecord, records_to_frame


def build_manifest(records: Iterable[PatientRecord]) -> pd.DataFrame:
    return records_to_frame(records)


def save_manifest(records: Iterable[PatientRecord], csv_path: str | Path, json_path: str | Path | None = None) -> pd.DataFrame:
    frame = build_manifest(records)
    write_dataframe(csv_path, frame)
    if json_path is not None:
        write_json(json_path, frame.to_dict(orient="records"))
    return frame

