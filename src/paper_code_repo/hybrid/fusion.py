from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.files import write_dataframe, write_json


def _feature_columns(frame: pd.DataFrame, id_column: str, label_column: str) -> list[str]:
    return [col for col in frame.columns if col not in {id_column, label_column}]


def _patient_level(frame: pd.DataFrame, id_column: str, label_column: str) -> pd.DataFrame:
    if label_column in frame and frame.groupby(id_column)[label_column].nunique().gt(1).any():
        raise ValueError("Repeated examinations of one patient have conflicting labels")
    if not frame[id_column].duplicated().any():
        return frame.copy()
    numeric = frame.select_dtypes(include="number").columns.difference([id_column]).tolist()
    aggregations = {column: "mean" for column in numeric}
    if label_column in frame:
        aggregations[label_column] = "first"
    return frame.groupby(id_column, as_index=False).agg(aggregations)


def merge_feature_tables(
    radiomics: pd.DataFrame,
    deep_features: pd.DataFrame,
    id_column: str = "patient_id",
    label_column: str = "label",
    deep_prefix: str = "deep",
    radiomics_prefix: str = "radiomics",
) -> pd.DataFrame:
    for name, frame in {"radiomics": radiomics, "deep_features": deep_features}.items():
        if id_column not in frame.columns:
            raise ValueError(f"{id_column} column is required in {name}.")
    radiomics = _patient_level(radiomics, id_column, label_column)
    deep_features = _patient_level(deep_features, id_column, label_column)
    if label_column in radiomics.columns and label_column in deep_features.columns:
        labels = radiomics[[id_column, label_column]].merge(
            deep_features[[id_column, label_column]], on=id_column, how="inner", suffixes=("_radiomics", "_deep")
        )
        if (labels[f"{label_column}_radiomics"].astype(str) != labels[f"{label_column}_deep"].astype(str)).any():
            raise ValueError("Radiomics and deep feature tables disagree on patient labels")
    for frame_name, frame, prefix in (
        ("radiomics", radiomics, radiomics_prefix),
        ("deep_features", deep_features, deep_prefix),
    ):
        rename = {
            col: f"{prefix}__{col}"
            for col in _feature_columns(frame, id_column, label_column)
            if not col.startswith(f"{prefix}__")
        }
        frame.rename(columns=rename, inplace=True)
        if frame_name == "deep_features" and label_column in frame.columns:
            frame.drop(columns=[label_column], inplace=True)
    merged = radiomics.merge(deep_features, on=id_column, how="inner", validate="one_to_one")
    if label_column not in merged.columns and f"{label_column}_x" in merged.columns:
        merged.rename(columns={f"{label_column}_x": label_column}, inplace=True)
    return merged


def _selected_feature_names(selection_csv: str | Path, classifier: str | None = None) -> list[str]:
    selection = pd.read_csv(selection_csv)
    if "feature_name" not in selection.columns:
        raise ValueError(f"feature_name column is required in {selection_csv}")
    if classifier and "classifier" in selection.columns:
        selection = selection[selection["classifier"].astype(str).str.lower() == classifier.lower()]
        if selection.empty:
            raise ValueError(f"No selected features for classifier '{classifier}' in {selection_csv}")
    if "rank" in selection.columns:
        selection = selection.sort_values("rank")
    return selection["feature_name"].astype(str).tolist()


def _strip_known_prefix(name: str) -> str:
    for prefix in ("radiomics__", "deep__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _subset_selected_columns(
    frame: pd.DataFrame,
    selected_names: list[str],
    id_column: str,
    label_column: str,
    prefix: str,
) -> pd.DataFrame:
    keep = [id_column]
    if label_column in frame.columns:
        keep.append(label_column)
    rename: dict[str, str] = {}
    missing: list[str] = []
    for name in selected_names:
        source = name if name in frame.columns else _strip_known_prefix(name)
        if source not in frame.columns:
            missing.append(name)
            continue
        output_name = source if source.startswith(f"{prefix}__") else f"{prefix}__{source}"
        keep.append(source)
        rename[source] = output_name
    if missing:
        raise ValueError(f"Selected features not found in source table: {missing[:10]}")
    subset = frame.loc[:, keep].copy()
    subset.rename(columns=rename, inplace=True)
    return subset


def fuse_selected_feature_tables(
    radiomics: pd.DataFrame,
    deep_features: pd.DataFrame,
    radiomics_selection_csv: str | Path,
    deep_selection_csv: str | Path,
    id_column: str = "patient_id",
    label_column: str = "label",
    classifier: str | None = None,
) -> pd.DataFrame:
    radiomics = _patient_level(radiomics, id_column, label_column)
    deep_features = _patient_level(deep_features, id_column, label_column)
    radiomics_names = _selected_feature_names(radiomics_selection_csv, classifier=classifier)
    deep_names = _selected_feature_names(deep_selection_csv, classifier=classifier)
    selected_radiomics = _subset_selected_columns(
        radiomics,
        radiomics_names,
        id_column=id_column,
        label_column=label_column,
        prefix="radiomics",
    )
    selected_deep = _subset_selected_columns(
        deep_features,
        deep_names,
        id_column=id_column,
        label_column=label_column,
        prefix="deep",
    )
    if label_column in selected_deep.columns:
        selected_deep = selected_deep.drop(columns=[label_column])
    fused = selected_radiomics.merge(selected_deep, on=id_column, how="inner", validate="one_to_one")
    return fused


def save_selected_fused_features(
    radiomics_csv: str | Path,
    deep_features_csv: str | Path,
    radiomics_selection_csv: str | Path,
    deep_selection_csv: str | Path,
    output_csv: str | Path,
    id_column: str = "patient_id",
    label_column: str = "label",
    classifier: str | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[Path, Path]:
    radiomics = pd.read_csv(radiomics_csv)
    deep = pd.read_csv(deep_features_csv)
    fused = fuse_selected_feature_tables(
        radiomics,
        deep,
        radiomics_selection_csv=radiomics_selection_csv,
        deep_selection_csv=deep_selection_csv,
        id_column=id_column,
        label_column=label_column,
        classifier=classifier,
    )
    output = Path(output_csv)
    write_dataframe(output, fused)
    metadata = Path(metadata_path) if metadata_path else output.with_suffix(".json")
    write_json(
        metadata,
        {
            "fusion_mode": "selected_radiomics_plus_selected_deep",
            "radiomics_csv": str(radiomics_csv),
            "deep_features_csv": str(deep_features_csv),
            "radiomics_selection_csv": str(radiomics_selection_csv),
            "deep_selection_csv": str(deep_selection_csv),
            "classifier": classifier,
            "output_csv": str(output),
            "rows": int(len(fused)),
            "columns": int(fused.shape[1]),
            "id_column": id_column,
            "label_column": label_column,
            "radiomics_selected_count": int(sum(col.startswith("radiomics__") for col in fused.columns)),
            "deep_selected_count": int(sum(col.startswith("deep__") for col in fused.columns)),
        },
    )
    return output, metadata


def concatenate_features(radiomics: pd.DataFrame, deep_features: pd.DataFrame) -> pd.DataFrame:
    return merge_feature_tables(radiomics, deep_features)


def align_feature_tables(*tables: pd.DataFrame) -> list[pd.DataFrame]:
    if not tables:
        return []
    lengths = {len(table) for table in tables}
    if len(lengths) != 1:
        raise ValueError("All feature tables must have the same number of rows.")
    return [table.reset_index(drop=True) for table in tables]


def save_fused_features(
    radiomics_csv: str | Path,
    deep_features_csv: str | Path,
    output_csv: str | Path,
    id_column: str = "patient_id",
    label_column: str = "label",
    metadata_path: str | Path | None = None,
) -> tuple[Path, Path]:
    radiomics = pd.read_csv(radiomics_csv)
    deep = pd.read_csv(deep_features_csv)
    fused = merge_feature_tables(radiomics, deep, id_column=id_column, label_column=label_column)
    output = Path(output_csv)
    write_dataframe(output, fused)
    metadata = Path(metadata_path) if metadata_path else output.with_suffix(".json")
    write_json(
        metadata,
        {
            "radiomics_csv": str(radiomics_csv),
            "deep_features_csv": str(deep_features_csv),
            "output_csv": str(output),
            "rows": int(len(fused)),
            "columns": int(fused.shape[1]),
            "id_column": id_column,
            "label_column": label_column,
        },
    )
    return output, metadata
