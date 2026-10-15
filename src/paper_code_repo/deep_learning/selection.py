from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.files import write_dataframe, write_json


def _read_training_metadata(path: Path) -> dict[str, object]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    result = data.get("result", {})
    return {
        "metadata_path": str(path),
        "backbone": data.get("backbone"),
        "fine_tuning": data.get("fine_tuning"),
        "best_epoch": result.get("best_epoch"),
        "best_loss": result.get("best_loss"),
        "best_accuracy": result.get("best_metric"),
        "checkpoint_path": result.get("checkpoint_path"),
    }


def summarize_deep_fine_tuning_results(results_root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    root = Path(results_root)
    metadata_files = sorted(root.rglob("deep_training_metadata.json"))
    if not metadata_files:
        raise ValueError(f"No deep_training_metadata.json files were found under {root}")
    rows = [_read_training_metadata(path) for path in metadata_files]
    summary = pd.DataFrame(rows)
    summary["best_accuracy"] = pd.to_numeric(summary["best_accuracy"], errors="coerce")
    summary["best_loss"] = pd.to_numeric(summary["best_loss"], errors="coerce")
    summary = summary.sort_values(["backbone", "best_accuracy", "best_loss"], ascending=[True, False, True])
    best = summary.groupby("backbone", as_index=False).head(1).reset_index(drop=True)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "deep_finetuning_summary.csv"
    best_path = output / "deep_best_finetuning_by_backbone.csv"
    metadata_path = output / "deep_best_finetuning_by_backbone.json"
    write_dataframe(summary_path, summary)
    write_dataframe(best_path, best)
    write_json(
        metadata_path,
        {
            "results_root": str(root),
            "selection_rule": "Highest validation accuracy; ties are broken by lower validation loss.",
            "best_by_backbone": best.to_dict(orient="records"),
        },
    )
    return {"summary": summary_path, "best_by_backbone": best_path, "metadata": metadata_path}
