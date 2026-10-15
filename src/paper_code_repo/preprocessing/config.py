from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


Backend = Literal["auto", "local", "docker"]


@dataclass
class PreprocessingConfig:
    input_dir: Path
    output_dir: Path
    study_uid: str | None = None
    mni_template: Path | None = None
    dcm2niix_path: Path | None = None
    t2_reference: Path | None = None
    flair_reference: Path | None = None
    t2_reference_mask: Path | None = None
    flair_reference_mask: Path | None = None
    allow_self_reference: bool = False
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    synthstrip_backend: Backend = "auto"
    synthseg_backend: Backend = "auto"
    synthstrip_image: str = "freesurfer/freesurfer:8.2.0"
    synthseg_image: str = "freesurfer/freesurfer:8.2.0"
    ants_image: str = "antsx/ants:v2.6.5"
    min_series_slices: int = 20
    force: bool = False
    keep_all_converted_series: bool = False

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        for name in (
            "mni_template",
            "dcm2niix_path",
            "t2_reference",
            "flair_reference",
            "t2_reference_mask",
            "flair_reference_mask",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, Path(value).expanduser().resolve())

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data
