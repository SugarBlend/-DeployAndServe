from typing import List, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from deploy2serve.deployment.models.common import ComponentOverride, ResolvedPath, is_url_urllib


class RoboflowDataset(BaseModel):
    name: str = Field(description="The code name of the data set, which will later be used as the section name in "
                                  "the converted data file.")
    api_key: str = Field(description="API key for connect Roboflow account.")
    workspace: str = Field(description="Workspace name on Roboflow platform.")
    version_number: int = Field(description="Project name on Roboflow platform.")
    model_format: str = Field(description="Version of dataset on Roboflow platform.")
    project_id: str = Field(description="Generate labels with chosen model pattern.")


class StandardDataset(BaseModel):
    name: str = Field(description="The code name of the data set, which will later be used as the section name in "
                                  "the converted data file.")
    images: str = Field(description="Link to the archive containing the data set.")
    annotations: str = Field(description="Link to an archive containing annotations to images.")


class CalibrationConfig(BaseModel):
    description: Optional[Union[StandardDataset, RoboflowDataset]] = Field(
        default=None, description="A structure describing the contents of supported data sets."
    )
    calibration_frames: Optional[int] = Field(default=None, description="Quantity limiting set of calibration images.")
    excluded_samples: List[int] = Field(
        default_factory=list,
        description="A set of image serial numbers that must be excluded from the calibration stage in case of an "
                    "unexpected error."
    )
    labels_generator: ComponentOverride = Field(description="A structure describing a 'LabelsGenerator' by module "
                                                            "and class name.")
    storage: ComponentOverride = Field(description="A structure describing a 'ChunkedDataset' by module and "
                                                        "class name.")
    cache_path: ResolvedPath = Field(
        default="checkpoints/",
        description="Path to the cache file. If it doesn't exist, the cache is written to the parent path of the "
                    "weights folder."
    )

    MAX_CALIBRATION_SAMPLES: int = 10000
    MAX_EXCLUDED_SAMPLES: int = 1000

    @field_validator("cache_path", mode="before")
    def validate_cache_path(cls, val: Union[str, Path]) -> Path:
        if isinstance(val, str):
            return Path(val)
        return val

    @classmethod
    @field_validator("excluded_samples", mode="before")
    def validate_excluded_samples(cls, val: List[int]) -> List[int]:
        if len(val) > cls.MAX_EXCLUDED_SAMPLES:
            raise ValueError(f"Too many excluded samples: {len(val)} > {cls.MAX_EXCLUDED_SAMPLES}")

        if val and min(val) < 0:
            raise ValueError("Excluded sample indices cannot be negative")

        return sorted(set(val))

    @classmethod
    @field_validator("calibration_frames", mode="before")
    def validate_calibration_frames(cls, val: Optional[int]) -> Optional[int]:
        if val is not None and val > cls.MAX_CALIBRATION_SAMPLES:
            raise ValueError(f"Max calibration samples too high: {val} > {cls.MAX_CALIBRATION_SAMPLES}")
        return val

    class Config:
        arbitrary_types_allowed = True
        validate_default = True
        use_enum_values = True
