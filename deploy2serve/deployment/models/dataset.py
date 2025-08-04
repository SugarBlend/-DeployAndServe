from typing import List, Optional, Union
from pydantic import BaseModel, Field

from deploy2serve.deployment.models.common import OverrideFunctionality


class RoboflowDataset(BaseModel):
    name: str = Field(description="The code name of the data set, which will later be used as the section name in "
                                  "the converted data file.")
    api_key: str = Field(description="API key for connect Roboflow account.")
    workspace: str = Field(description="Workspace name on Roboflow platform.")
    version_number: str = Field(description="Project name on Roboflow platform.")
    model_format: str = Field(description="Version of dataset on Roboflow platform.")
    project_id: str = Field(description="Generate labels with chosen model pattern.")


class StandardDataset(BaseModel):
    name: str = Field(description="The code name of the data set, which will later be used as the section name in "
                                  "the converted data file.")
    images_url: str = Field(description="Link to the archive containing the data set.")
    annotations_url: str = Field(description="Link to an archive containing annotations to images.")


class Dataset(BaseModel):
    description: Union[StandardDataset, RoboflowDataset] = Field(description="A structure describing the contents of "
                                                                             "supported data sets.")
    calibration_frames: Optional[int] = Field(default=None, description="Quantity limiting set of calibration images.")
    exclude_frames: List[int] = Field(default=[],
                                      description="A set of image serial numbers that must be excluded from the "
                                                  "calibration stage in case of an unexpected error.")
    labels_generator: OverrideFunctionality = Field(description="A structure describing a 'LabelsGenerator' by module "
                                                                "and class name.")
    data_storage: OverrideFunctionality = Field(description="A structure describing a 'ChunkedDataset' by module and "
                                                            "class name.")

    class Config:
        arbitrary_types_allowed = True
