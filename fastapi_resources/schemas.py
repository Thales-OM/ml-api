from fastapi.responses import JSONResponse
from pydantic import BaseModel, StringConstraints
from uuid import UUID
from typing import Annotated, Optional, Union, List
import datetime
import json


# Define Pydantic models for requests
class FitRequest(BaseModel):
    X_train: List[List[float]]
    y_train: List[float]
    params: dict = {}
    loss: Annotated[str, 'Loss function name: "mse", "cross_entropy", "huber" etc. Only applicable to PyTorch models.'] = 'mse'
    optim: Annotated[str, 'Optimizer name: "sgd", "adam", "rmsprop" etc. Only applicable to PyTorch models.'] = 'adam'
    optim_args: Annotated[dict, 'Optimizer args.'] = {'lr': 0.001}
    epochs: int = 10

class PredictRequest(BaseModel):
    X_test: List[List[float]]

class ExperimentMetadataResponse(BaseModel):
    experiment_id: UUID # Encode UUID as str - avoid JSON serialization error
    created_dttm: Optional[datetime.datetime]
    last_changed_dttm: Optional[datetime.datetime]
    model_filename: str
    name: Optional[str]
    origin_experiment_id: Optional[UUID]
    parent_experiment_id: Optional[UUID]
    template_flg: bool

class ExperimentStatusResponse(BaseModel):
    experiment_id: UUID # Encode UUID as str - avoid JSON serialization error
    status: Annotated[str, StringConstraints(min_length=1)]

class BasicSuccessResponse(BaseModel):
    success: bool
    msg: Optional[str] = None

class BasicExperimentResponse(BaseModel):
    experiment_id: UUID # Encode UUID as str - avoid JSON serialization error

class PredictResponse(BaseModel):
    predictions: list

class CustomJSONEncoder(json.JSONEncoder):
    """Encoder for datetime, date, UUID serialization in JSONResponse"""
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)  # Convert UUID to string
        return super().default(obj)

def custom_decoder(dct):
    """Decoder for datetime, date, UUID deserialization in JSONResponse"""
    for key, value in dct.items():
        if isinstance(value, str):
            try:
                # Try to parse the string as an ISO 8601 datetime
                value = datetime.datetime.fromisoformat(value)
                dct[key] = value
            except ValueError:
                # If parsing fails, keep the original string
                pass
        if isinstance(value, str):
            try:
                # Attempt to convert string to UUID
                value = UUID(value)
                dct[key] = value
            except ValueError:
                pass  # If conversion fails, keep the original value
    return dct

class CustomJSONResponse(JSONResponse):
    def render(self, content: dict) -> bytes:
        # Use the custom encoder to serialize the content
        return json.dumps(content, cls=CustomJSONEncoder).encode("utf-8")

