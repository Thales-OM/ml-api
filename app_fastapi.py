import os
from fastapi import FastAPI, HTTPException, Depends, Path, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, StringConstraints
from uuid import UUID
from resources.model_manager import ModelManager
import asyncio
from typing import Annotated, Optional, Union, List
import json
import datetime
from config import APP_NAME



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

def get_model_manager_instance(root_directory: str, templates_dir_path: Optional[str] = None) -> ModelManager:
        """
        Pytest dependency injection.
        Returns Singleton ModelManager instance rooted at main directory.
        """
        return ModelManager(root_directory=root_directory, templates_dir_path=templates_dir_path)

def create_app(root_directory: str, templates_dir_path: Optional[str] = None) -> FastAPI:
    # Define FastAPI app
    app = FastAPI()

    # Mount the static files directory
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # Set up the Jinja2 template directory
    templates = Jinja2Templates(directory="templates")

    # Handle unexpected exceptions
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={"detail": f"An unexpected error occurred: {exc}"},
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def read_root(request: Request):
        return templates.TemplateResponse("index.html", {'request': request, 'app_name': APP_NAME})

    # Define FastAPI endpoints
    @app.get("/experiments/list", response_model=List[ExperimentMetadataResponse])
    async def list_experiments(model_manager: ModelManager = Depends(lambda: get_model_manager_instance(root_directory ,templates_dir_path))):
        """List existing experiments. Returns a list of experiments with their metadata."""
        experiments_on_disk = model_manager.get_experiments_on_disk()
        response = [dict(experiment_id=experiment_id, **experiment_data['metadata']) for experiment_id, experiment_data in experiments_on_disk.items()]
        return CustomJSONResponse(content=response)

    @app.get("/experiments/active/status", response_model=List[ExperimentStatusResponse])
    async def active_experiments_status(model_manager: ModelManager = Depends(lambda: get_model_manager_instance(root_directory ,templates_dir_path))):
        """List status of active loaded Experiments."""
        response = model_manager.get_loaded_experiments_status()
        return CustomJSONResponse(content=response)

    @app.post("/experiments/{experiment_id}/branch", response_model=BasicExperimentResponse)
    async def branch_experiment(
        experiment_id: Annotated[UUID, Path(title="The ID of the base experiment")], 
        name: Annotated[Optional[str], Query(title="The name of the experiment")] = None,
        model_manager: ModelManager = Depends(lambda: get_model_manager_instance(root_directory ,templates_dir_path))
    ):
        """
        Generate new experiment (model instance + metadata) from an existing one.
        Saves to disk.
        """
        new_experiment_id = model_manager.branch_experiment(base_experiment_id=experiment_id, name=name)
        return CustomJSONResponse(content={"experiment_id": new_experiment_id})

    @app.post("/experiments/{experiment_id}/select", response_model=BasicSuccessResponse)
    async def select_experiment(
        experiment_id: Annotated[UUID, Path(title="The ID of the required experiment")], 
        model_manager: ModelManager = Depends(lambda: get_model_manager_instance(root_directory ,templates_dir_path))
    ):
        """Selects a given Experiment by ID from the project. Load from disk to memory if was not loaded yet."""
        experiment = model_manager.select_experiment(experiment_id=experiment_id)
        return JSONResponse(content={"success": True})

    @app.post("/experiments/save", response_model=BasicSuccessResponse)
    async def save_experiment(model_manager: ModelManager = Depends(lambda: get_model_manager_instance(root_directory ,templates_dir_path))):
        """Commit current Experiment state, overwrites experiment directory contents"""
        if not model_manager.get_current_experiment():
            # If no experiment currently selected
            return HTTPException(status_code=404, detail="No experiment currently selected to save")
        model_manager.save_experiment()
        return JSONResponse(content={"success": True})

    @app.post("/seed", response_model=BasicSuccessResponse)
    async def seed(
        value: Annotated[int, Query(title="The value of random seed")], 
        model_manager: ModelManager = Depends(lambda: get_model_manager_instance(root_directory ,templates_dir_path))
    ):
        """Set random seed for reproducibility"""
        model_manager.seed(value=value)
        return JSONResponse(content={"success": True})

    @app.post("/fit", response_model=BasicSuccessResponse)
    async def fit(request: FitRequest, model_manager: ModelManager = Depends(lambda: get_model_manager_instance(root_directory ,templates_dir_path))):
        """Trains the model in the currently selected Experiment (either Scikit-Learn or PyTorch) on the provided data."""
        model_manager.fit(request.X_train, request.y_train, request.params, request.loss, request.optim, request.optim_args, request.epochs)
        return JSONResponse(content={"success": True})

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest, model_manager: ModelManager = Depends(lambda: get_model_manager_instance(root_directory ,templates_dir_path))):
        """Outputs predictions from the model in the currently loaded Experiment."""
        predictions = model_manager.predict(request.X_test) # Expecting numpy.ndarray
        return JSONResponse(content={"predictions": predictions.tolist()})

    @app.get("/health", response_model=BasicSuccessResponse)
    async def health(model_manager: ModelManager = Depends(lambda: get_model_manager_instance(root_directory ,templates_dir_path))):
        """Endpoint for checking the service health status."""
        return JSONResponse(content={"success": True})
    
    return app
