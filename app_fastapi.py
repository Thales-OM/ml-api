import os
from fastapi import FastAPI, HTTPException, Depends, Path, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, StringConstraints
from uuid import UUID
from core.model_manager import ModelManager
from typing import Annotated, Optional, Union, List
from config import APP_NAME, GRAFANA_EXT_URL, GRAFANA_RESOURCES_DASHBOARD_UID, MINIO_UI_EXT_URL, DVC_REMOTE_NAME, CLEARML_UI_EXT_URL
from fastapi_resources.schemas import *


def get_model_manager_instance(root_directory: str, templates_dir_path: Optional[str] = None) -> ModelManager:
        """
        Pytest dependency injection.
        Returns Singleton ModelManager instance rooted at main directory.
        """
        return ModelManager(root_directory=root_directory, templates_dir_path=templates_dir_path, dvc_remote_name=DVC_REMOTE_NAME)

def create_app(root_directory: str, templates_dir_path: Optional[str] = None) -> FastAPI:
    # Define FastAPI app
    app = FastAPI()

    # Ensure /static directory exists
    os.makedirs('fastapi_resources/static', exist_ok=True)
    
    # Mount the static files directory
    app.mount("/static", StaticFiles(directory="fastapi_resources/static"), name="static")

    # Set up the Jinja2 template directory
    templates = Jinja2Templates(directory="fastapi_resources/templates")

    # Handle unexpected exceptions
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={"detail": f"An unexpected error occurred: {exc}"},
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def read_root(request: Request):
        return templates.TemplateResponse("index.html", {
            'request': request, 
            'app_name': APP_NAME, 
            'grafana_url': GRAFANA_EXT_URL, 
            'dashboard_uid': GRAFANA_RESOURCES_DASHBOARD_UID, 
            'clearml_ui_url': CLEARML_UI_EXT_URL,
            'minio_ui_url': MINIO_UI_EXT_URL
            }
        )

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
    
    @app.post("/dvc/add/{experiment_id}", response_model=BasicSuccessResponse)
    async def dvc_add(
        experiment_id: Annotated[UUID, Path(title="The ID of the required experiment")], 
        model_manager: ModelManager = Depends(lambda: get_model_manager_instance(root_directory ,templates_dir_path))
    ):
        """Add files to DVC tracking for the given experiment."""
        """Pushes the current experiments state to DVC remote"""
        if not model_manager._dvc_handler.dvc_present:
            raise HTTPException(status_code=404, detail="DVC is not set up. Please configure DVC before pushing data.")
        if not model_manager._dvc_handler.remote_name:
            raise HTTPException(status_code=404, detail="No remote configured for DVC to push to.")
        model_manager.dvc_add(experiment_id=experiment_id)
        return JSONResponse(content={"success": True})
    
    @app.post("/dvc/push", response_model=BasicSuccessResponse)
    async def dvc_push(model_manager: ModelManager = Depends(lambda: get_model_manager_instance(root_directory ,templates_dir_path))):
        """Pushes the current experiments state to DVC remote"""
        if not model_manager._dvc_handler.dvc_present:
            raise HTTPException(status_code=404, detail="DVC is not set up. Please configure DVC before pushing data.")
        if not model_manager._dvc_handler.remote_name:
            raise HTTPException(status_code=404, detail="No remote configured for DVC to push to.")
        model_manager.dvc_push()
        return JSONResponse(content={"success": True})

    return app
