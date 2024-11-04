import os
from sklearn.base import BaseEstimator
from uuid import UUID, uuid4
import yaml
import logging
from pydantic import BaseModel, StringConstraints, ValidationError, field_validator, constr
from typing import Annotated, Dict, Union, Optional, Any, Iterable
from tqdm import tqdm
import shutil
from torch import nn
import torch
import joblib
from experiment import Experiment, ExperimentMetadata
from loaders import ExperimentMetadataLoader, ModelLoader
import numpy as np


TEMPLATES_DIR_PATH = './template_experiments'


class ModelManager(BaseModel):
    """
        Abstraction layer class to hide backend for loading/storing/saving/logging ml models (supoport for sklear/pytorch APIs).
        Signleton-like pattern to avoid parallel local model storage access.
        If an instance of the class targeting the same root_directory already exists - return that instance.
    """
    # Exclude from pydantic validation using Optional
    ## Attributes for instantiation pattern
    _instances: Optional[Dict[Annotated[str, StringConstraints(min_length=1)], 'ModelManager']] = dict()
    _initialized: Optional[bool] = False
    ##  Attributes for experiment management
    _experiments_on_disk: Optional[Dict[UUID, Dict[str, Any]]] = dict() # Experiments in project directory on disk, <experiment_id, <'path': path, 'metadata': metadata>>
    _loaded_experiments: Optional[Dict[UUID, Experiment]] = dict() # Experiments loaded to memory, <experiment_id:  Experiment object>
    _current_experiment: Optional[Experiment] = None
    ## Template experiments
    _templates_dir_path: Optional[str] = TEMPLATES_DIR_PATH
    ## Loader classes
    _metadata_loader: Optional[ExperimentMetadataLoader] = None
    _model_loader: Optional[ModelLoader] = None

    # User inputs (validate using pydantic.BaseModel)
    _root_directory: Annotated[str, StringConstraints(min_length=1)]
    _metadata_filename: Annotated[str, StringConstraints(min_length=1)]
    _model_savefile_basename: Annotated[str, StringConstraints(min_length=1)]


    def __new__(cls, _root_directory: str, *args, **kwargs):
        if _root_directory not in cls._instances:
            instance = super(ModelManager, cls).__new__(cls)
            cls._instances[_root_directory] = instance
            # instance.root_directory = root_directory
        return cls._instances[_root_directory]

    def __init__(self, _root_directory, _metadata_filename: str = 'metadata', _model_savefile_basename: str = 'model'):
        # Prevent re-initialization of already created instances
        if not self._initialized:
            # Set necessary attributes within classs
            self._metadata_loader = ExperimentMetadataLoader()
            self._model_loader = ModelLoader(savefile_basename=_model_savefile_basename)
            # Validate user input and set attributes
            super().__init__(_root_directory=_root_directory, _metadata_filename=_metadata_filename, _model_savefile_basename=_model_savefile_basename)
            print(f"Initializing ModelManager within project root: {_root_directory}")
            self._initialize_project()
            self._initialized = True
    
    def _initialize_project(self) -> None:
        """Scan the root directory for existing models and load their metadata."""
        # Ensure working dir exists
        if not os.path.exists(self._root_directory):
            os.makedirs(self._root_directory)
        # Populate with template models if needed
        self._populate_root_with_templates(overwrite=False)
        # Scan directory for existing models and load short info into memory
        valid_experiments_data = dict()
        root_directory_contents = os.listdir(self._root_directory)
        valid_path_load_errors = 0
        invalid_paths = 0
        successful_loads = 0
        total_paths = len(root_directory_contents)
        print("Initializing project in directory: ", self._root_directory)
        for name in tqdm(root_directory_contents):
            path = os.path.join(self._root_directory, name)
            try:
                if self._path_is_valid_experiment(path):
                    metadata_path = os.path.join(self._root_directory, self._metadata_filename)
                    metadata = self._metadata_loader._read_metadata_file(metadata_path)
                    valid_experiments_data[name] = {'path': path, 'metadata': metadata}
                    successful_loads += 1
                else:
                    invalid_paths += 1
            except Exception as e:
                valid_path_load_errors += 1
                logging.error(f'Failed to load experiment at {path}. Error: {e}')
        logging.info(f'From {total_paths} paths in project directory: {successful_loads} loaded successfully,\
            {valid_path_load_errors}  valid paths failed to load,\
            {invalid_paths} were not valid experiment paths.')
        self._experiments_on_disk = valid_experiments_data
    
    @staticmethod
    def _string_is_valid_experiment_id(experiment_id_str) -> bool:
        """Check if a string is a valid experiment ID."""
        try:
            experiment_id = UUID(experiment_id_str)
        except ValueError:
            return False
        return True
    
    def _populate_root_with_templates(self, overwrite: bool = False) -> None:
        for name in os.listdir(self._templates_dir_path):
            base_path = os.path.join(self._templates_dir_path, name)
            if self._path_is_valid_experiment(base_path):
                new_path = os.path.join(self._root_directory, name)
                if overwrite or not os.path.exists(new_path):
                    self._safe_copy_experiment_directory(base_path, new_path, overwrite)

    def _path_is_valid_experiment(self, path: str) -> bool:
        """Check if a path is a valid experiment directory."""
        if not os.path.exists(path):
            logging.warning(f'Path not found: {path}')
            return False
        if not os.path.isdir(path):
            logging.warning(f'Non-directory inside project root: {path}')
            return False
        if not self._string_is_valid_experiment_id(os.path.basename(path)):
            logging.warning(f'directory name is not a valid experiment ID: {os.path.basename(path)}')
            return False
        # No experiment metadata file
        if not any([subpath == self._metadata_filename and os.path.isfile(os.path.join(path, subpath)) for subpath in os.listdir(path)]):
            logging.warning(f'Directory with no metadata inside project root: {path}')
            return False
        # TODO: add model file validation
        # No saved model file
        # if not any([subpath == self._model_savefile_basename and os.path.isfile(os.path.join(path, subpath)) for subpath in os.listdir(path)]):
        #     logging.warning(f'Directory with no model data inside project root: {path}')
        #     return False
        # Invalid metadata file contents
        if not ExperimentMetadataLoader._valid_metadata_at_path(os.path.join(path, self._metadata_filename)):
            logging.warning(f'Invalid metadata at path: {os.path.join(path, self._metadata_filename)}')
            return False
        return True

    def branch_experiment(self, base_experiment_id: UUID, name: str = None) -> None:
        """
        Generate new experiment (model instance + metadata) from an existing one.
        Saves to disk.
        """
        # Generate new ID
        new_experiment_id = ModelManager._generate_experiment_id()
        # Create new experiment metadata
        base_experiment_path = self._get_existing_experiment_path(base_experiment_id)
        base_metadata_path = os.path.join(base_experiment_path, self._metadata_filename)
        metadata = self._read_metadata_file(base_metadata_path)
        new_metadata = metadata.copy()
        new_metadata['name'] = name
        new_metadata['origin_experiment_id'] = base_experiment_id
        new_metadata['parent_experiment_id'] = base_experiment_id
        new_metadata['template_flg'] = False
        new_metadata = ExperimentMetadata(**new_metadata).get_metadata_dict() # Validate input
        # Copy files to new path
        new_experiment_path = self._produce_experiment_path(new_experiment_id)
        self._safe_copy_experiment_directory(base_experiment_path, new_experiment_path, overwrite=False)
        # Rewrite metadata
        new_metadata_path = os.path.join(new_experiment_path, self._metadata_filename)
        self._metadata_loader._write_metadata_file(new_metadata_path, new_metadata)
        # Update info about experiments on disk
        self._experiments_on_disk[new_experiment_id] = {'path': new_experiment_path, 'metadata': new_metadata}

    def _safe_copy_experiment_directory(base_path: str, new_path: str, overwrite: bool = False) -> None:
        """Method for safely copying an experiment directory during new experiment creation."""
        # Check if the source directory exists
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Source experiment directory '{base_path}' does not exist.")
        # Check if the destination directory already exists
        if not overwrite and os.path.exists(new_path):
            raise FileExistsError(f"During new experiment creation: '{new_path}' directory already exists. Possible ID collision.")
        # Copy the directory
        shutil.copytree(base_path, new_path, dirs_exist_ok=overwrite)
        logging.debug('Experiment directory successfully copied to a new one')
    
    def _produce_experiment_path(self, experiment_id: UUID) -> str:
        """
            Logic for producing new experiment paths within the current project.
            Existing experiments should (if no error is present) abide by this logic.
        """
        return os.path.join(self._root_directory, str(experiment_id))
    
    def _produce_metadata_path(self, experiment_id: UUID) -> str:
        """
            Logic for producing path to metadata file for a new experiment within the current project.
            Existing experiments should (if no error is present) abide by this logic.
        """
        return os.path.join(self._root_directory, str(experiment_id), self._metadata_filename)
    
    def _get_existing_experiment_path(self, experiment_id: UUID) -> str:
        """
            Get path to experiment directory.
            Applicable to valid experiments present in ModelManager in-memory data (_experiments_on_disk).
        """
        if experiment_id in self._experiments_on_disk:
            return self._experiments_on_disk[experiment_id]['path']
        raise ValueError(f"No valid experiment found on disk with id: {experiment_id}")

    @staticmethod
    def _generate_experiment_id():
        """Generates a new safe unique experiment id."""
        return uuid4()

    def load_experiment(self, experiment_id: UUID) -> Experiment:
        """
        Loads an experiment from disk into the ModelManager's in-memory data and creates an Experiment object.
        Returns:
            Loaded experiment instance
        """
        # Check if experiment already in memory
        if experiment_id in self._loaded_experiments:
            self._current_experiment = self._loaded_experiments[experiment_id]
            return self._current_experiment
        
        experiment_path = self._get_existing_experiment_path(experiment_id)
        model_filename = self._experiments_on_disk[experiment_id]['metadata']['model_filename']
        model_path = os.path.join(experiment_path, model_filename)
        model = self._model_loader.load_from_path(model_path)
        experiment = Experiment(id=experiment_id, model=model)
        self._loaded_experiments[experiment_id] = experiment
        self._current_experiment = experiment
        return experiment

    def create_experiment_locally(self, model: Union[nn.Module, BaseEstimator], name: Optional[str] = None, template_flg: int = 0) -> str:
        """
        Creates a new experiment from a locally provided data.
        Returns:
            Created experiment path
        """
        new_experiment_id = self._generate_experiment_id()
        new_experiment_path = self._produce_experiment_path(new_experiment_id)
        os.makedirs(new_experiment_path) # Raise custom exception on collision ?
        new_model_path = self._model_loader.save_to_dir(model=model, dir_path=new_experiment_path, file_basename=self._model_savefile_basename)
        new_model_filename = os.path.basename(new_model_path)
        new_metadata_path = os.path.join(new_experiment_path, self._metadata_filename)
        new_metadata = ExperimentMetadata(name=name, origin_experiment_id=None, parent_experiment_id=None, model_filename=new_model_filename, template_flg=template_flg)
        self._metadata_loader._write_metadata_file(path=new_metadata_path, metadata=new_metadata.get_metadata_dict())
        self._experiments_on_disk[new_experiment_id] = {'path': new_experiment_path, 'metadata': new_metadata.get_metadata_dict()}
        logging.info(f'New experiment created (ID={new_experiment_id})(method=local)')
        return new_experiment_path

    def save_experiment(self):
        """Commit current Experiment state, overwrites experiment directory contents"""
        current_model = self._current_experiment.get_model_obj()
        current_experiment_id = self._current_experiment.id
        current_experiment_path = self._get_existing_experiment_path(experiment_id=current_experiment_id)
        current_model_path = self._model_loader.save_to_dir(model=current_model, dir_path=current_experiment_path, file_basename=self._model_savefile_basename)
        current_model_filename = os.path.basename(current_model_path)
        metadata = ExperimentMetadata(**self._experiments_on_disk[current_experiment_id]['metadata'])
        metadata.safe_setattr(key='model_filename', value=current_model_filename)
        self._experiments_on_disk[current_experiment_id]['metadata'] = metadata.get_metadata_dict()
        
    def fit(self, X_train: Iterable, y_train: Iterable, params: dict = None, loss: str = 'mse', optim: str = 'adam', optim_args: dict = dict(), epochs: int = 10) -> None:
        """
        Trains the model in the currently selected Experiment (either Scikit-Learn or PyTorch) on the provided data.

        Args:
            X_train (Iterable): Input features for training.
            y_train (Iterable): Target values for training.
            params (dict): Parameters for training - attributes that are set to model object (includes loss/criterion for Scikit-Learn API).
            loss (str): Loss function for training - only applicable for PyTorch models.
            optim (str): Optimizer name - only applicable for PyTorch models.
            optim_args (dict): Parameters for PyTorch optimizer - only applicable for PyTorch models.
            epochs (int): Number of epochs for training - only applicable for PyTorch models.

        Returns:
            None
        """
        self._current_experiment.fit(X_train=X_train, y_train=y_train, params=params, loss=loss, optim=optim, optim_args=optim_args, epochs=epochs)

    def predict(self, X_test: Iterable) -> np.ndarray:
        """
        Outputs predictions from the model in the currently loaded Experiment.
        """
        return self._current_experiment.predict(X_test=X_test)