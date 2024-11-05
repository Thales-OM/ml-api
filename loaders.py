import os
from sklearn.base import BaseEstimator
import yaml
import logging
from pydantic import BaseModel, StringConstraints, ValidationError, field_validator, constr
from typing import Annotated, Dict, Union, Optional, Any
from torch import nn
import torch
import joblib
from experiment import ExperimentMetadata


class ExperimentMetadataLoader():
    """Class entirely for loading metadata from a file and validating contents."""
    # Attributes for storing data (produced while working)
    curr_file_path: Optional[Annotated[str, StringConstraints(min_length=1)]] = None
    curr_metadata: Optional[ExperimentMetadata] = None

    def  __init__(self):
        pass

    def load_from_path(self, path: str) -> ExperimentMetadata:
        """
        Loads metadata from current file.
        Returns:
            ExperimentMetadata object    
        """
        self.curr_file_path = path
        metadata_dict = self._read_metadata_file(path)
        self.curr_metadata = ExperimentMetadata(**metadata_dict)
        return self.curr_metadata
    
    def load_from_dict(self, _dict: dict) -> ExperimentMetadata:
        """
        Loads metadata from a dictionary.
        Returns:
            ExperimentMetadata object
        """
        self.curr_file_path = None
        self.curr_metadata = ExperimentMetadata(**_dict)
        return self.curr_metadata

    def get_metadata(self) -> ExperimentMetadata:
        """Returns the loaded metadata object of ExperimentMetadata class."""
        return self.curr_metadata
    
    def save_to_path(self, save_path: str, metadata: ExperimentMetadata = None) -> None:
        """
        Saves loaded metadata to a file.
        If metadata argument is not provided, currently loaded metadata will be saved.
        """
        if metadata is None:
            self._write_metadata_file(save_path, self.curr_metadata.get_metadata_dict())
        else:
            self._write_metadata_file(save_path, metadata.get_metadata_dict())

    @staticmethod
    def _read_metadata_file(path: str) -> dict:
        """Reads metadata file contents into dict."""
        with open(path, 'r') as file:
            metadata_dict = yaml.safe_load(file)
        return metadata_dict

    @staticmethod
    def _write_metadata_file(path: str, metadata: dict) -> None:
        """Writes metadata dict to file."""
        with open(path, 'w') as metadata_file:
            yaml.dump(metadata, metadata_file, default_flow_style=False)

    @staticmethod
    def _valid_metadata_dict(metadata_obj: dict) -> bool:
        try:
            assert isinstance(metadata_obj, dict), f"Invalid object ttype received. Expecting dict."
            metadata_instance = ExperimentMetadata(**metadata_obj)
        except Exception as e:
            logging.warning(f'Invalid metadata dict received: {e}')
            return False
        return True
    
    @staticmethod
    def _valid_metadata_at_path(path: str) -> bool:
        try:
            metadata = ExperimentMetadataLoader._read_metadata_file(path)
            assert metadata is not None, f"Failed to read metadata from file, empty file: {path}"
            assert isinstance(metadata, dict), f"Failed to read metadata from file, expecting key-value structure on top level: {path}"
            assert ExperimentMetadataLoader._valid_metadata_dict(metadata), f"Failed to read metadata from file, invalid structure: {path}"
        except Exception as e:
            logging.warning(f'Failed to validate metadata at path: {path}. Error: {e}')
            return False
        return True

class ModelLoader():
    """
    Logic for loading models (Scikit-Learn, PyTorch APIs) from disk to memory.
    Load method is determined using file extesion: .pt/.pth = PyTorch, .skl = Scikit-Learn
    Saving is done to: .pt (torch.jit) - PyTorch, .skl (joblib) - Scikit-Learn
    """
    # Necessary input for savefile naming
    _savefile_basename: Annotated[str, StringConstraints(min_length=1)]
    # Attributes for storing data (produced while working)
    _curr_model: Optional[Union[nn.Module, BaseEstimator]] = None
    _curr_model_path: Optional[Annotated[str, StringConstraints(min_length=1)]] = None

    def __init__(self,  savefile_basename: str = 'model'):
        # super().__init__(_savefile_basename=savefile_basename)
        self._savefile_basename = savefile_basename

    def load_from_path(self, path: str) -> Union[nn.Module, BaseEstimator]:
        filename = os.path.basename(path)
        file_extension = os.path.splitext(filename)[1]
        if file_extension in ('.pth', '.pt'):
            model = torch.jit.load(path)
        elif file_extension == '.skl':
            model = joblib.load(path)
        else:
            raise ValueError(f'Unsupported file extension: "{file_extension}" when loading from {path}. Expecting: .pt, .pth or .skl.')
        self._curr_model = model
        self._curr_model_path = path
        return model
    
    def get_model(self) -> Union[nn.Module, BaseEstimator]:
        """Return currently loaded model."""
        self._curr_model

    def save_to_dir(self, model: Union[nn.Module, BaseEstimator], dir_path: str, file_basename: str = None) -> str:
        """
        Save model object to disk using appropriate method (torch.jit.save or joblib.dump).
        Can provide custom file base name (name without extension).
        Returns:
            Path to saved model file.
        """
        file_basename = self._savefile_basename if file_basename is None else file_basename 
        if isinstance(model, nn.Module):
            save_path = os.path.join(dir_path, file_basename + ".pt")
            model_scripted = torch.jit.script(model)
            model_scripted.save(save_path)
            return save_path
        elif isinstance(model, BaseEstimator):
            save_path = os.path.join(dir_path, file_basename + ".skl")
            joblib.dump(model, save_path)
            return save_path
        else:
            raise ValueError(f"Unsupported model type: {type(model)}. Expecting: nn.Module or BaseEstimator.")
