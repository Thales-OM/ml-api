from core.model_manager import ModelManager
from torch import nn, optim
import torch
import shutil
import pytest
import numpy as np
import random
import logging
from datetime import datetime
import os
from config import TEST_PROJECTS_DIRECTORY, TEMPLATES_DIR_PATH
from app_fastapi import create_app
from fastapi.testclient import TestClient
from requests import Response
from typing import Union, Tuple, Generator, Literal, List
import json
from app_fastapi import custom_decoder


# Model to test local experiment creation
class SimpleModel(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 5, output_size: int = 1):
        super(SimpleModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )
    def forward(self, x):
        return self.layers(x)
    
def generate_test_dir_path() -> str:
    """Creates path for test project directory"""
    test_dir_name = '__test_project__' + datetime.now().strftime("%Y%m%d%H%M%S")
    test_dir_path = os.path.join(os.getcwd(), TEST_PROJECTS_DIRECTORY, test_dir_name)
    return test_dir_path

# Create ModelManager instance for susequent use in test
# Tests are less isolated but avoid spending time initializing a new project each time
@pytest.fixture(scope='module')
def model_manager_create_and_teardown() -> Generator[Tuple[ModelManager, str], None, None]:
    test_dir_path = generate_test_dir_path()
    manager = ModelManager(root_directory=test_dir_path, templates_dir_path=TEMPLATES_DIR_PATH)
    
    yield manager, test_dir_path

    # Teardown: Delete ModelManager instance and its directory
    try:
        del manager
    finally:
        shutil.rmtree(test_dir_path) # Ensure test project directory deletion

# Define constant maps for criterion/optimizer testing
LOSS_FUNCTIONS = {
    'mse': nn.MSELoss,
    'mae': nn.L1Loss,
    # 'bce': nn.BCELoss, # Not using BCELOSS because requires separate model for binary classification
    'cross_entropy': nn.CrossEntropyLoss,
    'huber': nn.HuberLoss
}

OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'rmsprop': optim.RMSprop,
    'adagrad': optim.Adagrad,
    'adamax': optim.Adamax,
}

@pytest.fixture(scope="module")
def test_client_setup_and_teardown() -> Generator[Tuple[TestClient, ModelManager, str], None, None]:
    test_project_dir_path = generate_test_dir_path()
    # Create the app and client
    app_fastapi = create_app(root_directory=test_project_dir_path, templates_dir_path=TEMPLATES_DIR_PATH)
    client = TestClient(app_fastapi)
    # Instantiate ModelManger Singleton
    model_manager = ModelManager(root_directory=test_project_dir_path, templates_dir_path=TEMPLATES_DIR_PATH)

    yield client, model_manager, test_project_dir_path

    # Teardown: Delete ModelManager instance and its directory
    try:
        del model_manager
    finally:
        shutil.rmtree(test_project_dir_path) # Ensure test project directory deletion

def response_to_decoded_json(response: Response) -> Union[dict, list]:
    """Correctly decode received JSON (for datetime, UUID custom serialized values)."""
    response_json = response.json()
    json_str = json.dumps(response_json)
    decoded_json = json.loads(json_str, object_hook=custom_decoder) # Deserialize data (for datetime values)
    return decoded_json

def generate_train_sample(input_size: int = 10, n_samples: int = 100, target_type: Literal["continuous", "binary"] = "continuous") -> Tuple[List[List[float]], List[float]]:
    """Generate training data."""
    # Independent variable (X)
    X = [np.random.uniform(0, 10, size=input_size).tolist() for obs in range(n_samples)]
    # True parameters
    slopes = np.random.uniform(0, 10, size=input_size)
    intercept = 1.0
    if target_type == 'continuous':
        # Generate dependent variable (Y) with some noise
        noise = np.random.normal(0, 1, n_samples).tolist()  # Gaussian noise
        Y = [(sum(slopes * np.array(sample)) + intercept + eps) for sample, eps in zip(X, noise)]
        return X, Y
    elif target_type == 'binary':
        # Generate a continuous variable Y
        continuous_Y = [sum(slopes * np.array(sample)) + intercept for sample in X]
        # Use a threshold to convert to binary (e.g., 0.5)
        threshold = np.mean(continuous_Y)  # You can choose a different threshold if needed
        Y = [1 if value > threshold else 0 for value in continuous_Y]
        return X, Y
    raise ValueError('Unsupported target_type, expecting: "continuous", "binary".')
 