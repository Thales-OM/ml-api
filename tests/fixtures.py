from model_manager import ModelManager
from torch import nn, optim
import torch
import shutil
import pytest
import numpy as np
import random
import logging
from datetime import datetime
import os


# Model to test local experiment creation
class SimpleModel(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 5, output_size: int = 1):
        super(SimpleModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def generate_test_dir_path() -> str:
    """Creates path for test project directory"""
    TEST_PROJECT_DIRECTORY = '__test_projects__'
    test_dir_name = '__test_project__' + datetime.now().strftime("%Y%m%d%H%M%S")
    test_dir_path = os.path.join(os.getcwd(), TEST_PROJECT_DIRECTORY, test_dir_name)
    return test_dir_path

# Create ModelManager instance for susequent use in test
# Tests are less isolated but avoid spending time initializing a new project each time
@pytest.fixture(scope='module')
def model_manager():
    test_dir_path = generate_test_dir_path()
    manager = ModelManager(root_directory=test_dir_path)
    
    yield manager

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