import shutil
import pytest
import numpy as np
import random
import logging
from fastapi.testclient import TestClient
from app_fastapi import create_app
from config import TEMPLATES_DIR_PATH
from fixtures import generate_test_dir_path, test_client_setup_and_teardown, response_to_decoded_json
import os
import yaml
from resources.model_manager import ModelManager
from uuid import UUID


def test_list_expriments(test_client_setup_and_teardown):
    # Get test client, ModelManager instance and test project directory path
    client, test_model_manager, test_project_dir_path  = test_client_setup_and_teardown
    response = client.get("/experiments/list")
    # Chech response code
    assert response.status_code == 200, f"Received unexpected status code = {response.status_code}. Response content: {response.content}"
    # Decode response JSON
    response_json = response_to_decoded_json(response=response)
    # Check for ID uniqueness
    response_experiment_id_set = set()
    for experiment in response_json:
        experiment_id = experiment['experiment_id']
        if experiment_id in response_experiment_id_set:
            raise ValueError(f"Experiment ID duplicates encountered (ID = {experiment_id})")
        response_experiment_id_set.add(experiment_id)
    # Extract metadata filename
    metadata_filename = test_model_manager._metadata_filename
    # Gather true experiment list
    expected_response = []
    test_project_experiment_dir_paths = [os.path.join(test_project_dir_path, name) for name in os.listdir(test_project_dir_path) if os.path.isdir(os.path.join(test_project_dir_path, name))]
    for dir_path in test_project_experiment_dir_paths:
        experiment_id = test_model_manager._ensure_experiment_id_type(os.path.basename(dir_path))
        experiment_metadata_path = os.path.join(dir_path, metadata_filename)
        with open(experiment_metadata_path, 'r') as metadata_file:
            experiment_metadata = yaml.safe_load(metadata_file)
        logging.debug(f'Read metadata file at "{experiment_metadata_path}". Received: {experiment_metadata}')
        experiment_data = dict(experiment_id=experiment_id, **experiment_metadata)
        expected_response.append(experiment_data)
    # Check whether response matches expected
    ## Compare the two lists of dictionaries
    assert len(response_json) == len(expected_response), "Response length does not match example length"
    ## Convert lists of dicts to sets of frozensets for comparison
    response_set = {frozenset(item.items()) for item in response_json}
    expected_set = {frozenset(item.items()) for item in expected_response}
    assert response_set == expected_set, f"Response data does not match example data.\n\tExpected: {expected_response}\n\tReceived:{response_json}\n"

def test_branch_experiment(test_client_setup_and_teardown):
    # Get test client, ModelManager instance and test project directory path
    client, test_model_manager, test_project_dir_path  = test_client_setup_and_teardown
    get_list_response = client.get("/experiments/list")
    experiments_list_json = response_to_decoded_json(response=get_list_response)
    for num, experiment in enumerate(experiments_list_json, start=1):
        experiment_id = experiment['experiment_id']
        logging.debug(f'Testing "/experiments/{{experiment_id}}/branch" on item #{num} (ID={experiment_id})')
        post_branch_response = client.post(f"/experiments/{experiment_id}/branch")
        experiments_branch_json = response_to_decoded_json(response=post_branch_response)
        assert 'experiment_id' in experiments_branch_json, f"Expected 'experiment_id' key in response, but received: {experiments_branch_json}"
        assert isinstance(experiments_branch_json['experiment_id'], UUID), f'Received unsuccessful/invalid response to "/experiments/{{experiment_id}}/branch" (ID = {experiment_id}): {experiments_branch_json}'

def test_select_experiment(test_client_setup_and_teardown):
    # Get test client, ModelManager instance and test project directory path
    client, test_model_manager, test_project_dir_path  = test_client_setup_and_teardown
    get_list_response = client.get("/experiments/list")
    experiments_list_json = response_to_decoded_json(response=get_list_response)
    for num, experiment in enumerate(experiments_list_json, start=1):
        experiment_id = experiment['experiment_id']
        logging.debug(f'Testing "/experiments/{{experiment_id}}/select" on item #{num} (ID={experiment_id})')
        post_select_response = client.post(f"/experiments/{experiment_id}/select")
        experiments_select_json = response_to_decoded_json(response=post_select_response)
        assert experiments_select_json.get("success", False) == True, \
            f'Experiment (ID = {experiment_id}) returned unsuccessful response to "/experiments/{{experiment_id}}/select": {experiments_select_json}'

@pytest.mark.parametrize("value", 
    [
        (0,),         # Test with zero
        (1,),         # Test with a positive integer
        (1000000,),       # Test with a larger positive integer
        (42,),        # Test with a commonly used number
        (7,),         # Test with another small positive integer
    ]
)
def test_seed(value, test_client_setup_and_teardown):
    # Get test client, ModelManager instance and test project directory path
    client, test_model_manager, test_project_dir_path  = test_client_setup_and_teardown
    post_seed_response = client.post("/seed", params={"value": value})
    seed_response_json = response_to_decoded_json(response=post_seed_response)
    assert seed_response_json.get("success", False) == True, f'Returned unsuccessful response to "/seed" with value = {value}: {seed_response_json}'

def test_fit(test_client_setup_and_teardown):
    # Get test client, ModelManager instance and test project directory path
    client, test_model_manager, test_project_dir_path  = test_client_setup_and_teardown
    get_list_response = client.get("/experiments/list")
    experiments_list_json = response_to_decoded_json(response=get_list_response)
    for experiment in experiments_list_json:
        experiment_id = experiment['experiment_id']
        logging.debug(f'Testing "/experiments/fit" on item (ID={experiment_id})')
        post_select_response = client.post(f"/experiments/{experiment_id}/select")
        experiments_select_json = response_to_decoded_json(response=post_select_response)
        assert experiments_select_json.get("success", False) == True, \
            f'Experiment (ID = {experiment_id}) returned unsuccessful response to "/experiments/{{experiment_id}}/select": {experiments_select_json}'

def test_predict(test_client_setup_and_teardown):
    # TODO: Implement test_predict
    pass
