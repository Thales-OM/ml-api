from resources.model_manager import ModelManager
from torch import nn, optim
import torch
from fixtures import generate_test_dir_path, SimpleModel, model_manager_create_and_teardown, LOSS_FUNCTIONS, OPTIMIZERS
import shutil
import pytest
import numpy as np
import random
import logging

logging.basicConfig(level=logging.DEBUG)


def test_create_experiment_locally(model_manager_create_and_teardown):
    model_manager, test_dir_path = model_manager_create_and_teardown
    input_size = 10
    model = SimpleModel(input_size=input_size)
    model.eval()
    model_input = torch.randn(1, input_size)
    model_output_expected = model(model_input).detach().cpu().numpy()
    created_experiment_id = model_manager.create_experiment_locally(model=model, name='Example PyTorch model', template_flg=1)
    model_manager.select_experiment(created_experiment_id)
    model_output_received = model_manager.predict(X_test=model_input) # expecting to get numpy.ndarray
    np.array_equal(model_output_expected, model_output_received), f"Methods returned different predictions:/n/tLocal: {model_output_expected}, /n/tModelManager: {model_output_received}/n"

def test_branch_experiment(model_manager_create_and_teardown):
    model_manager, test_dir_path = model_manager_create_and_teardown
    input_size = 10
    model = SimpleModel(input_size=input_size)
    model.eval()
    model_input = torch.randn(1, input_size)
    model_output_expected = model(model_input).detach().cpu().numpy()

    created_experiment_id = model_manager.create_experiment_locally(model=model, name='Example PyTorch model', template_flg=1)
    branch_experiment_id = model_manager.branch_experiment(base_experiment_id=created_experiment_id, name='Test experiment branch')
    branch_experiment_obj = model_manager.select_experiment(branch_experiment_id)
    branch_model = branch_experiment_obj.get_model_obj()

    branch_model.eval()
    model_output_received = branch_model(model_input).detach().cpu().numpy()
    assert np.array_equal(model_output_expected, model_output_received), f"Methods returned different predictions:/n/tLocal: {model_output_expected}, /n/tModelManager: {model_output_received}/n"

def test_load_experiment(model_manager_create_and_teardown):
    model_manager, test_dir_path = model_manager_create_and_teardown
    input_size = 10
    model = SimpleModel(input_size=input_size)
    model.eval()
    model_input = torch.randn(1, input_size)
    model_output_expected = model(model_input).detach().cpu().numpy()

    created_experiment_id = model_manager.create_experiment_locally(model=model, name='Example PyTorch model', template_flg=1)
    branch_experiment_id = model_manager.branch_experiment(base_experiment_id=created_experiment_id, name='Test experiment branch')
    branch_experiment_obj = model_manager.load_experiment(branch_experiment_id)
    branch_model = branch_experiment_obj.get_model_obj()
    
    branch_model.eval()
    model_output_received = branch_model(model_input).detach().cpu().numpy()
    assert np.array_equal(model_output_expected, model_output_received), f"Methods returned different predictions:/n/tLocal: {model_output_expected}, /n/tModelManager: {model_output_received}/n"

# Parametrize the test with combinations of loss functions and optimizers
@pytest.mark.parametrize("loss_name, optimizer_name", [(loss_name, optimizer_name) 
                                                         for loss_name in LOSS_FUNCTIONS.keys() 
                                                         for optimizer_name in OPTIMIZERS.keys()])
def test_train_torch(model_manager_create_and_teardown, loss_name, optimizer_name):
    model_manager, test_dir_path = model_manager_create_and_teardown
    
    # Parameters
    input_size = 10
    output_size = 1
    n_samples_train = 100
    n_samples_test = 3
    seed = 42

    # Create a dummy dataset
    X_train = torch.randn(n_samples_train, input_size)
    X_test = torch.randn(n_samples_test, input_size)
    # Account for loss used for binary classification
    if loss_name == 'bce':
        y_train = torch.randint(0, 2, (n_samples_train,)).float()
    else:
        y_train = torch.randn(n_samples_train, output_size)

    model = SimpleModel(input_size=input_size, output_size=output_size)
    created_experiment_id = model_manager.create_experiment_locally(model=model, name='Test PyTorch model for training', template_flg=0)
    model_manager.select_experiment(created_experiment_id)

    # Define loss function for ModelManager and example training
    loss = loss_name
    criterion = LOSS_FUNCTIONS[loss_name]()

    # Define  optimizer for ModelManager and example training
    optim = optimizer_name
    optimizer = OPTIMIZERS[optimizer_name](model.parameters())

    # Set epochs (only one value is enough to test)
    epochs = 5

    model_manager.seed(seed)
    model_manager.fit(X_train=X_train, y_train=y_train, loss=loss, optim=optim, epochs=epochs)
    model_output_received = model_manager.predict(X_test=X_test) # expecting to get numpy.ndarray

    # Seed for local reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Train the model manually 
    model.train()  
    for epoch in range(epochs):  
        optimizer.zero_grad()  
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate results
    model_output_expected = model(X_test).detach().cpu().numpy()
    logging.debug(f"Received predictions ({loss_name}-{optimizer_name}):/n/tLocal: {model_output_expected}, /n/tModelManager: {model_output_received}/n")
    np.array_equal(model_output_expected, model_output_received), f"Methods returned different predictions:/n/tLocal: {model_output_expected}, /n/tModelManager: {model_output_received}/n"

    # TODO: test training Scikit-Learn