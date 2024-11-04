from uuid import UUID
from typing import Union, Iterable, Tuple, Dict, Any, Optional, Annotated
from sklearn.base import BaseEstimator
from torch.nn import Module
from torch import nn, tensor, float32
from torch.nn.modules.loss import _Loss
from torch import optim
from pydantic import BaseModel, StringConstraints
from tqdm import tqdm
import torch
import numpy as np


class Experiment(BaseModel):
    """
    Logic for training model, outputting predictions.
    Signleton-like pattern to avoid parallel access to a model.
    """
    # Exclude from pydantic validation using Optional
    ## Attributes for instantiation pattern
    _instances: Optional[Dict[UUID, 'Experiment']] = dict()
    _initialized: Optional[bool] = False

    # Necessary user input
    id: UUID
    model: Union[BaseEstimator, Module]

    def __new__(cls, id: UUID, *args, **kwargs):
        if id not in cls._instances:
            instance = super(Experiment, cls).__new__(cls)
            cls._instances[id] = instance
            # instance.root_directory = root_directory
        return cls._instances[id]

    def __init__(self, id: UUID, model: Union[BaseEstimator, Module]):
        # Prevent re-initialization of already created instances
        if not self._initialized:
            # Validate user input and set attributes
            super().__init__(id=id, model=model)
            self._initialized = True
    
    def fit(self, X_train: Iterable, y_train: Iterable, params: dict = None, loss: str = 'mse', optim: str = 'adam', optim_args: dict = dict(), epochs: int = 10) -> None:
        """
        Trains a given model (either Scikit-Learn or PyTorch) on the provided data.

        Args:
            X_train (Iterable): Input features for training.
            y_train (Iterable): Target values for training.
            params (dict): Parameters for training - attributes that are set to model object (includes loss/criterion for Scikit-Learn API).
            loss (str): Loss function for training - only applicable for PyTorch models.
            epochs (int): Number of epochs for training - only applicable for PyTorch models.

        Returns:
            Trained model.
        """
        if isinstance(self.model, BaseEstimator):
            self._fit_sklearn(self.model, X_train, y_train, params)
        elif isinstance(self.model, Module):
            self._train_torch(self.model, X_train, y_train, params, loss, optim, epochs)
        else:
            raise ValueError("Model must be either a Scikit-Learn estimator or a PyTorch module.")

    @staticmethod
    def _get_torch_loss(loss_name: str) -> _Loss:
        # Map string loss function names to PyTorch loss functions
        loss_functions = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'bce': nn.BCELoss(),
            'cross_entropy': nn.CrossEntropyLoss(),
            'huber': nn.HuberLoss()
        }
        if loss_name in loss_functions:
            return loss_functions[loss_name]
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}. Choose from {list(loss_functions.keys())}.")
        
    @staticmethod
    def _get_torch_optimizer(optimizer_name: str, model_parameters: Iterable, optimizer_args: dict = dict()) -> optim.Optimizer:
        """
        Returns a PyTorch optimizer object based on the given name.

        Args:
            optimizer_name (str): The name of the optimizer (e.g., 'sgd', 'adam', 'adamw').

        Returns:
            torch.optim.Optimizer: An instance of the specified optimizer.
        
        Raises:
            ValueError: If the optimizer name is not recognized.
        """
        # Define a mapping of optimizer names to their corresponding PyTorch classes
        optimizers = {
            'sgd': optim.SGD,
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'rmsprop': optim.RMSprop,
            'adagrad': optim.Adagrad,
            'adamax': optim.Adamax,
            'lbfgs': optim.LBFGS
        }

        # Check if the optimizer name is valid
        if optimizer_name not in optimizers:
            raise ValueError(f"Optimizer '{optimizer_name}' is not recognized. Choose from {list(optimizers.keys())}.")

        # Handle args
        if optimizer_args:
            assert isinstance(optimizer_args, dict),  "optimizer_args must be a dict"
        else:
            optimizer_args = dict()
        
        # Get the optimizer instance
        optimizer = optimizers[optimizer_name](model_parameters, **optimizer_args)
        return optimizer

    @staticmethod
    def _set_torch_model_params(model: nn.Module, params: dict = dict()):
        for key, value in params.items():
            if hasattr(model, key):
                # If the model has the attribute, set it
                with torch.no_grad():
                    # If the attribute is a tensor (like weights or biases), use copy_
                    if isinstance(value, torch.Tensor):
                        getattr(model, key).copy_(value)
                    else:
                        # Otherwise, set it directly
                        setattr(model, key, value)
    
    @staticmethod
    def _train_torch(model: nn.Module, X_train: Iterable, y_train: Iterable, params: dict = dict(), loss: str = 'mse', optim: str = 'adam', optim_args: dict = dict(), epochs: int = 10) -> None:
        # Prepare the data
        X_tensor = tensor(X_train, dtype=float32)
        y_tensor = tensor(y_train, dtype=float32).view(-1, 1)

        # Handle model parameters
        if params:
            assert isinstance(params, dict),  "params must be a dict"
            # Set params
            Experiment._set_torch_model_params(model, params)

        # Get loss and optimizer
        criterion = Experiment._get_torch_loss(loss)
        optimizer = Experiment._get_torch_optimizer(optim, model.parameters(), optim_args)

        model.train()
        # Training loop
        print(f"Training PyTorch model (epochs = {epochs})")
        for epoch in tqdm(range(epochs)):
            model.train()  # Set the model to training mode
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(X_tensor)  # Forward pass
            curr_loss = criterion(outputs, y_tensor)  # Calculate loss
            curr_loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {curr_loss.item():.4f}')
    
    def _fit_sklearn(model: BaseEstimator,  X_train: Iterable, y_train: Iterable, params: dict = dict()) -> None:
        # Handle model parameters
        if params:
            model.set_params(**params)
        model.fit(X_train, y_train)

    def predict(self, X_test: Iterable) -> np.ndarray:
        if isinstance(self.model, nn.Module):
            self.model.eval()
            X_tensor = tensor(X_test, dtype=float32)
            predict =  self.model(X_tensor)
            predict = predict.detach().cpu().numpy()
            return predict
        elif isinstance(self.model, BaseEstimator) : 
            predict = self.model.predict(X_test)
            predict = np.array(predict)
            return predict
        else:
            raise ValueError("Model must be either PyTorch (nn.Module) or Scikit-learn (BaseEstimator) model")
        
    def get_model_obj(self):
        return self.model
