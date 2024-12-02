import os
import subprocess
import logging
from functools import wraps

class DVCHandler:
    def __init__(self, project_dir: str, remote_name: str = None):
        self.project_dir = project_dir
        self.remote_name = remote_name
        self.dvc_present = self._check_dvc_presence()

        if not self.dvc_present:
            logging.warning("DVC is not installed in this environment. DVCHandler will not perform any actions.")

    def dvc_required(func):
        """Decorator to skip execution if DVC is not present."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.dvc_present:
                logging.warning(f"DVC is not present. Skipping execution of {func.__name__}.")
                return
            return func(self, *args, **kwargs)
        return wrapper

    def remote_required(func):
        """Decorator to skip execution if remote name is not defined."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.remote_name:
                logging.warning(f"Remote name is not defined. Skipping execution of {func.__name__}.")
                return
            return func(self, *args, **kwargs)
        return wrapper
    
    def _check_dvc_presence(self) -> bool:
        """Check if DVC is installed in the environment."""
        try:
            subprocess.run(["dvc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _init_dvc(self):
        """Initialize DVC in the specified project directory."""
        if not os.path.exists(self.project_dir):
            raise ValueError(f"Project directory {self.project_dir} does not exist.")
        
        os.chdir(self.project_dir)
        
        # Run dvc init
        subprocess.run(["dvc", "init"], check=True)
        
        # Set up the remote
        self._setup_remote(remote_name=self.remote_name)

    @dvc_required
    @remote_required
    def _setup_remote(self, remote_name: str):
        """Set up the DVC remote."""
        os.chdir(self.project_dir)
        
        # Run dvc remote add
        subprocess.run(["dvc", "remote", "add", "-d", remote_name, f"{remote_name}://"], check=True)
    
    @dvc_required
    def add(self, path: str):
        """Add a given path to DVC tracking."""
        os.chdir(self.project_dir)
        if not os.path.exists(path):
            raise ValueError(f"The path {path} does not exist.")
        
        # Run dvc add
        subprocess.run(["dvc", "add", path], check=True)

    @dvc_required
    def push(self, remote_name: str = None):
        """Push changes to the specified remote."""
        os.chdir(self.project_dir)
        if not remote_name: 
            remote_name = self.remote_name
        subprocess.run(["dvc", "push", "-r", remote_name], check=True)
