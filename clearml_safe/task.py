from clearml import Task
import requests
import logging
from .logger import BlankLogger


class SafeTask():
    """
    Wrapper around ClearML Task that allows to integrate ClearML into application 
    when no ClearML server is running to avoid raising exceptions.
    """
    def __init__(self):
        self.logger = BlankLogger()

    @staticmethod
    def is_clearml_server_up(url='http://localhost:8008'):
        try:
            response = requests.get(url)
            logging.debug(f'Response received from ClearML server: {response.status_code}')
            if response.status_code == 200:
                logging.info('ClearML server is confirmed up and running.')
                return True    
            logging.warning(f'Received unsuccessful status code from ClearML server: {response.status_code}')
            return False
        except requests.ConnectionError:
            logging.warning('Failed to connect to ClearML server at URL: {url}')
            return False
    
    @classmethod
    def init(cls, project_name, task_name, task_type='other', **kwargs):
        if cls.is_clearml_server_up():
            # If the server is up, return a ClearML Task object
            logging.info('ClearML server is up and running. Creating task...')
            return Task.init(project_name=project_name, task_name=task_name, task_type=task_type, **kwargs)
        else:
            # If the server is down, return a SafeTask object
            logging.warning('Returning SafeTask instance as ClearML server is down.')
            return cls()
    
    def connect(self, *args, **kwargs):
        logging.debug('Calling connect method on SafeTask instance.')
    
    def get_logger(self):
        return self.logger
