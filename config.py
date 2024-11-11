import os

APP_NAME = 'ml-api'

ENV='dev'

ROOT_DIR_PATH =  os.path.dirname(os.path.abspath(__file__)) # Persistent path to root directory
TEMPLATES_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template_experiments') # Persistent path to templates directory
MODEL_MANAGER_MAIN_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '__main_project__') # Persistent path to main project directory
TEST_PROJECTS_DIRECTORY = '__test_projects__'

RESTRICTED_METADATA_FIELDS = ['path'] # Metadata fields that should not be exposed to users

HOST_REST_API = '0.0.0.0'
PORT_REST_API = 8000
