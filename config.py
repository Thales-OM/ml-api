import os

APP_NAME = 'ml-api'

ENV='dev'

# ModelManager (core)
ROOT_DIR_PATH =  os.path.dirname(os.getcwd()) # Persistent path to root directory
TEMPLATES_DIR_PATH = os.path.join(os.getcwd(), 'template_experiments') # Persistent path to templates directory
MODEL_MANAGER_MAIN_DIR_PATH = os.path.join(os.getcwd(), '__main_project__') # Persistent path to main project directory
TEST_PROJECTS_DIRECTORY = '__test_projects__'

RESTRICTED_METADATA_FIELDS = ['path'] # Metadata fields that should not be exposed to users

# FastAPI
HOST_REST_API = '0.0.0.0'
PORT_REST_API = 8000

# gRPC
HOST_GRPC = '0.0.0.0'
PORT_GRPC = 50051

# Prometheus
PORT_PROMETHEUS_SERVER = 9090

# Grafana
GRAFANA_EXT_URL = 'http://localhost:3000'
GRAFANA_RESOURCES_DASHBOARD_UID = 'fe39almqz2u4gr'

# DVC
DVC_REMOTE_NAME = 'minio_remote'

# Minio
MINIO_UI_EXT_URL = 'http://localhost:9001'

# ClearML
CLEARML_UI_EXT_URL = 'http://localhost:8080'