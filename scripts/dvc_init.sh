#!/bin/bash

# Check if required environment variables are set
if [ -z "$MAIN_PROJECT_DIR" ]; then
  echo "Error: MAIN_PROJECT_DIR is not set."
  exit 1
fi

if [ -z "$MINIO_ROOT_USER" ]; then
  echo "Error: MINIO_ROOT_USER is not set."
  exit 1
fi

if [ -z "$MINIO_ROOT_PASSWORD" ]; then
  echo "Error: MINIO_ROOT_PASSWORD is not set."
  exit 1
fi

if [ -z "$MINIO_BUCKET_NAME" ]; then
  echo "Error: MINIO_BUCKET_NAME is not set."
  exit 1
fi

echo "Setup project directory"
# Create the directory if it doesn't exist
mkdir -p /app/$MAIN_PROJECT_DIR
# Navigate to the project directory
cd /app/$MAIN_PROJECT_DIR


# Set up MinIO client
mc alias set myminio http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"

# Create a bucket
mc mb myminio/"$MINIO_BUCKET_NAME"

# Initialize DVC
echo "Initialize DVC in the project directory"
dvc init --no-scm

# Configure DVC remote storage
dvc remote add -d minio_remote s3://"$MINIO_BUCKET_NAME"
dvc remote modify minio_remote access_key_id "$MINIO_ROOT_USER"
dvc remote modify minio_remote secret_access_key "$MINIO_ROOT_PASSWORD"
dvc remote modify minio_remote endpointurl http://minio:9000  # Service name defined in docker-compose.yml


