# Use the official Python image as a base
FROM python:3.12-slim

# Set environment variables
ENV VENV_PATH=/app/.venv
ENV PATH="$VENV_PATH/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install poetry

# Create a virtual environment
RUN python -m venv $VENV_PATH

# Activate venv
RUN . $VENV_PATH/bin/activate

# Install dependencies
RUN pip install --upgrade pip && \
    poetry install --no-root --no-dev

# Copy the rest of the application code
COPY . .

# Make scripts executable
RUN chmod +x ./scripts/dvc_init.sh

# Expose the port that the FastAPI app will run on
EXPOSE 8000

# Command to run the FastAPI app using the activated virtual environment
CMD ["python", "app.py"]