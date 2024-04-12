FROM --platform=linux/amd64 python:3.10.12-slim

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /src

# Copy only the requirements file to optimize rebuilding the Docker image
COPY pyproject.toml poetry.lock /src/

# Install Poetry
RUN pip install --no-cache-dir poetry

# Install project dependencies
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

# Copy the whole project to the container
COPY . /src

# Change working directory to src/app
WORKDIR /src/end_to_end_ml_project/app

# Install the end_to_end_ml_project package
RUN poetry install
# Expose the port where FastAPI will run
EXPOSE 8000

# Command to run the application
CMD ["poetry", "run","uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
