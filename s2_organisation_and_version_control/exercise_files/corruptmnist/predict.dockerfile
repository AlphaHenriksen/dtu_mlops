# For all python projects
# Base image
FROM python:3.9-slim

# Install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# For this project specifically
# Get the data and necessary files for build
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY corruptmnist/ corruptmnist/
COPY data/ data/
COPY reports/ reports/
COPY reports/figures/ reports/figures/
COPY models/ models/

# Set working directory and install dependencies
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "corruptmnist/models/predict_model.py"]

# Build using 
# docker build -f predict.dockerfile . -t predict:latest

# Run
# docker run --name experiment1 predict:latest

# Run with docker volumes
# docker run --name predict --rm -v ${PWD}/models/checkpoint.pth:/models/checkpoint.pth -v ${PWD}/data/processed/example_images.npy:/example_images.npy predict:latest predict ../../models/checkpoint.pth ../../example_images.npy