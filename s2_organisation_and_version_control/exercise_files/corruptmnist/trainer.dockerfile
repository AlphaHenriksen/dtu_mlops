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

ENTRYPOINT ["python", "-u", "corruptmnist/models/train_model.py"]

# Build using 
# docker build -f trainer.dockerfile . -t trainer:latest

# Run
# docker run --name experiment1 trainer:latest

# Run with shared folder models/
# powershell
# docker run --name experiment1 -v ${PWD}/models:/models/ trainer:latest
# cmd
# docker run --name experiment1 -v "%cd%"/models:/models/ trainer:latest