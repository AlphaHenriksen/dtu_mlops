FROM python:3.9
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN pip install wandb
COPY s4_debugging_and_logging/exercise_files/wandb_tester.py wandb_tester.py
ENTRYPOINT ["python", "-u", "wandb_tester.py"]

# From the dtu_mlops directory
# docker build -f s2_organisation_and_version_control/exercise_files/corruptmnist/wandb.dockerfile . -t wandbtest
# docker run -e WANDB_API_KEY=19ca3cbc3d8d24e692ebd54957d48f399d7acbec wandbtest