FROM python:3.12-slim

# Built by an administrator, never by model-generated code. Runtime containers
# have no network and a read-only root filesystem.
RUN pip install --no-cache-dir \
    "pandas>=2" matplotlib markdown "altair>=5.5,<6" \
    "vl-convert-python>=1.7" jupyter-client ipykernel

WORKDIR /workspace
