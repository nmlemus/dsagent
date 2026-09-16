# Meridian execution environment for the `mmm-meridian` workflow.
# CPU by default; build with --build-arg CUDA=1 for the GPU extra (Linux + NVIDIA runtime).
FROM python:3.12-slim
ARG CUDA=0
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN if [ "$CUDA" = "1" ]; then pip install --no-cache-dir "google-meridian[and-cuda]"; \
    else pip install --no-cache-dir "google-meridian"; fi \
    && pip install --no-cache-dir pandas pyarrow plotly markdown
WORKDIR /workspace
