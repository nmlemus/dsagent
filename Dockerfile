# DSAgent Docker Image
# Provides both CLI and API server for data science tasks
#
# Build variants:
#   docker build -t dsagent:latest .                           # Without LaTeX (~1GB)
#   docker build -t dsagent:full --build-arg INSTALL_LATEX=true .  # With LaTeX (~1.5GB)

FROM python:3.11-slim

# Build argument for LaTeX installation
ARG INSTALL_LATEX=false

# Labels
LABEL maintainer="DSAgent Contributors"
LABEL version="0.8.4"
LABEL description="AI-powered autonomous agent for data science"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DSAGENT_WORKSPACE=/workspace \
    DSAGENT_SESSIONS_DIR=/workspace

# Install system dependencies (base)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install LaTeX (optional, for :full variant)
RUN if [ "$INSTALL_LATEX" = "true" ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-xetex \
    latexmk \
    && rm -rf /var/lib/apt/lists/*; \
    fi

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash dsagent

# Create workspace directory
RUN mkdir -p /workspace && chown dsagent:dsagent /workspace

# Set working directory
WORKDIR /app

# Copy package files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package with API extras
RUN pip install --no-cache-dir ".[api]"

# Switch to non-root user
USER dsagent

# Create user config directory
RUN mkdir -p /home/dsagent/.dsagent

# Set workspace as volume
VOLUME ["/workspace"]

# Expose API port
EXPOSE 8000

# Default command: start API server
# Can be overridden to run CLI: docker run -it dsagent dsagent chat
CMD ["sh", "-c", "dsagent serve --host 0.0.0.0 --port ${PORT:-8000}"]
