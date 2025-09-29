# Zaif Trade Bot Development Container
FROM mcr.microsoft.com/devcontainers/python:3.11

# Install Node.js (already included via features, but explicit for clarity)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs

# Install additional system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        curl \
        git \
        htop \
        jq \
        tree \
        vim \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/workspaces/zaif-trade-bot"

# Create workspace directory
WORKDIR /workspaces/zaif-trade-bot

# Pre-install common Python packages for faster setup
RUN pip install --upgrade pip setuptools wheel

# Copy requirements files for better caching
COPY requirements*.txt ./

# Install Python dependencies (will be overridden by postCreateCommand)
RUN pip install -r requirements.txt || true
RUN pip install -r requirements-dev.txt || true

# Install common type stubs
RUN pip install types-requests types-psutil || true

# Set default shell
ENV SHELL /bin/bash