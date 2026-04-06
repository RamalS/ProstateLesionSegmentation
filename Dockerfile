FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    TORCH_HOME=/cache/torch \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3 -m venv ${VENV_PATH} && \
    pip install --upgrade pip setuptools wheel

# Create non-root user
RUN useradd -ms /bin/bash trainer

# Create working directories
RUN mkdir -p /workspace /data /outputs /cache && \
    chown -R trainer:trainer /workspace /data /outputs /cache /opt/venv

WORKDIR /workspace

# Install Python dependencies first for Docker layer caching
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# Copy project files
COPY src /workspace/src
COPY scripts /workspace/scripts
COPY configs /workspace/configs

# Make shell scripts executable
RUN chmod +x /workspace/scripts/*.sh

USER trainer

EXPOSE 6006

ENTRYPOINT ["/workspace/scripts/start.sh"]
CMD ["shell"]
