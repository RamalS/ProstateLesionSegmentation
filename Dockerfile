FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG TORCH_VERSION=2.7.0
ARG TORCHVISION_VERSION=0.22.0
ARG TORCHAUDIO_VERSION=2.7.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

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
RUN set -eux; \
    rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true; \
    if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        git \
        curl \
        ca-certificates \
        gnupg \
        build-essential \
        pkg-config \
        ffmpeg \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        nano \
        neovim \
        htop; \
    ARCH="$(dpkg --print-architecture)"; \
    if [ "$ARCH" = "amd64" ]; then \
        curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor --yes -o /usr/share/keyrings/google-chrome.gpg; \
        echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list; \
        apt-get update; \
        apt-get install -y --no-install-recommends google-chrome-stable; \
    else \
        echo "Skipping google-chrome-stable install on unsupported architecture: ${ARCH}"; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3 -m venv ${VENV_PATH} && \
    pip install --upgrade pip "setuptools<80" wheel

# Create non-root user
RUN useradd -ms /bin/bash trainer

# Create working directories
RUN mkdir -p /workspace /data /outputs /cache && \
    chown -R trainer:trainer /workspace /data /outputs /cache /opt/venv

WORKDIR /workspace

# Install PyTorch explicitly first.
# Compute-capability support depends on the selected wheel index.
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
    --index-url ${TORCH_INDEX_URL}

# Install the rest of requirements, but WITHOUT torch/torchvision/torchaudio in requirements.txt
COPY requirements.txt /workspace/requirements.txt
RUN grep -viE '^(torch|torchvision|torchaudio)([<>=].*)?$' /workspace/requirements.txt > /tmp/requirements.no-torch.txt && \
    pip install --no-cache-dir -r /tmp/requirements.no-torch.txt

# Re-install setuptools last to guarantee pkg_resources is available.
# Pinned to <80 because setuptools 80+ removed pkg_resources as a top-level
# site-packages module, which tensorboard still depends on.
RUN pip install --no-cache-dir "setuptools<80"

# Copy project files
COPY src /workspace/src
COPY scripts /workspace/scripts
COPY configs /workspace/configs

# Make shell scripts executable
RUN chmod +x /workspace/scripts/*.sh

USER trainer

EXPOSE 6006 8501

ENTRYPOINT ["/workspace/scripts/start.sh"]
CMD ["shell"]
