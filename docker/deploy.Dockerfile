FROM nvcr.io/nvidia/tritonserver:25.05-py3

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:/app/deployment:/app/utils" \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN sed -i 's/^#\s*deb-src/deb-src/' /etc/apt/sources.list && \
    apt-get update

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv && \
    python3.10 -m venv /opt/venv && \
    apt-get clean && \
    apt-get install -y  \
    libxcb-xinerama0  \
    libxkbcommon-x11-0 \
    python3-pyqt5 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    x11-apps && \
    rm -rf /var/lib/apt/lists/*

RUN /opt/venv/bin/python -m pip install --upgrade pip

RUN /opt/venv/bin/pip install poetry
RUN mkdir -p /app/deployment

COPY pyproject.toml README.md ./
COPY deployment/*.py deployment
COPY deployment/core deployment/core
COPY deployment/resources deployment/resources
COPY deployment/models deployment/models
COPY utils utils

RUN poetry config virtualenvs.create false && \
    poetry update
