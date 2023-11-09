FROM python:3.8-slim

# Install system dependencies
RUN \
    set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
    python3-venv \
    ffmpeg \
    git \
    libgl1-mesa-glx \
    tesseract-ocr \
    tesseract-ocr-rus \
    ; \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip3 install -U pip && pip3 install -U wheel && pip3 install -U setuptools==59.5.0
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt && rm -r /tmp/requirements.txt

# Install OpenCV Python package
RUN pip3 install opencv-python-headless

# Install pytesseract
RUN pip3 install pytesseract

COPY . /code
WORKDIR /code

CMD ["bash"]