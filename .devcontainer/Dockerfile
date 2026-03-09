# Base Image
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# Stop apt-get from asking for user input (like timezones)
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for GDAL, OpenCV, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    build-essential \
    python3-dev \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure gdal-config is on PATH
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Upgrade build tools and install GDAL bindings
RUN pip install --no-cache-dir --upgrade pip setuptools wheel numpy \
    && pip install --no-cache-dir "GDAL==$(gdal-config --version)"

WORKDIR /app

# Correct path assuming the Dockerfile is in the .devcontainer folder
COPY .devcontainer/requirements.txt .
RUN pip install --requirement requirements.txt --no-cache-dir

COPY . .

# Match the port in devcontainer.json
EXPOSE 8050

ENTRYPOINT ["/bin/bash"]