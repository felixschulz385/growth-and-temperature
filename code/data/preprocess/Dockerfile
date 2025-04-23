# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for scientific libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    libnetcdf-dev \
    libgdal-dev \
    proj-bin \
    libgeos-dev \
    libproj-dev \
    wget \
    curl \
    git \
    gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# First the basic requirements that change less frequently
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code
COPY . /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcs-key.json

# Set the entrypoint to run the workflow
ENTRYPOINT ["python", "-m", "workflow"]
CMD ["--config", "/config/workflow-config.json"]