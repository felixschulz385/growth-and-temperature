#!/bin/bash

# Build the unified Docker image from the project root
PROJECT_ROOT=$(pwd)
GCP_PROJECT="ee-growthandheat"
IMAGE_NAME="data-processor"
VERSION=$(git rev-parse --short HEAD)

echo "Building Docker image: gcr.io/${GCP_PROJECT}/${IMAGE_NAME}:${VERSION}"
echo "Building Docker image: gcr.io/${GCP_PROJECT}/${IMAGE_NAME}:latest"

# Build the Docker image
docker build -t "gcr.io/${GCP_PROJECT}/${IMAGE_NAME}:${VERSION}" \
             -t "gcr.io/${GCP_PROJECT}/${IMAGE_NAME}:latest" \
             -f "${PROJECT_ROOT}/Dockerfile" \
             "${PROJECT_ROOT}"

echo "Image built successfully!"

# Ask if user wants to push the image to GCR
read -p "Push image to Google Container Registry? (y/n): " PUSH_IMAGE

if [[ "${PUSH_IMAGE}" == "y" || "${PUSH_IMAGE}" == "Y" ]]; then
    echo "Pushing images to GCR..."
    docker push "gcr.io/${GCP_PROJECT}/${IMAGE_NAME}:${VERSION}"
    docker push "gcr.io/${GCP_PROJECT}/${IMAGE_NAME}:latest"
    echo "Images pushed successfully!"
fi