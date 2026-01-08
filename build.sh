#!/bin/bash

# Build and push InfiniteTalk Docker image to Docker Hub

set -e

VERSION=0.13
IMAGE_NAME=infinite-talk
DOCKER_HUB_USER=jonathan28alkalay

FULL_IMAGE_NAME=${IMAGE_NAME}:${VERSION}
DOCKER_HUB_IMAGE=${DOCKER_HUB_USER}/${IMAGE_NAME}:${VERSION}

echo "Building Docker image: ${FULL_IMAGE_NAME}"
docker build --platform=linux/amd64 -t ${FULL_IMAGE_NAME} .

echo "Tagging for Docker Hub: ${DOCKER_HUB_IMAGE}"
docker tag ${FULL_IMAGE_NAME} ${DOCKER_HUB_IMAGE}

echo "Pushing to Docker Hub: ${DOCKER_HUB_IMAGE}"
docker push ${DOCKER_HUB_IMAGE}
echo "Successfully pushed ${DOCKER_HUB_IMAGE}"

echo "Done!"
