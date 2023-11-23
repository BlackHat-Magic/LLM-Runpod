#!/bin/bash

source .env

docker buildx build --no-cache \
    --build-arg "MODEL_ID=$MODEL_ID" \
    --build-arg "MAX_TOKEN_LENGTH=$MAX_TOKEN_LENGTH" \
    --build-arg "DEFAULT_TEMPERATURE=$DEFAULT_TEMPERATURE" \
    --build-arg "CACHE_DIR=$CACHE_DIR" \
    -f ./src/Dockerfile .