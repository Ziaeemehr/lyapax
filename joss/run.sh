#!/usr/bin/env bash
# Build paper.pdf locally with the JOSS/Inara Docker image.
# Run from this directory: bash run.sh
docker run --rm \
    --volume $PWD:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/inara
