#!/bin/bash

docker build -t docker.internal.starwit-infra.de/sae-anomaly-detection:$(poetry version --short) .
