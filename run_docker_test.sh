#!/bin/bash
docker run --gpus all --rm -v $(pwd):/app geogaussian:latest python3 train.py -s data/Replica-OFF2 --sparse_num 1