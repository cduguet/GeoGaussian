#!/bin/bash
docker run --gpus all --rm -v $(pwd):/app geogaussian:latest python3 train.py -s data/RealData --sparse_num 1