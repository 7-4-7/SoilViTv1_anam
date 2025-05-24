#!/bin/bash

# Exit if any command fails
set -e

# Name of the competition
COMP_NAME="soil-classification-part-2"
# Create data directory if not exists
mkdir -p data
cd data

# Download data using Kaggle CLI
kaggle competitions download -c $COMP_NAME

# Unzip the downloaded data
unzip -o "${COMP_NAME}.zip"
