#!/bin/bash

# Check if Kaggle is already installed, if not, install Kaggle
if ! command -v kaggle &> /dev/null
then
    echo "Kaggle could not be found, installing..."
    pip install --user kaggle
else
    echo "Kaggle is already installed."
fi

# Create the Kaggle directory if it does not exist
if [ ! -d "$HOME/.kaggle" ]; then
    mkdir ~/.kaggle
    echo "Kaggle directory created."
else
    echo "Kaggle directory already exists."
fi

# Move the Kaggle API key to the directory (Ensure that kaggle.json is in the current directory)
if [ -f "kaggle.json" ]; then
    mv kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    echo "Kaggle API key moved successfully."
else
    echo "kaggle.json not found in the current directory."
    exit 1
fi

# Download the dataset using the Kaggle CLI
kaggle competitions download -c 'image-matching-challenge-2023'

# Check if download was successful and unzip the dataset
if [ -f "image-matching-challenge-2023.zip" ]; then
    unzip image-matching-challenge-2023.zip
    echo "Dataset unzipped successfully."
else
    echo "Download failed or file does not exist."
fi
