#!/bin/bash

DEST="/root/mcs_pcr/datasets"
mkdir -p "$DEST"

# Install gdown
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    apt-get update && apt-get install -y python3 python3-pip
    pip3 install gdown
    # Ensure gdown is in PATH
    export PATH="$PATH:/usr/local/bin:$HOME/.local/bin"
fi

# Verify gdown installed
if ! command -v gdown &> /dev/null; then
    echo "ERROR: gdown installation failed, exiting."
    exit 1
fi

# Apartment
if [ ! -d "$DEST/Apartment" ]; then
    echo "Apartment folder not found, downloading..."
    gdown "https://drive.google.com/uc?export=download&id=1ZEdiMbWp8ls3RnZP62xM2suzC511OLmI" -O "$DEST/Apartment.tar.gz"

    if [ -f "$DEST/Apartment.tar.gz" ]; then
        echo "Extracting Apartment.tar.gz..."
        tar -xzf "$DEST/Apartment.tar.gz" -C "$DEST"
        rm -f "$DEST/Apartment.tar.gz"
        echo "Apartment done."
    else
        echo "ERROR: Apartment download failed."
    fi
else
    echo "Apartment folder already exists, skipping."
fi

# 5-Park
if [ ! -d "$DEST/5-Park" ]; then
    echo "5-Park folder not found, downloading..."
    gdown "https://drive.google.com/uc?export=download&id=1KG4Afa7rvEYpmoUvmuIHofX2ebjLDBug" -O "$DEST/5-Park.tar.gz"

    if [ -f "$DEST/5-Park.tar.gz" ]; then
        echo "Extracting 5-Park.tar.gz..."
        tar -xzf "$DEST/5-Park.tar.gz" -C "$DEST"
        rm -f "$DEST/5-Park.tar.gz"
        echo "5-Park done."
    else
        echo "ERROR: 5-Park download failed."
    fi
else
    echo "5-Park folder already exists, skipping."
fi

echo "All done!"