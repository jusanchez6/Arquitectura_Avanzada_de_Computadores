#!/bin/bash

# Gem5 and McPAT Installation Script
echo "Installing Gem5 and McPAT..."

# Update package list
sudo apt update

# Install all gem5 dependencies
echo "Installing Gem5 dependencies..."
sudo apt install -y \
    git \
    build-essential \
    scons \
    python3-dev \
    python3-six \
    python-is-python3 \
    libprotobuf-dev \
    python3-protobuf \
    protobuf-compiler \
    libgoogle-perftools-dev \
    libboost-all-dev \
    pkg-config \
    m4 \
    zlib1g \
    zlib1g-dev \
    libprotoc-dev \
    libpng-dev \
    libpng++-dev

# Clone and build gem5
echo "Cloning and building gem5..."
git clone https://github.com/gem5/gem5
cd gem5
echo "Building gem5 for ARM architecture (this may take 1-2 hours)..."
scons build/ARM/gem5.fast -j 7
cd ..

# Install McPAT dependencies
echo "Installing McPAT dependencies..."
sudo apt install -y g++-multilib libc6-dev-i386

# Clone and build McPAT
echo "Cloning and building McPAT..."
git clone https://github.com/HewlettPackard/mcpat.git
cd mcpat
make -j 6
cd ..

echo "=================================================="
echo "Installation completed!"
echo "Gem5 location: $(pwd)/gem5"
echo "McPAT location: $(pwd)/mcpat"
echo "=================================================="
echo "To test gem5: ./gem5/build/ARM/gem5.fast ./gem5/configs/learning_gem5/part1/simple-arm.py"
echo "To use McPAT: ./mcpat/mcpat -infile config.xml -print_level 1"