#!/bin/bash

echo "Setting up Small Object Detector project..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{raw/{train,val,test},processed,annotations}
mkdir -p checkpoints
mkdir -p logs
mkdir -p inference/results
mkdir -p notebooks

# Download sample data structure
echo "Creating example data structure..."
python scripts/download_sample_data.py

echo ""
echo "Setup complete! 🎉"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Add your dataset to data/raw/train and data/raw/val"
echo "3. Configure training in configs/config.yaml"
echo "4. Start training: python train.py --config configs/config.yaml"
echo ""
echo "For more information, see README.md"
