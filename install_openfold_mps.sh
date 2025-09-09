#!/bin/bash

# Installation script for OpenFold on Apple Silicon (MPS)
# This script installs OpenFold without CUDA dependencies for use with MPS

echo "Installing OpenFold for Apple Silicon (MPS support)..."
echo "=================================================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Warning: This script is optimized for macOS. Proceeding anyway..."
fi

# Install dependencies first
echo "Installing base dependencies..."
pip install biopython scipy ml-collections dm-tree einops

# Install dllogger (required dependency)
echo "Installing dllogger..."
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'

# For Apple Silicon, we need to install OpenFold without CUDA compilation
# We'll use environment variables to skip CUDA-specific compilation
echo "Installing OpenFold (CPU/MPS compatible version)..."

# Set environment to skip CUDA compilation
export CUDA_HOME=""
export FORCE_CUDA="0"

# Try to install OpenFold
# Note: This may fail if OpenFold requires CUDA, in which case we need a workaround
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307' --no-build-isolation 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Direct installation failed. Trying alternative approach..."
    
    # Clone and modify OpenFold for CPU/MPS only
    TEMP_DIR=$(mktemp -d)
    cd $TEMP_DIR
    
    echo "Cloning OpenFold repository..."
    git clone https://github.com/aqlaboratory/openfold.git
    cd openfold
    git checkout 4b41059694619831a7db195b7e0988fc4ff3a307
    
    # Modify setup.py to remove CUDA dependencies
    echo "Modifying setup.py for CPU/MPS compatibility..."
    
    # Create a modified setup.py that doesn't require CUDA
    cat > setup_mps.py << 'EOF'
import os
from setuptools import setup, find_packages

# Skip CUDA extensions for MPS compatibility
ext_modules = []

setup(
    name="openfold",
    version="1.0.0",
    author="AQ Laboratory",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=ext_modules,
    install_requires=[
        "torch",
        "biopython",
        "ml-collections",
        "numpy",
        "scipy",
        "dm-tree",
        "einops",
    ],
)
EOF
    
    # Install using the modified setup
    echo "Installing modified OpenFold..."
    python setup_mps.py install
    
    # Clean up
    cd /
    rm -rf $TEMP_DIR
fi

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import openfold; print('✅ OpenFold imported successfully')" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ OpenFold installation completed successfully!"
else
    echo "⚠️  OpenFold installation may have issues. Testing minimal import..."
    
    # Try importing just the essential modules
    python << EOF
try:
    from openfold.data import data_transforms
    from openfold.np import residue_constants
    from openfold.utils.loss import compute_predicted_aligned_error, compute_tm
    print("✅ Essential OpenFold modules imported successfully")
    print("   ESMFold should work with limited functionality")
except ImportError as e:
    print(f"❌ Failed to import OpenFold modules: {e}")
    print("   ESMFold may not work properly")
EOF
fi

echo ""
echo "Installation process completed!"
echo ""
echo "Note: On Apple Silicon, some OpenFold features requiring CUDA will not be available,"
echo "but ESMFold should work with MPS acceleration for the main model components."