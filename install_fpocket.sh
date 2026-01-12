#!/bin/bash
# Install FPOCKET for baseline comparison

echo "=== Installing FPOCKET ==="

# Check if fpocket is already installed
if command -v fpocket &> /dev/null; then
    echo "FPOCKET is already installed"
    fpocket --version
    exit 0
fi

# Installation options
echo "Select installation method:"
echo "1. Install from source (recommended)"
echo "2. Use pre-built binary"
echo "3. Use Docker container"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "Installing from source..."
        
        # Install dependencies
        sudo apt-get update
        sudo apt-get install -y build-essential cmake git \
            libgsl-dev liblapack-dev libblas-dev
        
        # Clone and build
        git clone https://github.com/Discngine/fpocket.git
        cd fpocket
        mkdir build && cd build
        cmake ..
        make
        sudo make install
        
        echo "FPOCKET installed from source"
        ;;
    2)
        echo "Downloading pre-built binary..."
        
        # Download latest release
        wget https://github.com/Discngine/fpocket/releases/latest/download/fpocket-linux.tar.gz
        tar -xzf fpocket-linux.tar.gz
        sudo cp fpocket /usr/local/bin/
        
        echo "FPOCKET binary installed"
        ;;
    3)
        echo "Setting up Docker container..."
        
        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            echo "Docker not found. Installing..."
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            echo "Please log out and back in for Docker group changes"
            exit 1
        fi
        
        # Pull fpocket Docker image
        docker pull discngine/fpocket:latest
        
        echo "Docker image pulled. Create alias:"
        echo "alias fpocket='docker run --rm -v \$(pwd):/data discngine/fpocket'"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Verify installation
if command -v fpocket &> /dev/null || [[ $choice -eq 3 ]]; then
    echo "✅ FPOCKET installation successful"
    
    # Test with a sample protein
    echo "Testing installation with sample protein..."
    SAMPLE_PROTEIN="./data/PDBbind/refined-set/1a0q/1a0q_protein.pdb"
    
    if [[ -f "$SAMPLE_PROTEIN" ]]; then
        echo "Running FPOCKET on sample protein..."
        
        if [[ $choice -eq 3 ]]; then
            docker run --rm -v $(pwd):/data discngine/fpocket fpocket -f /data/$SAMPLE_PROTEIN
        else
            fpocket -f $SAMPLE_PROTEIN
        fi
        
        if [[ $? -eq 0 ]]; then
            echo "✅ FPOCKET test successful"
        else
            echo "❌ FPOCKET test failed"
        fi
    else
        echo "Sample protein not found at $SAMPLE_PROTEIN"
    fi
else
    echo "❌ FPOCKET installation failed"
    exit 1
fi

echo "=== Installation Complete ==="
