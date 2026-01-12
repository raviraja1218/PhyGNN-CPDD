#!/bin/bash
# Simple FPOCKET installer

echo "=== Installing FPOCKET (Simplified) ==="

# First install dependencies
sudo apt-get update
sudo apt-get install -y wget unzip

# Download pre-compiled binary
echo "Downloading pre-compiled FPOCKET..."
wget https://raw.githubusercontent.com/Discngine/fpocket/master/fpocket/bin/fpocket -O fpocket_binary

if [ -f "fpocket_binary" ]; then
    chmod +x fpocket_binary
    sudo mv fpocket_binary /usr/local/bin/fpocket
    echo "✅ FPOCKET binary installed to /usr/local/bin/fpocket"
    
    # Test installation
    echo "Testing installation..."
    if command -v fpocket &> /dev/null; then
        echo "✅ FPOCKET successfully installed!"
        fpocket --help | head -5
    else
        echo "❌ FPOCKET not in PATH"
    fi
else
    echo "❌ Failed to download FPOCKET"
    echo "Alternative: Use Docker method"
    echo "docker pull discngine/fpocket:latest"
    echo "alias fpocket='docker run --rm -v \$(pwd):/data discngine/fpocket'"
fi
