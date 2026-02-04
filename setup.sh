#!/bin/bash
# Music Skills CLI Setup Script

set -e

echo "Music Skills CLI Setup"
echo "====================="

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo "0")
if [ "$PYTHON_VERSION" -lt "3" ]; then
    echo "Error: Python 3 is required."
    exit 1
fi

echo "Python 3 found."

# Install dependencies
echo ""
echo "Installing dependencies..."
python3 -m pip install --break-system-packages \
    click>=8.1.0 \
    pyyaml>=6.0 \
    pydub>=0.25.1 \
    librosa>=0.10.0 \
    soundfile>=0.12.1 \
    numpy>=1.24.0 \
    tqdm>=4.65.0 \
    requests>=2.31.0 \
    beautifulsoup4>=4.12.0 \
    yt-dlp>=2023.9.0

# Install the package
echo ""
echo "Installing music-skills package..."
python3 -m pip install --break-system-packages -e .

# Verify installation
echo ""
echo "Verifying installation..."
music-skills --version

echo ""
echo "Setup complete!"
echo ""
echo "To get started, run:"
echo "  music-skills --help"
echo ""
echo "Or configure your paths in config.yaml"
