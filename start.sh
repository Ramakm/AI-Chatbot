#!/bin/bash

# AI Chatbot - Quick Start Script
# This script sets up and runs the chatbot locally

set -e

echo "🤖 AI Chatbot - Quick Start Setup"
echo "=================================="
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

python_version=$(python3 --version | awk '{print $2}')
echo "✅ Found Python $python_version"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip > /dev/null
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

echo ""
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)" 2>/dev/null
echo "✅ NLTK data downloaded"

echo ""
echo "=================================="
echo "✨ Setup Complete!"
echo "=================================="
echo ""
echo "🚀 Starting the chatbot server..."
echo "📍 Open your browser at: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
