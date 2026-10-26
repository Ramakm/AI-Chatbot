@echo off
REM AI Chatbot - Quick Start Script for Windows

echo 🤖 AI Chatbot - Quick Start Setup
echo ==================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python is installed
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

echo.
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo 📥 Installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -q -r requirements.txt
echo ✅ Dependencies installed

echo.
echo 📚 Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)" 2>nul
echo ✅ NLTK data downloaded

echo.
echo ==================================
echo ✨ Setup Complete!
echo ==================================
echo.
echo 🚀 Starting the chatbot server...
echo 📍 Open your browser at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py
pause
