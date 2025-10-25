@echo off
REM Startup script for NWMSU RAG Chatbot (Windows)
REM Run: run.bat

echo 🚀 Starting NWMSU RAG Chatbot
echo ==============================

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found!
    echo Please run: python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo 📦 Activating virtual environment...
call venv\Scripts\activate

REM Check if dependencies are installed
echo 🔍 Checking dependencies...
python -c "import streamlit, langchain, transformers" 2>nul || (
    echo ❌ Missing dependencies. Installing...
    pip install -r requirements.txt
)

REM Start the application
echo 🌐 Starting Streamlit application...
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the application
echo.

streamlit run app.py
