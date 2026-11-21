@echo off
echo ========================================================
echo        DOCUMIND BACKEND SERVICE - WAICY 2026
echo ========================================================
echo.

cd /d "%~dp0"

if not exist "venv" (
    echo [INFO] Virtual environment not found. Creating venv...
    python -m venv venv
    echo [INFO] Installing required dependencies...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo [INFO] Starting FastAPI server on http://localhost:8000
echo [INFO] Interactive Swagger Documentation: http://localhost:8000/docs
echo.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
