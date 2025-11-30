@echo off
REM Setup script for creating virtual environment and installing dependencies (Windows)

echo Setting up virtual environment for Subtask B...

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Download Italian spaCy model
echo Downloading Italian spaCy model...
python -m spacy download it_core_news_sm

echo.
echo Setup complete!
echo.
echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate, run:
echo   deactivate

pause

