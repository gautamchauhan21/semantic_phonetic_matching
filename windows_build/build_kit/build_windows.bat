@echo off
REM Build CompanyMatch.exe on a 64-bit Windows PC.
REM Needs: Python 3.13 (64-bit) from python.org, with the "py" launcher, and an internet connection.
REM Everything is created inside this folder (.venv, model_bundle, build, dist). Nothing else is changed.
setlocal
cd /d "%~dp0"

py -3.13 -m venv .venv || goto :fail
call .venv\Scripts\activate.bat || goto :fail
python -m pip install --upgrade pip || goto :fail
REM CPU-only torch keeps the program much smaller than the default download.
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu || goto :fail
python -m pip install -r requirements.txt || goto :fail
python prepare_model.py || goto :fail
pyinstaller --noconfirm CompanyMatch.spec || goto :fail

powershell -NoProfile -Command "Compress-Archive -Force -Path dist\CompanyMatch -DestinationPath dist\CompanyMatch-v1.0.0-windows.zip" || goto :fail
echo.
echo Built: dist\CompanyMatch\CompanyMatch.exe
echo Zip for download: dist\CompanyMatch-v1.0.0-windows.zip
echo Next: run the check in README_build.md before sharing it.
exit /b 0

:fail
echo.
echo BUILD FAILED. Scroll up for the first error.
exit /b 1
