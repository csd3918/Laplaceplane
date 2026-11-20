@echo off
REM ------------------------------------------------------------
REM Builds dist\LaplaceGUI.exe from laplace_gui.py with PyInstaller.
REM The .exe runs on Windows without Python or any IDE installed.
REM Run this file from the folder that contains laplace_gui.py.
REM ------------------------------------------------------------
cd /d "%~dp0"

python -m pip install --upgrade pyinstaller numpy matplotlib
if errorlevel 1 goto failed

python -m PyInstaller --noconfirm --clean --onefile --windowed --name LaplaceGUI --hidden-import PIL._tkinter_finder laplace_gui.py
if errorlevel 1 goto failed

echo.
echo Done: %~dp0dist\LaplaceGUI.exe
pause
exit /b 0

:failed
echo.
echo BUILD FAILED - see the messages above.
pause
exit /b 1
