@echo off
REM Clean Build Artifacts for Sequenzo (Windows)
REM Usage: scripts\clean_build.bat

echo ====================================================================
echo            Sequenzo Build Artifacts Cleaner (Windows)
echo ====================================================================
echo.
echo This script will remove all compiled extensions and build artifacts
echo that might cause import errors after upgrading dependencies.
echo.

REM Get the project root (parent directory of scripts\)
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
cd /d "%PROJECT_ROOT%"

echo Project root: %CD%
echo.

REM Ask for confirmation
set /p CONFIRM="Do you want to proceed? [Y/n]: "
if /i not "%CONFIRM%"=="Y" if /i not "%CONFIRM%"=="" (
    echo [CLEAN] Cancelled by user.
    exit /b 0
)

echo.
echo [CLEAN] Removing build directories...

REM Remove top-level build directories
if exist build\ (
    rmdir /s /q build
    echo   - Removed build\
)
if exist dist\ (
    rmdir /s /q dist
    echo   - Removed dist\
)
if exist .eggs\ (
    rmdir /s /q .eggs
    echo   - Removed .eggs\
)

REM Remove egg-info directories
for /d %%i in (*.egg-info) do (
    rmdir /s /q "%%i"
    echo   - Removed %%i
)

echo [CLEAN] Removing compiled extensions...

REM Remove compiled extensions (.pyd files)
for /r %%i in (*.pyd) do (
    del /q "%%i" 2>nul
    echo   - Removed %%i
)

REM Remove bytecode files
for /r %%i in (*.pyc) do (
    del /q "%%i" 2>nul
)
for /r %%i in (*.pyo) do (
    del /q "%%i" 2>nul
)

REM Remove Cython-generated C files (only in utils\ directories)
for /r sequenzo %%i in (utils\*.c) do (
    del /q "%%i" 2>nul
    echo   - Removed %%i
)

echo [CLEAN] Removing cache directories...

REM Remove cache directories
for /d /r %%i in (__pycache__) do (
    if exist "%%i" (
        rmdir /s /q "%%i" 2>nul
    )
)
for /d /r %%i in (.pytest_cache) do (
    if exist "%%i" (
        rmdir /s /q "%%i" 2>nul
    )
)

echo.
echo ====================================================================
echo [CLEAN] Done! Cleanup complete.
echo.
echo Next steps:
echo   1. Reinstall Sequenzo:
echo      pip install -e .  (for development)
echo      pip install --no-cache-dir sequenzo  (from PyPI)
echo.
echo   2. Verify installation:
echo      python -c "import sequenzo; print(sequenzo.__version__)"
echo.
pause

