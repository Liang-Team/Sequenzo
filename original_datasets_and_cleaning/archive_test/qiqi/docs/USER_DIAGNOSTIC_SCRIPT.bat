@echo off
REM Sequenzo 用户诊断脚本 (Windows)
REM 让用户运行这个脚本并把结果发给你

echo ======================================================================
echo Sequenzo 诊断脚本 (Windows)
echo ======================================================================
echo.

REM 1. 系统信息
echo [1] 系统信息:
python --version
where python
if defined CONDA_DEFAULT_ENV (
    echo Conda 环境: %CONDA_DEFAULT_ENV%
)
echo.

REM 2. 测试 NumPy
echo [2] 测试 NumPy:
python -c "import sys; import numpy; print(f'✓ NumPy {numpy.__version__}'); print(f'  路径: {numpy.__file__}')" 2>nul
if errorlevel 1 (
    echo ✗ NumPy 导入失败
    echo.
    echo ======================================================================
    echo 诊断结果: NumPy 安装有问题
    echo ======================================================================
    echo.
    echo 解决方案:
    echo   pip uninstall numpy -y
    echo   pip install "numpy>=2.0.0"
    pause
    exit /b 1
)
echo.

REM 3. 检查多个 NumPy
echo [3] 检查 NumPy 版本:
pip list | findstr /i numpy
if defined CONDA_DEFAULT_ENV (
    conda list numpy 2>nul
)
echo.

REM 4. 测试 Sequenzo
echo [4] 测试 Sequenzo:
python -c "import sequenzo; print('✓ Sequenzo 导入成功'); print(f'  路径: {sequenzo.__file__}')" 2>temp_error.txt
if errorlevel 1 (
    echo ✗ Sequenzo 导入失败
    echo.
    echo 错误信息:
    type temp_error.txt
    del temp_error.txt
    echo.
    echo ======================================================================
    echo 诊断结果: Sequenzo 有问题 (但 NumPy 正常^)
    echo ======================================================================
    echo.
    echo 建议的解决方案:
    echo 1. 重装 sequenzo:
    echo    pip uninstall sequenzo -y
    echo    pip install --no-cache-dir sequenzo
    echo.
    echo 2. 如果还不行，从源码安装:
    echo    pip install --no-binary sequenzo sequenzo
    pause
    exit /b 1
)

if exist temp_error.txt del temp_error.txt

echo.
echo ======================================================================
echo 诊断结果: 一切正常! ✓
echo ======================================================================
pause

