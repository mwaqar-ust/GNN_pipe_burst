@echo off
setlocal
cd /d "%~dp0"
where conda >nul 2>&1
if errorlevel 1 (
    echo Open Anaconda Prompt or Miniconda Prompt, then run this file again.
    exit /b 1
)
echo Setting up the separate environment: iahr-gnn-hkust-tutorial
call conda run -n iahr-gnn-hkust-tutorial python --version >nul 2>&1
if errorlevel 1 (
    call conda env create -f environment.yml
    if errorlevel 1 goto failed
) else (
    echo The tutorial environment already exists. Continuing its setup.
)
call conda run -n iahr-gnn-hkust-tutorial python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)"
if errorlevel 1 (
    echo This environment has the wrong Python version. See docs\SETUP.md before installing packages.
    exit /b 1
)
call conda run --no-capture-output -n iahr-gnn-hkust-tutorial python -m pip install -r requirements.txt
if errorlevel 1 goto failed
call conda run --no-capture-output -n iahr-gnn-hkust-tutorial python scripts\install_torch.py %*
if errorlevel 1 goto failed
call conda run --no-capture-output -n iahr-gnn-hkust-tutorial python -m ipykernel install --user --name iahr-gnn-hkust-tutorial --display-name "Python (IAHR GNN HKUST Tutorial)"
if errorlevel 1 goto failed
call conda run --no-capture-output -n iahr-gnn-hkust-tutorial python scripts\check_environment.py
if errorlevel 1 goto failed
echo.
echo Setup complete. In this prompt, run:
echo   conda activate iahr-gnn-hkust-tutorial
echo   jupyter lab
echo Open IAHR_GNN_HKUST_Tutorial.ipynb and select Python (IAHR GNN HKUST Tutorial).
exit /b 0

:failed
echo.
echo Setup stopped at the reported error. See docs\SETUP.md.
echo You can rerun this same setup command after correcting the cause.
exit /b 1
