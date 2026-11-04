#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if ! command -v conda >/dev/null 2>&1; then
    echo "Open a terminal with Conda available, then run this script again."
    exit 1
fi
echo "Setting up the separate environment: iahr-gnn-hkust-tutorial"
if conda run -n iahr-gnn-hkust-tutorial python --version >/dev/null 2>&1; then
    echo "The tutorial environment already exists. Continuing its setup."
else
    conda env create -f environment.yml
fi
conda run -n iahr-gnn-hkust-tutorial python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "This environment needs Python 3.12. See docs/SETUP.md.")'
conda run --no-capture-output -n iahr-gnn-hkust-tutorial python -m pip install -r requirements.txt
conda run --no-capture-output -n iahr-gnn-hkust-tutorial python scripts/install_torch.py "$@"
conda run --no-capture-output -n iahr-gnn-hkust-tutorial python -m ipykernel install --user --name iahr-gnn-hkust-tutorial --display-name "Python (IAHR GNN HKUST Tutorial)"
conda run --no-capture-output -n iahr-gnn-hkust-tutorial python scripts/check_environment.py
printf '\nSetup complete. Run:\n  conda activate iahr-gnn-hkust-tutorial\n  jupyter lab\n'
echo "Open IAHR_GNN_HKUST_Tutorial.ipynb and select Python (IAHR GNN HKUST Tutorial)."
