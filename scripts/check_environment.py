"""Check dependencies, the active Python, and real CPU/CUDA tensor execution."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

# Also works when Python safe-path mode omits the script's own directory.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from install_torch import probe_torch

ROOT = HERE.parent
PACKAGES = {"numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "matplotlib": "matplotlib",
            "networkx": "networkx", "wntr": "wntr", "tsnet": "tsnet", "ipywidgets": "ipywidgets",
            "jupyterlab": "jupyterlab", "ipykernel": "ipykernel"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="Fail if the real CUDA tensor test fails.")
    args = parser.parse_args(argv)
    report = {"python": sys.executable, "python_version": sys.version.split()[0], "packages": {}, "errors": []}
    for package, module in PACKAGES.items():
        try:
            importlib.import_module(module)
            report["packages"][package] = importlib.metadata.version(package)
        except Exception as exc:
            report["errors"].append(f"{package}: {exc}")
    report["torch"] = probe_torch()
    report["device"] = "cuda" if report["torch"].get("cuda_ok") else "cpu"
    if not report["torch"].get("cpu_ok"):
        report["errors"].append("PyTorch could not perform a CPU tensor calculation.")
    if args.require_cuda and not report["torch"].get("cuda_ok"):
        report["errors"].append("CUDA tensor calculation did not succeed.")
    if sys.version_info[:2] != (3, 12):
        report["errors"].append("This environment should use Python 3.12.")
    for package, expected in {"numpy":"1.26.4", "wntr":"1.3.2", "tsnet":"0.3.1"}.items():
        if package in report["packages"] and report["packages"][package] != expected:
            report["errors"].append(f"Expected {package} {expected}; reinstall requirements.txt inside the tutorial environment.")
    print(json.dumps(report, indent=2))
    (ROOT/"outputs").mkdir(exist_ok=True)
    (ROOT/"outputs/environment_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    if report["errors"]:
        print("Environment check needs attention. See docs/SETUP.md.")
        return 1
    print(f"Environment ready. Notebook device: {report['device']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
