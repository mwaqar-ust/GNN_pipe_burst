"""Install the tutorial's pinned PyTorch build for this computer.

Run inside the new Conda environment. No CUDA Toolkit or driver is installed.
Detection and selection use only the standard library; importing this module
never installs a package. --dry-run prints the plan without changing anything.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

TORCH_VERSION = "2.7.1"
ROOT = Path(__file__).resolve().parents[1]
INDEX_ROOT = "https://download.pytorch.org/whl/"
# Conservative toolkit driver versions, not the lower minor-compatibility
# limits. cu126/cu128 use the latest update in their respective toolkit line.
DRIVER_MINIMUM = {
    "Linux": {"cu118": (520, 61, 5), "cu126": (560, 35, 5), "cu128": (570, 124, 6)},
    "Windows": {"cu118": (522, 6, 0), "cu126": (561, 17, 0), "cu128": (572, 61, 0)},
}
CUDA_VERSION = {"cu118": (11, 8), "cu126": (12, 6), "cu128": (12, 8)}


def version_tuple(value):
    parts = [int(x) for x in re.findall(r"\d+", str(value))[:3]]
    return tuple((parts + [0, 0, 0])[:3])


def infer_compute_capability(name):
    """Recognise common GeForce cards only when nvidia-smi omits compute_cap."""
    if re.search(r"\bRTX\s*30\d\d", name, re.I):
        return 8.6
    if re.search(r"\bRTX\s*40\d\d", name, re.I):
        return 8.9
    if re.search(r"\bRTX\s*50\d\d", name, re.I):
        return 12.0
    return None


def parse_gpu_rows(text, has_compute=True):
    result = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 2 or not re.match(r"^\d", row[1].strip()):
            continue
        name, driver = row[0].strip(), row[1].strip()
        try:
            capability = float(row[2].strip()) if has_compute else None
        except (ValueError, IndexError):
            capability = None
        result.append({"name": name, "driver": driver,
                       "compute_capability": capability or infer_compute_capability(name)})
    return result


def detect_nvidia(run=subprocess.run):
    """Return GPU metadata; no driver or failed detection is a CPU path."""
    for fields, has_compute in (("name,driver_version,compute_cap", True),
                                ("name,driver_version", False)):
        try:
            done = run(["nvidia-smi", "--query-gpu="+fields, "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=15, check=False)
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return {"gpus": [], "cuda_supported": None}
        if done.returncode == 0:
            gpus = parse_gpu_rows(done.stdout, has_compute)
            break
    else:
        gpus = []
    supported = None
    if gpus:
        try:
            header = run(["nvidia-smi"], capture_output=True, text=True, timeout=15, check=False)
            match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", header.stdout)
            if match:
                supported = tuple(map(int, match.groups()))
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {"gpus": gpus, "cuda_supported": supported}


def select_build(system, machine, detection, requested="auto"):
    """Pure selection function, also exercised with mocked hardware in tests."""
    if requested == "cpu":
        return "cpu", "CPU was explicitly selected."
    if system not in DRIVER_MINIMUM or machine.lower() not in {"x86_64", "amd64"}:
        return "cpu", "Automatic NVIDIA installation targets Windows/Linux x86-64; use CPU on this platform."
    gpus = detection.get("gpus", [])
    if not gpus:
        return "cpu", "No working NVIDIA driver was detected; CPU execution is available."
    driver = min(version_tuple(gpu["driver"]) for gpu in gpus)
    capabilities = [gpu.get("compute_capability") for gpu in gpus]
    # Unknown/future architectures do not justify a large speculative download.
    if any(cap is None or not (5.0 <= cap <= 12.0) for cap in capabilities):
        return "cpu", "GPU architecture could not be verified for the pinned builds; using CPU."
    requires_128 = any(cap >= 10.0 for cap in capabilities)
    choices = [requested] if requested != "auto" else (["cu128"] if requires_128 else ["cu126", "cu118"])
    for build in choices:
        if build not in CUDA_VERSION or (requires_128 and build != "cu128"):
            continue
        if driver < DRIVER_MINIMUM[system][build]:
            continue
        supported = detection.get("cuda_supported")
        if supported is not None and tuple(supported) < CUDA_VERSION[build]:
            continue
        return build, "Detected NVIDIA architecture and driver meet this build's conservative requirements."
    return "cpu", "The detected driver/architecture does not meet the selected CUDA build requirements; using CPU."


def install_command(build, system=None):
    system = platform.system() if system is None else system
    # Official macOS wheels have no '+cpu' suffix and come from PyPI.
    if system == "Darwin":
        return [sys.executable, "-m", "pip", "install", "--upgrade", "torch=="+TORCH_VERSION]
    return [sys.executable, "-m", "pip", "install", "--upgrade", "--only-binary=:all:",
            "torch=="+TORCH_VERSION+"+"+build, "--index-url", INDEX_ROOT+build]


PROBE = r'''
import json
result = {"import_ok": False, "cpu_ok": False, "cuda_ok": False}
try:
    import torch
    result.update(import_ok=True, version=str(torch.__version__), cuda_build=torch.version.cuda)
    x = torch.ones((8, 8), device="cpu")
    result["cpu_ok"] = bool(torch.all(x @ x == 8).item())
    if torch.cuda.is_available():
        try:
            x = torch.ones((8, 8), device="cuda")
            result["cuda_ok"] = bool(torch.all(x @ x == 8).item())
            torch.cuda.synchronize()
            result.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)))
        except Exception as exc:
            result["cuda_error"] = str(exc)
except Exception as exc:
    result["error"] = str(exc)
print(json.dumps(result))
'''


def probe_torch(run=subprocess.run):
    try:
        done = run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=90, check=False)
        for line in reversed(done.stdout.splitlines()):
            try:
                result = json.loads(line)
                if isinstance(result, dict) and "import_ok" in result:
                    return result
            except json.JSONDecodeError:
                continue
        return {"import_ok": False, "cpu_ok": False, "cuda_ok": False,
                "error": done.stderr[-2000:] or "PyTorch probe did not return a result."}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"import_ok": False, "cpu_ok": False, "cuda_ok": False, "error": str(exc)}


def isolated_environment():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    return (bool(conda_prefix) and conda_name not in {"", "base", "root"}) or sys.prefix != sys.base_prefix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpu", action="store_true", help="Install the CPU wheel without NVIDIA detection.")
    group.add_argument("--cuda", choices=tuple(CUDA_VERSION), help="Request a build, still checking the detected driver.")
    parser.add_argument("--dry-run", action="store_true", help="Print the hardware and pip plan without installing.")
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 12):
        parser.error("Use Python 3.12 in the iahr-gnn-hkust-tutorial environment. See docs/SETUP.md.")
    requested = "cpu" if args.cpu else (args.cuda or "auto")
    detection = {"gpus": [], "cuda_supported": None} if args.cpu else detect_nvidia()
    build, reason = select_build(platform.system(), platform.machine(), detection, requested)
    command = install_command(build)
    print(json.dumps({"python": sys.executable, "requested": requested, "selected_build": build,
                      "reason": reason, "hardware": detection, "command": command}, indent=2))
    if args.dry_run:
        return 0
    if not isolated_environment():
        parser.error("Activate iahr-gnn-hkust-tutorial first. This installer does not modify base/system Python.")
    current = probe_torch()
    wanted = TORCH_VERSION if platform.system()=="Darwin" else TORCH_VERSION+"+"+build
    ready = current.get("version")==wanted and current.get("cpu_ok") and (build=="cpu" or current.get("cuda_ok"))
    install_ok = True
    if not ready:
        print("Installing the selected PyTorch build. A CUDA download can take several minutes.", flush=True)
        install_ok = subprocess.run(command, check=False).returncode == 0
    result = probe_torch()
    if not install_ok or not result.get("cpu_ok"):
        if build == "cpu":
            print("PyTorch installation/probe failed. Check the network and the error above; then rerun setup.")
            return 1
        print("CUDA package installation/probe failed; trying the CPU wheel once.", flush=True)
        if subprocess.run(install_command("cpu"), check=False).returncode != 0:
            return 1
        result = probe_torch()
        if not result.get("cpu_ok"):
            print(json.dumps(result, indent=2))
            return 1
        build = "cpu"
    device = "cuda" if result.get("cuda_ok") and not args.cpu else "cpu"
    if build != "cpu" and device == "cpu":
        print("PyTorch works on CPU, but the CUDA execution test did not succeed. The tutorial will continue on CPU.")
    report = {"torch_requested": TORCH_VERSION, "selected_build": build, "device": device,
              "hardware": detection, "probe": result, "platform": platform.platform()}
    (ROOT/"outputs").mkdir(exist_ok=True)
    (ROOT/"outputs/setup_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"Ready: PyTorch {result.get('version')} | device: {device}")
    print("Next: python scripts/check_environment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
