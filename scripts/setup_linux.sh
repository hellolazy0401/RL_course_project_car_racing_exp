#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install pip==24.3.1 setuptools==75.6.0 wheel==0.45.1 swig==4.3.0
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python -m pytest -q
