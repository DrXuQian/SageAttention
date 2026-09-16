#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python - "$bundle_dir" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys
import torch

if sys.version_info[:2] != (3, 12) or torch.__version__.split('+', 1)[0] != '2.9.0':
    raise SystemExit('This wheel requires Python 3.12 and the existing PPU Torch 2.9.0.')
if not torch._C._GLIBCXX_USE_CXX11_ABI:
    raise SystemExit('This wheel requires C++11 ABI=1.')
root = pathlib.Path(sys.argv[1])
release = json.loads((root / 'release.json').read_text())
filename = release['wheel']
if pathlib.Path(filename).name != filename:
    raise SystemExit('Invalid wheel filename in manifest')
wheel = root / filename
if not wheel.is_file() or hashlib.sha256(wheel.read_bytes()).hexdigest() != release['wheel_sha256']:
    raise SystemExit('Wheel missing or SHA256 mismatch; fetch the complete artifact branch.')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--force-reinstall', str(wheel)], check=True)
print('Installed', release['version'], '; import from outside any SageAttention source checkout.')
PY
