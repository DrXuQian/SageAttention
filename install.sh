#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python - "$bundle_dir" "$@" <<'PY'
import argparse
import pathlib
import subprocess
import sys
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--verify-only', action='store_true')
args = parser.parse_args(sys.argv[2:])

if sys.version_info[:2] != (3, 12) or torch.__version__.split('+', 1)[0] != '2.9.0':
    raise SystemExit('This wheel requires Python 3.12 and the existing PPU Torch 2.9.0.')
if not torch._C._GLIBCXX_USE_CXX11_ABI:
    raise SystemExit('This wheel requires C++11 ABI=1.')
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
from verify_release import verify_release
release, wheel = verify_release(root)
print('Verified', release['channel'], 'PV=' + release['default_pv'], release['version'])
if args.verify_only:
    raise SystemExit(0)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--force-reinstall', str(wheel)], check=True)
print('Installed', release['version'], '; import from outside any SageAttention source checkout.')
PY
