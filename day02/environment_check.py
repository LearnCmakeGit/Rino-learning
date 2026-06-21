"""Day 02: Environment inspection.

Run:

    python day02/environment_check.py

Goal:
- Understand where the code is running.
- Inspect CPU, memory, Python, NumPy and PyTorch.
- Check whether CUDA/GPU is available.
"""

import os
import platform
import shutil
import sys

print('=' * 60)
print('HOST')
print('=' * 60)
print(platform.node())

print('\n' + '=' * 60)
print('PYTHON')
print('=' * 60)
print(sys.version)

print('\n' + '=' * 60)
print('OPERATING SYSTEM')
print('=' * 60)
print(platform.platform())

print('\n' + '=' * 60)
print('CPU')
print('=' * 60)
print('Logical CPUs:', os.cpu_count())

print('\n' + '=' * 60)
print('DISK')
print('=' * 60)
usage = shutil.disk_usage('/')
print(f'Total: {usage.total / 1e9:.1f} GB')
print(f'Used : {usage.used / 1e9:.1f} GB')
print(f'Free : {usage.free / 1e9:.1f} GB')

print('\n' + '=' * 60)
print('NUMPY')
print('=' * 60)
try:
    import numpy as np
    print('Version:', np.__version__)
except Exception as e:
    print('NumPy not available:', e)

print('\n' + '=' * 60)
print('PYTORCH')
print('=' * 60)
try:
    import torch

    print('Version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())

    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        print('GPU count:', torch.cuda.device_count())
    else:
        print('Running on CPU only.')

except Exception as e:
    print('PyTorch not available:', e)
