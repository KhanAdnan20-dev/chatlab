import sys, platform
import torch
print('Python:', sys.version.split()[0])
print('OS:', platform.platform())
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU device:', torch.cuda.get_device_name(0))

for pkg in ['transformers', 'peft', 'trl', 'unsloth', 'bitsandbytes', 'datasets', 'joblib', 'sklearn']:
    try:
        mod = __import__(pkg)
        print(f'{pkg}:', getattr(mod, '__version__', 'installed'))
    except Exception as e:
        print(f'{pkg}: not installed', e)
