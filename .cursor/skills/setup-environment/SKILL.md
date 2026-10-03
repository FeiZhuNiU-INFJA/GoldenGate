---
name: setup-environment
description: >-
  Creates or refreshes the conda env extreme_quant (Python 3.12). Use when
  the user asks for 环境配置, 装环境, conda env, pip install, a missing
  lightgbm or torch import, or an Eastmoney ProxyError.
---

# Set up the extreme_quant environment

Python 3.12. The env name is `extreme_quant`. Work from the repo root. Do not install Tushare. Do not commit `dataset/`. Do not download market data as part of setup.

Non-interactive shells do not have `conda` on `PATH`. Load it before any conda command:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
```

If `conda` itself is missing, stop and say so.

## 1. Create or update

`conda env list` has no `extreme_quant`:

```bash
conda env create -f environment.yml
```

The env already exists:

```bash
conda env update -f environment.yml --prune
```

`environment.yml` uses the Tsinghua conda mirrors and does not install LightGBM. Ranking imports `lightgbm`, so always continue:

```bash
conda activate extreme_quant
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. Check the device

```bash
python -c "import torch, lightgbm; print(torch.__version__, 'mps=', torch.backends.mps.is_available(), 'cuda=', torch.cuda.is_available(), 'lgb=', lightgbm.__version__)"
```

Apple Silicon: the conda-forge PyTorch pin includes MPS. NVIDIA: that pin is not the CUDA build. After the env exists, install the CUDA build named in `environment.yml` (`pytorch-cuda=12.1` from the pytorch and nvidia channels), then run the check again. Neither MPS nor CUDA means CPU.

## 3. Proxy

An Eastmoney `ProxyError` means leave `EXTREME_QUANT_KEEP_PROXY` unset. `data/akshare_client.py` drops `http_proxy` / `https_proxy` and forces direct HTTP, including the macOS system proxy. Set `EXTREME_QUANT_KEEP_PROXY=1` only when the user explicitly wants the proxy kept.

## 4. Reply

Say whether the env was created or updated, which device PyTorch sees, and the LightGBM version.
