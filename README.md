# This is hymba inference test

```bash
# test environment
# GPU             : A100
# CUDA version    : 12.5
# Ubuntu version  : 20.04
# Python version  : 3.12.11
# Pytorch version : 2.8.0+cu128
# Gcc-13          : This is required to install flash_attn.
```

```bash
git clone git@github.com:Dongyunkam/Hymba_test.git
cd Hymba_test
conda create -n 'Hymba_test' python=3.12
conda activate Hymba_test
```

```bash
# hymba uses old transformer libaray (for class DynamicCache ...).
# transformers 4.5x doesn't work for hymba.
pip install torch
pip install transformers==4.43
```

```bash
# You need to install some libraries.
# If there is some error during mamba-ssm installation, you can check <https://pypi.org/project/mamba-ssm/>.
pip install casual-conv1d mamba-ssm einops
pip install sentencepiece
pip install protobuf
```

```bash
# hymba requires flash_attn.
# But, Ubuntu 20.04 version is not good at flash_attn.
# You have to check if you have gcc-13 for flash_attn installation.
# If you don't have it, you need to install gcc-13 (sudo apt install gcc-13).
# And you need to install flash_attn through its git repository.

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

```bash
python main_test.py
```



