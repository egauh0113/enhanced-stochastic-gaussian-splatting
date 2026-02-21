# Enhanced Stochastic Gaussian Splatting

[![Ubuntu 22.04 LTS](https://img.shields.io/badge/Ubuntu-22.04-orange)](https://releases.ubuntu.com/22.04/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)
[![Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Enhanced stochastic Gaussian splatting via Laplace negative log-likelihood and geometry-aware losses

by [Eugene Auh](https://orcid.org/0000-0001-9349-5046), [RISE LAB](https://rise.skku.edu), Sungkyunkwan University, Republic fo Korea.

## Dependencies

This repository requires [gsplat](https://github.com/nerfstudio-project/gsplat) library and its requirements.

The code was tested on Ubuntu 22.04 LTS with Python 3.10, CUDA 13.0 and PyTorch.
The detailed dependencies are listed in *pyproject.toml* and *requirements.txt*.

## Usage

- Put the dataset (Mip-NeRF 360) under *data* directory. (i.e., data/360_v2)
- Execute *scripts/basic_esgs.sh* for evaluation.
    ```bash
    cd scripts
    bash basic_esgs.sh
    ```
- The results will be generated under *results* directory.

## Citation

If you use this repository in an academic context, please cite the following publication:
```bibtex
TBA
```

## Acknowledgement

We would like to thank to the authors of following works.
- [stochastic-gaussian-splatting](https://github.com/mezzelfo/stochastic-gaussian-splatting)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
