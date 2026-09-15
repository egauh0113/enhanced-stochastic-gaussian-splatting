# [ARSO 2026] Enhanced Stochastic Gaussian Splatting

[![Ubuntu 22.04 LTS](https://img.shields.io/badge/Ubuntu-22.04-orange)](https://releases.ubuntu.com/22.04/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)
[![Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository contains the official implementation of
"[Enhanced Stochastic Gaussian splatting via Laplace Negative Log-Likelihood and Geometry-Aware Losses](https://doi.org/10.1109/ARSO68304.2026.11536110)"
by [Eugene Auh](https://orcid.org/0000-0001-9349-5046) *et al.*
([RISE LAB](https://rise.skku.edu), Sungkyunkwan University, Republic fo Korea)

## Dependencies

This repository requires [gsplat](https://github.com/nerfstudio-project/gsplat) library along with its associated requirements.

The code has been tested on Ubuntu 22.04 LTS with Python 3.10, CUDA 13.0 and PyTorch.
Detailed dependencies are listed in *pyproject.toml* and *requirements.txt*.

## Usage

- Place the dataset (Mip-NeRF 360) under the *data* directory. (i.e., data/360_v2)
- Execute *scripts/basic_esgs.sh* for evaluation.
    ```bash
    cd scripts
    bash basic_esgs.sh
    ```
- The results will be generated in the *results* directory.

### Efficient Optimization

In *v0.1.1*, images are sampled and the KL divergence loss is computed using the currently active GSs.
To use entire set of GSs, set *full_sample* to *True* in *simple_trainer_esgs.py*, which corresponds to the results reported in the paper.

## Citation

If you use this repository in an academic context, please cite the following publication.
```bibtex
@inproceedings{auh2026enhanced,
  title={Enhanced Stochastic Gaussian Splatting via Laplace Negative Log-Likelihood and Geometry-Aware Losses},
  author={Auh, Eugene and Siriwa, Jenipher and Pico, Nabih and Balozi, Nehemiah and Moon, Hyungpil},
  booktitle={2026 IEEE International Conference on Advanced Robotics and its Social Impacts (ARSO)},
  pages={112--117},
  year={2026},
  organization={IEEE}
}
```

## Acknowledgement

We would like to thank to the authors of following works.
- [stochastic-gaussian-splatting](https://github.com/mezzelfo/stochastic-gaussian-splatting)
- [gsplat](https://github.com/nerfstudio-project/gsplat)

## Changelog

- *v0.1.0*: Initial release.
- *v0.1.1*: Perform MC sampling and KLD computation only on useful GSs.  
  - Reduced training time by ~50% while maintaining the same level of evaluation metrics. (tested on the  *garden* scene of the Mip-NeRF 360 dataset)
