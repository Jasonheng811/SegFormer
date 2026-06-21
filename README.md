![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
[![License: NSCL](https://img.shields.io/badge/license-NSCL-blue.svg)](LICENSE)

# Paddy Leaf Disease Segmentation — SegFormer / MMSegmentation Framework

Final Year Project (Bachelor of Electrical Engineering, Universiti Malaya).
Transformer-based **semantic segmentation of paddy (rice) leaf diseases**, built on top of the
[SegFormer](https://github.com/NVlabs/SegFormer) architecture and the
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0) framework.

> **Scope of this repository.** This public repo contains the open-source **SegFormer /
> MMSegmentation training & evaluation framework** that the project is built on. The project's
> **custom model architecture, trained weights, dataset, and experimental results are kept in a
> separate private repository pending academic publication**, and are intentionally not included here.

---

## About the project

- **Author:** Heng Zi Xuan
- **Programme:** B. Eng. (Electrical), Universiti Malaya
- **Domain:** Semantic segmentation of paddy leaf disease lesions using transformer-based encoders
- **Foundation:** [SegFormer (NVlabs, NeurIPS 2021)](https://arxiv.org/abs/2105.15203) on
  [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0)

## What is included

- The stock SegFormer / MMSegmentation training and evaluation framework (`mmseg/`)
- Standard SegFormer **B0–B5** model configs (`local_configs/`, `configs/`)
- Generic tooling: training, testing, benchmarking, FLOPs/params, ONNX export (`tools/`)
- Demo, docs, and Docker setup

## What is *not* included (kept private)

- The project's custom model components and modifications
- Trained checkpoints / weights
- The dataset
- Thesis materials, experimental logs, and results

## Installation

This uses MMSegmentation v0.13.0 as the codebase. For full install and data-preparation
guidance, see the [MMSegmentation v0.13.0 docs](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

```bash
pip install torchvision==0.8.2
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
pip install -e . --user
```

## Usage

```bash
# Single-GPU training (example with a stock SegFormer-B1 config)
python tools/train.py local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py

# Single-GPU testing
python tools/test.py local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint.pth
```

## Acknowledgements

This project builds directly on the work of:

- **SegFormer** — Xie *et al.*, NVlabs ([repo](https://github.com/NVlabs/SegFormer))
- **MMSegmentation** — OpenMMLab ([repo](https://github.com/open-mmlab/mmsegmentation))

## License

This repository inherits the **NVIDIA Source Code License (NSCL)** of the upstream SegFormer
project — for **non-commercial** (research / evaluation) use only. See [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```
