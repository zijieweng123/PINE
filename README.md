# PINE: Perceptual Quality Assessment of Real-World Noise in Ultra-High-Definition Images

This repository is the official implementation of **PINE**, the method proposed in our IEEE Transactions on Image Processing (TIP) paper:

> **Perceptual Quality Assessment of Real-World Noise in Ultra-High-Definition Images**
> *IEEE Transactions on Image Processing (TIP)*

PINE is a no-reference image quality assessment method for evaluating real-world noise in ultra-high-definition images. This implementation is built on the [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) training framework.

The repository provides the PINE model, training and inference code, the selected training configuration, and a pretrained checkpoint. Datasets, experiment outputs, visualization experiments, RAG assets, hyperparameter sweeps, and ablation studies are not included.

## Contents

- `pyiqa/archs/pine_arch.py`: complete PINE architecture, including multi-scale feature fusion, PSAM, PFEM, and the quality regression head.
- `pyiqa/data/hr_dataset.py`: fragment-based image preprocessing and dataset loader.
- `pyiqa/models/hr_iqa_model.py`: training and evaluation wrapper.
- `options/train_PINE.yml`: the selected PINE training configuration.
- `inference.py`: single-image inference entry point.
- `ckpt/PINE.pth`: pretrained PINE checkpoint.

## Installation

Python 3.8 or newer is required. Install the package and its dependencies with:

```bash
pip install -e .
```

The Swin backbone may be downloaded by `timm` on first use. A CUDA-enabled PyTorch installation is recommended for training.

## Training

```bash
python -m pyiqa.train --opt options/train_PINE.yml
```

## Inference

The pretrained checkpoint is provided at `ckpt/PINE.pth`. Run single-image inference with:

```bash
python inference.py \
  --image_path path/to/image.jpg \
  --model_path ckpt \
  --trained_model_file PINE.pth
```

## Citation

If you find PINE useful in your research, please cite our IEEE TIP paper:

> **Perceptual Quality Assessment of Real-World Noise in Ultra-High-Definition Images**
> *IEEE Transactions on Image Processing (TIP)*

The complete BibTeX entry will be added when the publication metadata is available.
