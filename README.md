# PINE: Perceptual Quality Assessment of Real-World Noise in Ultra-High-Definition Images

This repository is the official implementation of **PINE**, the method proposed in our IEEE Transactions on Image Processing (TIP) paper:

> **[Perceptual Quality Assessment of Real-World Noise in Ultra-High-Definition Images](https://ieeexplore.ieee.org/document/11724642)**
> *IEEE Transactions on Image Processing (TIP)*

PINE is a no-reference image quality assessment method for evaluating real-world noise in ultra-high-definition images. This implementation is built on the [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) training framework.

The repository provides the UNIQ-5k dataset, the PINE model, training and inference code, the selected training configuration, and a pretrained checkpoint. 

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

## UNIQ-5k Dataset

The UNIQ-5k dataset can be downloaded from either of the following links:

- [Google Drive](https://drive.google.com/drive/folders/1-5mx87UwToYP-j2EKkFvOybMEvz1mxa5?usp=drive_link)
- [Quark Cloud Drive](https://pan.quark.cn/s/4cb61b396a15)

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

> **[Perceptual Quality Assessment of Real-World Noise in Ultra-High-Definition Images](https://ieeexplore.ieee.org/document/11724642)**
> *IEEE Transactions on Image Processing (TIP)*

```bibtex
@ARTICLE{11724642,
  author={Zheng, Bolun and Weng, Zijie and Zhang, Qianyu and Zhang, Hua and Yan, Chenggang and Jia, Xu and Zhou, Wei},
  journal={IEEE Transactions on Image Processing},
  title={Perceptual Quality Assessment of Real-World Noise in Ultra-High-Definition Images},
  year={2026},
  volume={35},
  number={},
  pages={10757-10772},
  keywords={Modeling;Noise;Noise reduction;Image quality;Quality assessment;Conferences;Training;Computers;UHDTV;Computer vision;Blind image quality assessment;real-world noise;ultra-high-definition;IQA database;denoising evaluation metric},
  doi={10.1109/TIP.2026.3739609}
}
```
