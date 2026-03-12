# JCSAM
[![DOI](https://zenodo.org/badge/1179423272.svg)](https://doi.org/10.5281/zenodo.18974559)
This repo is the official implementation for:\
[Enhanced Segmentation of Jaw Cysts Using a High-Frequency Awareness Adapted Segment Anything Model] submitted to *The Visual Computer*

## Installation
Following [Segment Anything](https://github.com/facebookresearch/segment-anything), `python=3.8.16`, `pytorch=1.8.0`, and `torchvision=0.9.0` are used in SAMUS.

1. Clone the repository.
    ```
    git clone https://github.com/xxd2977/jaw-cyst-segmentation.git
    cd jaw-cyst-segmentation
    ```
2. Create a virtual environment for jaw-cyst-segmentation and activate the environment.
    ```
    conda create -n jaw-cyst-segmentation python=3.8
    conda activate jaw-cyst-segmentation
    ```
3. Install Pytorch and TorchVision.
   (you can follow the instructions [here](https://pytorch.org/get-started/locally/))
5. Install other dependencies.
  ```
    pip install -r requirements.txt
  ```
## Checkpoints
We use checkpoint of SAM in [`vit_b`](https://github.com/facebookresearch/segment-anything) version.

## Training
Once you have the data ready, you can start training the model.
```
cd "/home/...  .../jaw-cyst-segmentation/"
python train.py --modelname JCSAM --task <your dataset config name>
```
## Testing
Do not forget to set the load_path in [./utils/config.py](https://github.com/xxd2977/jaw-cyst-segmentation/utils/config.py) before testing.
```
python test.py --modelname JCSAM --task <your dataset config name>
```
