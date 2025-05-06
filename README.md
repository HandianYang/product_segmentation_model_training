# Product Segmentation Model Training

## Description

A script for training instance segmentation model for supermarket products detection, using [Detectron2](https://github.com/facebookresearch/detectron2/tree/main) library with [ResNet-50-FPN](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml) backbone model.

## How to use

### Step 1. Environment setup

#### [Option 1] Use Docker image (Recommended)

+ Prerequisites
  - Hardware:
    * GPU: any CUDA 12.1-capable (RTX 3060 Ti or better)
    * RealSense: D435 or D435i
  - Software:
    * GPU driver: >= 525.60
    * Docker: >= 20.10
    * NVIDIA Container Toolkit (nvidia-docker2)
  ```bash
  # under root directory of project
  source docker_run.sh
  ``` 

#### [Option 2] Build local machine environment

+ Install/Prepare all [dependencies for Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#requirements):
  - Linux or macOS with Python ≥ 3.7
  - PyTorch ≥ 1.8 and torchvision that matches the PyTorch installation. Install them together at pytorch.org to make sure of this
  - OpenCV is optional but needed by demo and visualization


+ Install `gdown` and Detectron2:
  ```bash
  pip install \
    gdown \
    ninja \
    wheel \
    --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
  ```

### Step 2. Obtain backbone model and datasets

+ backbone model:
  ```bash
  gdown https://drive.google.com/uc?id=1wtaCRWjdqhCjTGmi5sPRso3OvDLmOIKV
  unzip -n backbone.zip
  ```
+ datasets:
  ```bash
  mkdir datasets
  cd datasets
  gdown https://drive.google.com/uc?id=1NjpcIp9ZHwbhhgQR_n2oCQG92q4oFBgs
  unzip -n supermarket_products_v3.zip
  ```
  :point_right: For all versions of datasets check [here](https://drive.google.com/drive/folders/1PG5CmgxLy9Y33JAIGPxPq84VNS99YJ62?usp=sharing)

### Step 3. Remove supercategory `supermarket-product` from JSON file

```bash
# latest version of dataset (v3)
python3 scripts/remove_supercategory.py

# (optional) specific version of dataset
python3 scripts/remove_supercategory.py --version v1
```

### Step 4. Train model

```bash
# latest version of dataset (v3)
python3 scripts/train_and_eval_model.py

# (optional) specific version of dataset
python3 scripts/train_and_eval_model.py --version v1
```

### Step 5. Demonstrate segmentation resuls

```bash
# latest version of dataset (v3)
python3 examples/display_result.py

# (optional) specific version of dataset
python3 examples/display_result.py --version v1
```

### Step 6. Infer model in your project

```bash
# under root directory of project
cp ./output/v3/config.yaml {YOUR_PROJECT_PATH}/config/
cp ./output/v3/model_final.pth {YOUR_PROJECT_PATH}/model/
```
