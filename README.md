# Attention-based-Unet-for-Segmentation
## Introduction
This project proposes a segmentation model based on a U-Net architecture improved by an attention mechanism. This architecture aims to improve the performance of image segmentation by allowing the model to focus on relevant regions while processing contextual details. In this project, we compare the performances between a classic unet and an attention based unet. The Unets of this project were trained to determine the liver, spleen and kidneys on a database of 2D images of abdominal scanners.
## Prerequisites
Ensure you have the following dependencies installed:

- Python 3.8.10
- PyTorch
- NumPy
- scikit-learn
- Matplotlib
- Nibabel
  
To install the dependencies, run:

```
pip install -r requirements.txt
```
## Usage
### Training 
This model was trained on an abdominal image database which is not provided in this repository. nevertheless, this model can be trained on another medical or other imaging dataset.

To train the model with a custom dataset, use the following script:
a modifier
```
python scripts/train.py --dataset_path ./data/train --epochs 50 --batch_size 16
```

