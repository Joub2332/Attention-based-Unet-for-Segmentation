# Attention-based-Unet-for-Segmentation
## Introduction
This project proposes a segmentation model based on a **U-Net architecture** improved by an **attention mechanism**. This architecture aims to improve the performance of image segmentation by allowing the model to focus on relevant regions while processing contextual details. In this project, we compare the performances between a classic unet and an attention based unet. The Unets of this project were trained to determine the liver, spleen and right/left kidneys on a database of 2D images of abdominal scanners.
## Prerequisites
Ensure you have the following dependencies installed:

- Python 3.8.10
- PyTorch
- NumPy
- scikit-learn
- Matplotlib
- Nibabel
- Skimage
  
To install the dependencies, run:

```
pip install -r requirements.txt
```


## U-Net Architecture Overview
U-Net is a convolutional neural network designed for **image segmentation**, especially in medical imaging. Its architecture is based on a **symmetric encoder-decoder structure**, with **skip connections** that link the encoder and decoder, helping the model retain both high-level features and fine-grained spatial information.

![U-Net Architecture](Pictures/unet.png)

### Key Features:
- **Encoder-Decoder Structure** : The encoder reduces spatial dimensions, while the decoder upsamples to the original image size.
- **Skip Connections** : These connections between encoder and decoder layers help preserve detailed spatial information.
- **Fully Convolutional** : U-Net doesn’t use fully connected layers, making it efficient for images of different sizes.
- **Output** : A segmentation mask is produced where each pixel is assigned a class.
### Why U-Net Works:
- **Efficient with Small Datasets**: Thanks to skip connections, U-Net can perform well even with limited data.
- **Highly Accurate** : Combines both global context and local details for precise segmentation.
- **Flexible** : Originally designed for medical imaging, but works well for many other segmentation tasks.
### Applications:
- **Medical Image Segmentation** : Segmenting organs, tumors, etc.
- **Satellite Image Analysis** : Segmenting land features, water bodies, etc.
- **Autonomous Driving** : Segmenting roads, pedestrians, vehicles, etc.
  
## Attention-based U-Net Overview
The **Attention-based U-Net** enhances the traditional U-Net architecture by incorporating an **attention mechanism**, which allows the model to focus on relevant image regions while suppressing irrelevant ones. This leads to improved segmentation performance, particularly in complex images where precise localization is crucial.

![U-Net Architecture](Pictures/Unet_augmented.png)

### Key Features:
- **Attention Mechanism** : Dynamically highlights important features, improving the model's ability to focus on key regions for more accurate segmentation.
- **Improved Performance** : The attention mechanism helps the model perform better in scenarios with small datasets or complex structures, reducing the impact of irrelevant background noise.

### Why Attention-based U-Net Works:
- **Better Focus on Key Features**: The attention mechanism allows the model to better capture important structures and details in the image, leading to more accurate segmentations.
- **Flexible and Efficient** : Maintains the efficiency of U-Net while providing enhanced results in various domains, such as medical and satellite image segmentation.

## Usage
### Preparation 

Expliquer coment mettre les fichiers au bon endroit

### Training 
This model was trained on an abdominal image database which you can find [here](https://chaos.grand-challenge.org/Data/). However, this model can be trained on another medical or other imaging dataset.

To train the model with a custom dataset, use the following script:

a modifier
```
python scripts/train.py --dataset_path ./data/train --epochs 50 --batch_size 16
```
### Evaluation

### Visualisation

## Results
The performance of the Attention-based U-Net was compared to the standard U-Net version, showing a significant improvement in precision and recall metrics:
here is a table comparing the dice score of the two structures

| Structure         | Class 0  | Class 1  | Class 2  | Class 3  | Class 4  | Overall Dice Score |
| ----------------- | -------- | -------- | -------- | -------- | -------- | ------------------ |
| UNet              | 0.9920   | 0.6117   | 0.4747   | 0.4764   | 0.4907   | 0.6509             |
| Augmented UNet    |  0.9908 | 0.8290 | 0.6133 | 0.6318 | 0.7015 | 0.7533 |


## Acknowledgements
This project was developed as part of the TAF Deep Learning course led and supervised by Pierre-Henri Conze at IMT Atlantique.

## Authors
Skander MAHJOUB, email : skander.mahjoub@imt-atlantique.net

Maria Florenza LAMBERTI,email : maria.florenza-lamberti@imt-atlantique.net
