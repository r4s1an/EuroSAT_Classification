# EuroSAT Land Use Classification with ResNet18

## Overview
This project demonstrates a deep learning approach for **land use and land cover (LULC) classification** using the **EuroSAT RGB dataset**.  
A **ResNet18** model is trained with PyTorch to classify Sentinel-2 satellite images into **10 distinct classes**, achieving **over 98% test accuracy**.  
This project showcases end-to-end ML workflow: data preparation, model training, evaluation, and visualization.

---

## Dataset
- **Name:** EuroSAT RGB  
- **Images:** 27,000 labeled images, RGB channels only  
- **Classes (10):** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake  
- **Source:** [EuroSAT GitHub](https://github.com/phelber/EuroSAT) 
---

## Project Structure

- training.py # script to train the model
- evaluation.py # script to evaluate the model
- resnet18_eurosat.pth # trained model weights
- requirements.txt # required packages
- README.md # project documentation
- Confusion matrix.png # shows the performance of the trained ResNet18 model on the test set
- First 8 images with true and predicted labels.png # sample Predictions on Test Set

---
## Model Training
This was done with training.py. The trained weights are saved as: resnet18_eurosat.pth. To run the train script:
```bash
python train.py
```
**Model:** ResNet18 (pretrained on ImageNet optionally)
**Optimizer:** Adam (lr=1e-4)
**Epochs:** 5
**Batch size:** 32
**Device:** GPU if available, else CPU

---

## Dependencies
- Python >= 3.10  
- PyTorch 2.7.1+cu118  
- torchvision 0.22.1+cu118  
- numpy 2.3.1  
- matplotlib 3.10.0  
- scikit-learn 1.6.1

---
## Evaluation
The model evaluation
---
