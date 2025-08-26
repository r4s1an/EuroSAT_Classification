# EuroSAT RGB Image Classification with ResNet18
This project demonstrates a deep learning approach for **land use and land cover (LULC) classification** using the **EuroSAT RGB dataset**.  
A **ResNet18** model is trained with PyTorch to classify Sentinel-2 satellite images into **10 distinct classes**, achieving **over 98% test accuracy**.  
This project showcases end-to-end ML workflow: data preparation, model training, evaluation, and visualization.

## Dataset
- **Name:** EuroSAT RGB  
- **Images:** 4050 labeled images, RGB channels only  
- **Classes (10):** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake  
- **Source:** [EuroSAT GitHub](https://github.com/phelber/EuroSAT) 

## Project Structure

- training.py # script to train the model
- evaluation.py # script to evaluate the model
- resnet18_eurosat.pth # trained model weights
- requirements.txt # required packages
- README.md # project documentation
- Confusion matrix.png # shows the performance of the trained ResNet18 model on the test set
- First 8 images with true and predicted labels.png # sample Predictions on Test Set

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

## Evaluation
The model evaluation was done with evaluation.py. The scripts provides overall test accuracy, per-class accuracy, confusion matrix, classification report (precision, recall, F1-score), sample predictions with true vs predicted labels. To run the evaluation.py script:
```bash
python evaluation.py
```
The trained model had the following sample per-class accuracy:
```makefile
AnnualCrop: 0.993
Forest: 0.989
HerbaceousVegetation: 0.980
Highway: 0.997
Industrial: 0.997
Pasture: 0.990
PermanentCrop: 0.985
Residential: 0.998
River: 0.987
SeaLake: 0.995
```
The trained model had the following classification report:
```makefile
                      precision    recall  f1-score   support

          AnnualCrop       0.98      0.99      0.99       434      
              Forest       1.00      0.99      0.99       465      
HerbaceousVegetation       0.98      0.98      0.98       455      
             Highway       0.99      1.00      1.00       369      
          Industrial       1.00      1.00      1.00       366      
             Pasture       1.00      0.99      0.99       303      
       PermanentCrop       0.98      0.99      0.98       403      
         Residential       1.00      1.00      1.00       456      
               River       0.99      0.99      0.99       378      
             SeaLake       1.00      1.00      1.00       421      

            accuracy                           0.99      4050      
           macro avg       0.99      0.99      0.99      4050      
        weighted avg       0.99      0.99      0.99      4050
```

## Dependencies
- Python >= 3.10  
- PyTorch 2.7.1+cu118  
- torchvision 0.22.1+cu118  
- numpy 2.3.1  
- matplotlib 3.10.0  
- scikit-learn 1.6.1

## License notice
This project is licensed under the MIT License.  

