# IMPORT LIBRARIES
import os  # for working with file paths
import torch  # core PyTorch library
import torch.nn as nn  # for defining loss functions and neural network layers
import torch.optim as optim  # for optimizers like Adam, SGD
from torch.utils.data import DataLoader, random_split  # for batching and splitting datasets
from torchvision import datasets, transforms, models  # for image datasets, transforms, and pretrained models
import matplotlib.pyplot as plt  # for plotting images and results
import numpy as np  # for array manipulation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # to evaluate predictions

# CONFIGURATION
base_dir = os.getcwd()  # get the current working directory
data_dir = os.path.join(base_dir,"EuroSAT_RGB")  # path to the dataset folder
batch_size = 32  # number of images per batch
num_classes = 10  # EuroSAT has 10 classes
num_epochs = 5  # number of times to iterate over the full dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
# if GPU is available, use it; otherwise, fallback to CPU

# DATASET AND DATALOADER
# Define transformations to apply to each image:
# - Resize to 224x224 (required by ResNet)
# - Convert to Tensor (PyTorch format)
# - Normalize using ImageNet mean/std (ResNet pretrained weights were trained with these)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# Load the dataset using ImageFolder
# It automatically maps subfolder names (classes) to labels
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split the dataset into train/validation/test
train_size = int(0.7 * len(dataset))  # 70% for training
val_size   = int(0.15 * len(dataset))  # 15% for validation
test_size  = len(dataset) - train_size - val_size  # 15% for testing
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders to efficiently load batches of data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # shuffle for training
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)   # no shuffle for validation
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # no shuffle for test

# MODEL SETUP
# Load pretrained ResNet18 model (weights trained on ImageNet)
model = models.resnet18(pretrained=True)  

# Replace the final fully connected layer to match EuroSAT classes
# Original ResNet18 has 1000 outputs; we need 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
model = model.to(device)

# Loss function: Cross-Entropy Loss (standard for multi-class classification)
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam (adaptive learning rate)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# TRAINING LOOP
for epoch in range(num_epochs):
    model.train()  # set model to training mode
    running_loss, correct = 0.0, 0  # initialize accumulators
    
    # Loop over batches of training data
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # move batch to GPU

        optimizer.zero_grad()  # clear previous gradients
        outputs = model(images)  # forward pass
        loss = criterion(outputs, labels)  # compute loss
        loss.backward()  # backpropagate
        optimizer.step()  # update weights

        # Accumulate loss and accuracy
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    train_loss = running_loss / len(train_dataset)  # average training loss
    train_acc = correct / len(train_dataset)  # training accuracy

    # VALIDATION
    model.eval()  # set model to evaluation mode
    val_correct = 0
    with torch.no_grad():  # no gradient computation during validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(val_dataset)  # validation accuracy

    # Print epoch summary
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

# Save the model
torch.save(model.state_dict(), "resnet18_eurosat.pth")
print("Model saved as resnet18_eurosat.pth")