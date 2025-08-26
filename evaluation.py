# IMPORT LIBRARIES
from collections import Counter
import os  # for working with file paths
import torch  # core PyTorch library
import torch.nn as nn  # for neural network layers and loss functions
from torch.utils.data import DataLoader, random_split  # for batching and splitting datasets
from torchvision import datasets, transforms, models  # for datasets, image transformations, and pretrained models
import matplotlib.pyplot as plt  # for plotting images and results
import numpy as np  # for numerical operations
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # to evaluate model performance
from sklearn.metrics import classification_report

# CONFIGURATION
base_dir = os.getcwd()  # current working directory
data_dir = os.path.join(base_dir,"EuroSAT_RGB")  # folder containing EuroSAT dataset
batch_size = 32  # number of images per batch
num_classes = 10  # number of classes in EuroSAT dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
# use GPU if available; otherwise CPU

# DATASET AND DATALOADER
# Define the transformations applied to each image:
# - Resize to 224x224 pixels (ResNet input size)
# - Convert to PyTorch tensor
# - Normalize using ImageNet mean and std (ResNet pretrained model expectation)
transform = transforms.Compose([
    transforms.Resize((224,224)),  
    transforms.ToTensor(),  
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# Load dataset from folder structure (subfolders = class names)
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split dataset into train/validation/test sizes
# We only need the test set here because the model has been trained already
train_size = int(0.7 * len(dataset))  # 70% for training (ignored in this script)
val_size   = int(0.15 * len(dataset))  # 15% for validation (ignored in this script)
test_size  = len(dataset) - train_size - val_size  # remaining 15% for testing
_, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoader for test dataset
# shuffle=False because we do not need to randomize test order
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# MODEL SETUP
# Initialize ResNet18 architecture
# pretrained=False because we are loading our own trained weights
model = models.resnet18(pretrained=False)

# Replace the final fully connected layer to match the EuroSAT class count
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the trained model weights
model.load_state_dict(torch.load("resnet18_eurosat.pth"))

# Move model to GPU if available
model = model.to(device)

# Switch model to evaluation mode
# This disables dropout and batch norm updates
model.eval()

# TEST EVALUATION AND CONFUSION MATRIX
# Evaluate the model on the test set to get predictions
all_preds, all_labels = [], []  # lists to store predictions and true labels
with torch.no_grad():  # no gradients needed for evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # move to GPU if available
        outputs = model(images)  # forward pass
        all_preds.extend(outputs.argmax(1).cpu().numpy())  # predicted class labels
        all_labels.extend(labels.cpu().numpy())  # true class labels

# Compute and display confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=dataset.classes)
disp.plot(xticks_rotation=45)  # rotate class labels for readability
plt.show()

# SAMPLE PREDICTIONS VISUALIZATION
def imshow(img, title):
    """
    Helper function to display a single image.
    Input: img tensor (C,H,W), title string
    """
    img = img.numpy().transpose((1, 2, 0))  # convert tensor from CxHxW to HxWxC
    img = np.clip(img, 0, 1)  # clip values between 0 and 1 for display
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

# Take a single batch of images from the test set
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Get model predictions for the batch
outputs = model(images.to(device))
preds = outputs.argmax(1).cpu()  # convert to CPU and get predicted class indices

# Define a helper function to reverse ImageNet normalization
def unnormalize(img):
    img = img.clone()  # avoid modifying the original tensor
    img = img * torch.tensor([0.229,0.224,0.225]).view(3,1,1)  # multiply by std
    img = img + torch.tensor([0.485,0.456,0.406]).view(3,1,1)  # add mean
    img = torch.clamp(img, 0, 1)  # clip to [0,1]
    return img

# Display first 8 images with true and predicted labels
plt.figure(figsize=(12,6))
for i in range(8):
    plt.subplot(2,4,i+1)
    # unnormalize before displaying
    imshow(unnormalize(images[i]), f"True: {dataset.classes[labels[i]]}\nPred: {dataset.classes[preds[i]]}")
plt.show()

# Count how many samples per class
class_counts = Counter(all_labels)

# Compute per-class accuracy
per_class_acc = {}
for i, class_name in enumerate(dataset.classes):
    idxs = [j for j, label in enumerate(all_labels) if label == i]
    correct = sum([all_preds[j] == all_labels[j] for j in idxs])
    per_class_acc[class_name] = correct / len(idxs)

print("Per-class accuracy:")
for class_name, acc in per_class_acc.items():
    print(f"{class_name}: {acc:.3f}")

# Compute overall metrics, precision, recall F1-score
print("Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))