import kagglehub
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

class FaceMaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) #3 color channels to 32 which is apparently standard
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  #64 channels * the size of the image being 32*32 from the pool in the forward (pooled twice one here and one out in forward)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU() #as causes problems when doing nn.ReLu on the x normally in forward

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # -> [batch, 32, 64, 64]
        x = self.pool(self.relu(self.conv2(x)))  # -> [batch, 64, 32, 32]
        x = x.view(-1, 64 * 32 * 32) #reshape to batchsize of 32 by channel*width*height
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    """def __init__(self): #1 layer is way too inconsistent so changed to 2
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) #3 color channels to 32 which is apparently standard
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  #number of channels channels * the size of the image 
        self.relu = nn.ReLU() #as causes problems when doing nn.ReLu on the x normally in forward

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) 
        x = x.view(-1, 32 * 64 * 64)
        x = self.relu(self.fc1(x))
        return x"""

model = FaceMaskCNN()
model.load_state_dict(torch.load("face_mask_cnn.pth", map_location=torch.device("cpu")))

diffDataSet = kagglehub.dataset_download("omkargurav/face-mask-dataset") 

diffDataSetPath = os.path.join(diffDataSet, "data")
#print(os.listdir(diffDataSetPath)) is "with_mask", "without_mask"

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGBA') if img.mode == 'P' else img),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = ImageFolder(diffDataSetPath, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

from sklearn.metrics import classification_report, accuracy_score

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

print("Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=dataset.classes))

import matplotlib.pyplot as plt
import numpy as np
class_names = dataset.classes
predicted_labels = [class_names[i] for i in preds]
imageLoader = DataLoader(dataset, batch_size=9, shuffle=False)
images, labels = next(iter(dataloader))
images = images.numpy()
fig, axes = plt.subplots(3, 3, figsize=(8,8))
class_names = dataset.classes  
for i, ax in enumerate(axes.flatten()):
    img = np.transpose(images[i], (1, 2, 0))
    ax.imshow(img)
    ax.set_title(f"True: {class_names[labels[i]]}\nPred: {predicted_labels}")
    ax.axis('off')
plt.tight_layout()
plt.show()