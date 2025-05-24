import kagglehub
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Download latest version #(provided by the kaggle thing)
path = kagglehub.dataset_download("ashishjangra27/face-mask-12k-images-dataset")

#data_path = os.path.join(path, "Face Mask Dataset") #finding what the heck is in path
#print(os.listdir(data_path)) #results were Test, Train, Validation

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

test_dir = os.path.join(path, "Face Mask Dataset", "Test")
train_dir = os.path.join(path, "Face Mask Dataset", "Train")
val_dir = os.path.join(path, "Face Mask Dataset", "Validation")

test_dataset  = datasets.ImageFolder(root=test_dir, transform=transform)
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset   = datasets.ImageFolder(root=val_dir, transform=transform)

test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = len(train_dataset.classes) #could just hard code but prob better this way

class FaceMaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) #3 color channels to 32 which is apparently standard
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  #64 channels * the size of the image being 32*32 from the pool in the forward (pooled twice one here and one out in forward)
        self.fc2 = nn.Linear(128, num_classes)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FaceMaskCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5 #as is standard apparently 
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        uneededMaxValues, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
torch.save(model.state_dict(), "face_mask_cnn.pth")
