import kagglehub
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from PIL import Image
import numpy as np
import cv2 as cv
#source venv/bin/activate to activate venv

# Download latest version #(provided by the kaggle thing)
path = kagglehub.dataset_download("ashishjangra27/face-mask-12k-images-dataset")

#data_path = os.path.join(path, "Face Mask Dataset") #finding what the heck is in path
#print(os.listdir(data_path)) #results were Test, Train, Validation

test_dir = os.path.join(path, "Face Mask Dataset", "Test")
train_dir = os.path.join(path, "Face Mask Dataset", "Train")
val_dir = os.path.join(path, "Face Mask Dataset", "Validation")

def transformProcess(directory):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=directory, transform=transform)
    return dataset

test_dataset  = transformProcess(test_dir)
train_dataset = transformProcess(train_dir)
val_dataset   = transformProcess(val_dir)

def loader(dataset, batchSize, shuffleStatus):
    return DataLoader(dataset, batch_size=batchSize, shuffle=shuffleStatus)

test_loader  = loader(test_dataset, 32, False)
train_loader = loader(train_dataset, 32, True)
val_loader   = loader(val_dataset, 32, False)

device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = len(train_dataset.classes) #could just hard code but prob better this way

class FaceMaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) #3 color channels to 32 which is apparently standard
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  #num channels * width and height, ori is 128 but pool 3 times so divide by 8 for dimensions, doubling channels each time so
        self.fc2 = nn.Linear(256, 128) 
        self.fc3 = nn.Linear(128, num_classes)  
        self.relu = nn.ReLU() #as causes problems when doing nn.ReLu on the x normally in forward
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) #added this layer to try and improve accuracy for online eval 

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # -> [batch, 32, 64, 64]
        x = self.pool(self.relu(self.conv2(x)))  # -> [batch, 64, 32, 32]
        x = self.pool(self.relu(self.conv3(x))) #128 channels, 16 by 16 
        x = x.view(-1, 128 * 16 * 16) #reshape to batchsize to match fc1's expected channel*width*height
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FaceMaskCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def modelTraining(numEpochs):
    for epoch in range(numEpochs):
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
        print(f"Epoch {epoch+1}/{numEpochs}, Loss: {running_loss/len(train_loader):.4f}")
        
#modelTraining(5) #5 epochs is standard apparently
model.load_state_dict(torch.load("face_mask_cnn.pth", map_location=torch.device("cpu")))

def modelOfflineEvaluation(save=True):
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
    if save:
        torch.save(model.state_dict(), "face_mask_cnn.pth")

def squareResizePrep(frame):
    #0 is height, 1 is width
    diff = abs(frame.shape[0] - frame.shape[1]) #how much we need to pad 
    if frame.shape[0] > frame.shape[1]: #if height taller than width, so pad sides
        left = diff//2 #half on left
        right = diff - left #go diff forward to make "equal" length then go back left to center
        return cv.copyMakeBorder(frame,0,0,left,right,cv.BORDER_CONSTANT,value=[0,0,0]) #add black border to make square
    else: #if width wider than height, so pad top and bottom
        top = diff//2 #half on top
        bottom = diff - top #same logic as first case 
        return cv.copyMakeBorder(frame,top,bottom,0,0,cv.BORDER_CONSTANT,value=[0,0,0])

def modelOnlineEvaluation(cameraNum=0):
    cap = cv.VideoCapture(cameraNum)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    model.eval() #so we can use the model
    with torch.no_grad(): #not changing the weights
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            #frame prep
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = squareResizePrep(frame)
            frame = Image.fromarray(frame)  # convert to PIL for transforms
            frame = transform(frame).unsqueeze(0).to(device)

            #predict
            outputs = model(frame)
            unneededMaxValues, predicted = torch.max(outputs.data, 1)
            class_label = "Without Mask" if predicted.item() == 1 else "With Mask"
            print(f"Predicted class is {class_label}")

            #show what the camera sees 
            display_frame = (frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            display_frame = cv.cvtColor(display_frame, cv.COLOR_RGB2BGR)
            cv.imshow("Live Video", display_frame)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

def modelEval(offline=True, save=True, cameraNum=0):
    if offline:
        modelOfflineEvaluation(save)
    else:
        modelOnlineEvaluation(cameraNum)
modelEval(offline=False, save=True, cameraNum=0) #change to offline=True to do test dataset
