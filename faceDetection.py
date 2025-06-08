import kagglehub
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

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
        
modelTraining(5) #5 epochs is standard apparently
#model.load_state_dict(torch.load("face_mask_cnn.pth", map_location=torch.device("cpu"))) #this is if I want to skip training and just evaluate

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

import cv2 as cv
def modelOnlineEvaluation(cameraNum=0):
    cap = cv.VideoCapture(cameraNum) #assume is capturing from some camera that exists
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        #Actual frame ops here
        model.eval() #otherwise we start changing the model
        
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  #Convert BGR to RGB
        frame = cv.resize(frame, (128, 128))  #Resize to match model input size
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)  #Convert to tensor and change shape to [channel, height, width]
        frame = frame.unsqueeze(0)  #Add batch dimension
        outputs = model(frame)
        outputs = outputs.to(device)  #Ensure outputs are on the same device as the model
        uneededMaxValues, predicted = torch.max(outputs.data, 1)
        predicted = (lambda x: "Without Mask" if x == 0 else "With Mask")(predicted.item()) #convert to string
        print(f"Predicted class is {predicted}")

        cv.imshow("Live Video", cv.cvtColor(frame.squeeze(0).permute(1, 2, 0).byte().numpy(), cv.COLOR_RGB2BGR)) #so that the waitkey actually works as otherwise pressing q does nothing 
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    #When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

def modelEval(offline=True, save=True, cameraNum=0):
    if offline:
        modelOfflineEvaluation(save)
    else:
        modelOnlineEvaluation(cameraNum)
modelEval(offline=False, save=True, cameraNum=0) #change to offline=True to do test dataset
