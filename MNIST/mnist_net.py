import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2)

        # flatten

        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # print("input x.shape: ", x.shape)
        x = self.pool1(F.relu(self.conv1(x)))
        # print("after pool1: ", x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        # print("after pool2: ", x.shape)
        x = x.reshape(-1, 7*7*64)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def predict(self, image):
        if torch.max(image) > 1.0 or torch.min(image) < 0.0:
            image = torch.clamp(image, 0.0, 1.0)
        self.eval()
        with torch.no_grad():
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            image_batch = image.clone()
            if len(image.size()) < 4:
                image_batch = image_batch.unsqueeze(0)
            image_batch = image_batch.to(device)
            output = self(image_batch)
            _, predict = torch.max(output.data, 1)

        if len(image.size()) < 4:
            return predict[0].item()
        return predict

    def getProbs(self, image):
        if torch.max(image) > 1.0 or torch.min(image) < 0.0:
            image = torch.clamp(image, 0.0, 1.0)
        self.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image_batch = image.clone()
        if len(image.size()) < 4:
            image_batch = image_batch.unsqueeze(0)
        image_batch = image_batch.to(device)
        print("image_batch.shape: ", image_batch.shape)
        output = self.forward(image_batch)
        # _, predict = torch.max(output.data, 1)
        return output.data