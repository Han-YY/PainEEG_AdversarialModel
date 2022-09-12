import torch
import torch.nn as nn
import torch.nn.function as F

# The main classifier for classifying the pain-related conditions
class main_clf(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.conv1 = nn.Conv2D(1, 128, 7, padding_mode='same')
        self.pool = nn.MaxPool2d(3)
        self.norm = nn.BatchNorm2d()
        self.conv2 = nn.Conv2D(128, 64, 5, padding_mode='same')
        self.conv3 = nn.Conv2D(64, 32, 3, padding_mode='same')
        self.dropout = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(100)
        self.fc2 = nn.Linear(class_count)
        self.softmax = nn.Softmax()


    
    def forward(self, x):
        x = self.norm(self.pool(F.relu(self.conv1(x))))
        x = self.norm(self.pool(F.relu(self.conv2(x))))
        x = self.norm(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(x)
        # Flatten
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = torch.flatten(x, 1)
        x = F.sigmoid(x)
        x = self.softmax(self.fc2(x))
        return x

# The adversary classifier for classifying the participant ids
class adv_clf(nn.Module):
    def __init__(self, sub_count):
        super().__init__()
        self.conv = nn.Conv2D(1, 16, 3, padding_mode='same')
        self.norm = nn.BatchNorm2d()
        self.fc = nn.Linear(sub_count)
        
    def forward(self, x):
        x = self.norm(self.conv(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x