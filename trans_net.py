import torch
import torch.nn as nn
import torch.nn.function as F
from torch.utils.data import Dataset, DataLoader

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

# The class for building the dataset with both class labels and subject labels for tuning
class PainDataset(Dataset):
    def __init__(self, data_samples, class_labels, subject_labels):
        self.data_samples = data_samples
        self.class_labels = class_labels
        self.subject_labels = subject_labels
    
    def __len__(self):
        return self.data_sample.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data_sample = torch.from_numpy(self.data_samples[idx])
        class_label = self.class_labels[idx]
        subject_label = self.subject_labels[idx]

        sample = {'data_sample': data_sample, 'class': class_label, 'subject': subject_label}

        return sample

# The function for initializing the weights of two networks
def weights_init(m):
    nn.init.normal_(m.weight.data, 0.0, 0.02)
