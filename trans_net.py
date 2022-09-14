import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# The main classifier for classifying the pain-related conditions
class main_clf(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 7, padding=3, stride=1)
        self.pool = nn.MaxPool2d(3)
        self.norm1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 5, padding=2, stride=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1, stride=1)
        self.norm3 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(in_features=32, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=class_count)
        self.softmax = nn.Softmax()


    
    def forward(self, x):
        x = self.norm1(self.pool(F.relu(self.conv1(x))))
        x = self.norm2(self.pool(F.relu(self.conv2(x))))
        x = self.norm3(self.pool(F.relu(self.conv3(x))))
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
        self.conv = nn.Conv2d(1, 16, 3, padding=1, stride=1)
        self.norm = nn.BatchNorm2d(16)
        self.fc = nn.Linear(in_features=16384, out_features=sub_count)
        
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

        # Convert the class labels into array
        # class_unique = np.unique(class_labels)
        # class_unique = class_unique.tolist()
        # class_count = len(class_unique)
        # class_labels_n = np.zeros((class_labels.shape[0], class_count))
        # for i in range(class_labels.shape[0]):
        #     class_labels_n[i, class_unique.index(class_labels[i])] = 1
        # self.class_labels = class_labels_n
        
        # Convert the subject labels into array
        subject_unique = np.unique(subject_labels)
        subject_unique = subject_unique.tolist()
        subject_count = len(subject_unique)
        # subject_labels_n = np.zeros((subject_labels.shape[0], subject_count))
        # for i in range(subject_labels.shape[0]):
        #     subject_labels_n[i, subject_unique.index(subject_labels[i])] = 1
        subject_labels_n = []
        for subject_label in subject_labels:
            subject_labels_n.append(subject_unique.index(subject_label))
        self.subject_labels = subject_labels_n
    
    def __len__(self):
        return self.data_samples.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        orig_shape = self.data_samples[idx].shape
        data_sample = torch.from_numpy(np.reshape(self.data_samples[idx], newshape=(orig_shape[2], orig_shape[0], orig_shape[1]))).float()
        # class_label = torch.from_numpy(self.class_labels[idx])
        # subject_label = torch.from_numpy(self.subject_labels[idx])
        class_label = self.class_labels[idx]
        subject_label = self.subject_labels[idx]

        sample = {'data_sample': data_sample, 'class': class_label, 'subject': subject_label}

        return sample

# The function for initializing the weights of two networks
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
