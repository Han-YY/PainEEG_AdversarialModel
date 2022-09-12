import numpy as np
import torch
import torch.nn as nn
import torch.nn.function as F
from torch.utils.data import Dataset, DataLoader
import trans_net
from sklearn.model_selection import train_test_split


import pandas as pd
import os
import data_prep_func as prep
import random

### The basic parameters for training and testing
batch_size = 128 # Batch size during training
image_size = 32 # The size of width and height of the connectivity matrix
nc = 1 # Number of channels
nf = 100 # Size of z latent vector (size of adversary input)
num_epochs = 5 # Number of training epochs
lr = 0.0001 # Learning rate for Optimizers
beta1 = 0.5 # Beta1 hyperparam for Adam optimizer
class AdversarialModel:
    def __init__(self, data_samples, class_label, subject_label, lam=0.01):
         ### Parameters ###
        # data_samples: the feature trials
        # class_label: the labels of conditions
        # subject_label: the labels of participants
        # main_clf: the classifier classifying conditions
        # adv_clf: the classifier classifying participants
        # loss_func: the loss function

        # Assign the values to variables
        self.data_samples = data_samples
        self.class_label = class_label
        self.subject_label = subject_label
        # Create the dataloader
        self.painDataset = trans_net.PainDataset(data_samples, class_label, subject_label)

        # Get the number of classes and subjects involved in the classeification
        class_unique = np.unique(class_label)
        class_count = class_unique.shape[0]
        subject_unique = np.unique(subject_label)
        subject_count = subject_unique.shape[0]

        # Assign the models
        self.main_clf = trans_net.main_clf(class_count)
        self.adv_clf = trans_net.adv_clf(subject_count)
        # Initialize the weights
        self.main_clf.apply(trans_net.weights_init)
        self.adv_clf.apply(trans_net.weights_init)

        # Loss functions
        self.criterion = nn.CrossEntropyLoss() 

        # Optimizers
        self.main_optim = torch.optim.Adam(self.main_clf.parameters(), lr=0.001, betas=(beta1, 0.999))
        self.adv_optim = torch.optim.Adam(self.adv_clf.parameters(), lr=0.001, betas=(beta1, 0.999))

    def train(self, log_path): 
        # Training Loop

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0   

        for i, data in enumerate(self.painDataset, 0):
            



