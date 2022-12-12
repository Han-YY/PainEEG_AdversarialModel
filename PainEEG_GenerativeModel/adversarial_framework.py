# Author: Yiyuan Han
# Date: 08/12/2022
# Make all the steps into function, so that the pipeline can be built as blocks
## Import the packages
import numpy as np
import torch 
import torch.nn as nn
from torch.utils import Dataset, DataLoader
import trans_net

from torch.optim import Adam

################ Functions for training and testing #################
# Train a model (The main classifier or the adversary classifier)
def train(clf, train_set, epoch_num, batch_size, device, output=False):
    # Initilize the classifier
    enc = trans_net.encoder()
    enc.apply(trans_net.weights_init)
    clf.apply(trans_net.weights_init)

    # Initialize the optimizers
    enc_opt = Adam(enc.parameters(), lr=1e-3, betas = (0.9, 0.99), weight_decay=1e-8)
    main_opt = Adam(clf.parameters(), lr=1e-3, betas = (0.9, 0.99), weight_decay=1e-8)    

    criterion = nn.CrossEntropyLoss()

    # Load the dataset
    trainloader = DataLoader(train_set, batch_size=batch_size)
    losses = []

    # Train the model
    for epoch in range(epoch_num):
        for i, data in enumerate(trainloader, 0):
            enc_opt.zero_grad()
            main_opt.zero_grad()

            enc.train()
            clf.train()

            # format the batch
            data_sample = data['data_sample'].to(device)
            data_enc = enc(data_sample)
            label = data['class'].long().to(device)

            # Generate outputs
            output = clf(data_enc)
            loss = criterion(output, label)
            losses.append(loss.item())

            loss.backward()
            main_opt.step()
            enc_opt.step()

    if output:
        return clf, enc, losses
    else:
        return clf, enc
    

# Train the main and adversary models together
def train_combined(train_set, epoch_num, batch_size, device, lam1, lam2, output=False):


# Test the model with single samples
def test_sample():

# Test tthe model with accumulated evidence
def test_acc_evi():

################ Functions for splitting the dataset #################
# Exclude one participant (subject) as the testing set and keep the other participants as training set
def sub_exclude(sub_id):

# Combine the data
def sub_combine(sub_id):

# Split the training set into a training set and a validation set
def train_test():
