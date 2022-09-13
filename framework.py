import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
ngpu = 0 # Number of GPUs (CHANGE IT WHEN RUNNING ON THE HPC)

class AdversarialModel:
    def __init__(self, data_samples, class_label, subject_label):
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

        # Create the training and testing datasets
        # Shuffle the indices
        dataset_indices = list(range(self.data_samples.shape[0]))

        random.shuffle(dataset_indices)
        train_idx = dataset_indices[0:int(0.75 * self.data_samples.shape[0])]
        test_idx = dataset_indices[int(0.75 * self.data_samples.shape[0]):]


        # Create the dataloader
        self.painDataset_train = trans_net.PainDataset(data_samples[train_idx], class_label[train_idx], subject_label[train_idx])
        self.painDataset_test = trans_net.PainDataset(data_samples[test_idx], class_label[test_idx], subject_label[test_idx])



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

    def train(self): 
        # Training Loop

        # Lists to keep track of progress
        main_losses = []
        adv_losses = []
        iters = 0   

        # For each epoch
        for epoch in range(num_epochs):
            
            # For each batch in the dataloader
            dataloader = DataLoader(self.painDataset_train, batch_size=batch_size, shuffle=True)
            for i, data in enumerate(dataloader, 0):
                
                ################
                # Update the main classifier (minimize the loss)
                self.main_clf.zero_grad()
                # Format the batch
                data_sample = data['data_sample']
                label = data['class']
                # Forward the labels through the main classifier
                output = self.main_clf(data_sample)
                # Calculate the loss
                main_loss = self.criterion(output, label)
                # Calculate the gradients for the main_clf in backward pass
                
                main_x = output.mean().item()
                

            

                ########################
                # Train the adversary classifier (maximize the loss)
                self.adv_clf.zero_grad()
                label = data['subject'].long()
                output = self.adv_clf(data_sample)
                # Loss of the adversary model
                adv_loss = self.criterion(output, label)

                # Optimize the losses
                mix_loss = main_loss - adv_loss # For maximizing the main loss and minimize the adversary loss
                mix_loss.backward()
                self.adv_optim.step()
                self.main_optim.step()
                adv_x = output.mean().item()

                 # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_main: %.4f\tLoss_adv: %.4f\tLoss(x): %.4f\tD(G(x)): %.4f / %.4f'
                        % (epoch, num_epochs, i, len(self.painDataset_train),
                            main_loss.item(), adv_loss.item(), mix_loss.item(), main_x, adv_x))

                # Save Losses for plotting later
                main_losses.append(main_loss.item())
                adv_losses.append(adv_loss.item())

                iters += 1

        return main_losses, adv_losses




            
            



