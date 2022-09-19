import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import trans_net
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold


import pandas as pd
import os
import data_prep_func as prep
import random

### The basic parameters for training and testing
batch_size = 512 # Batch size during training
image_size = 32 # The size of width and height of the connectivity matrix
nc = 1 # Number of channels
nf = 100 # Size of z latent vector (size of adversary input)
num_epochs = 200 # Number of training epochs
lr = 0.0001 # Learning rate for Optimizers
beta1 = 0.9 # Beta1 hyperparam for Adam optimizer
k_fold = 10 # Number of folds in cross-validation
ngpu = 0 # Number of GPUs (CHANGE IT WHEN RUNNING ON THE HPC)
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class AdversarialModel:
    def __init__(self, data_samples, class_label, subject_label, exclude_idx, lam1, lam2):
         ### Parameters ###
        # data_samples: the feature trials
        # class_label: the labels of conditions
        # subject_label: the labels of participants
        # main_clf: the classifier classifying conditions
        # adv_clf: the classifier classifying participants
        # loss_func: the loss function
        # exclude_idx: the INDEX of the exclude subject used as the training set
        # lam1: relative convergence rate of the adversary network to the main network
        # lam2: relative convergence rate of the loss targetting at minimizing the loss of the adversary network to its maximization
        self.lam1 = lam1
        self.lam2 = lam2

        # Assign the values to variables
        self.data_samples = data_samples
        self.class_label = class_label
        self.subject_label = subject_label


        # Create the dataloader
        # self.painDataset_train = trans_net.PainDataset(data_samples[train_idx], class_label[train_idx], subject_label[train_idx])
        # self.painDataset_test = trans_net.PainDataset(data_samples[test_idx], class_label[test_idx], subject_label[test_idx])



        # Get the number of classes and subjects involved in the classeification
        class_unique = np.unique(class_label)
        class_count = class_unique.shape[0]
        subject_unique = np.unique(subject_label)
        subject_count = subject_unique.shape[0]

        # Create the training set and the testing set according to the subject id
        train_idx = [i for i, e in enumerate(subject_label) if e != subject_unique[exclude_idx]]
        test_idx = [i for i, e in enumerate(subject_label) if e == subject_unique[exclude_idx]]

        # Create the training and pre-training datasets
        # Shuffle the indices
        random.shuffle(train_idx)
        
        pre_idx = train_idx[int(0.75 * len(train_idx)):]
        
        train_idx_0 = train_idx[0:int(0.75 * len(train_idx))]
        

        # Create the dataloader
        self.painDataset_train = trans_net.PainDataset(data_samples[train_idx_0], class_label[train_idx_0], subject_label[train_idx_0])
        self.painDataset_pre = trans_net.PainDataset(data_samples[pre_idx], class_label[pre_idx], subject_label[pre_idx])
        self.painDataset_test = trans_net.PainDataset(data_samples[test_idx], class_label[test_idx], subject_label[test_idx])

        # Assign the models
        self.main_clf = trans_net.main_clf(class_count)
        self.adv_clf = trans_net.adv_clf(subject_count-1) 
        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            self.main_clf = nn.DataParallel(self.main_clf, list(range(ngpu)))
            self.adv_clf = nn.DataParallel(self.adv_clf, list(range(ngpu)))
        # Initialize the weights
        self.main_clf.apply(trans_net.weights_init)
        self.adv_clf.apply(trans_net.weights_init)

        # Initialize the encoder
        self.enc = trans_net.encoder()
        self.enc.apply(trans_net.weights_init)

        # Loss functions
        self.criterion = nn.CrossEntropyLoss() 

        # Optimizers
        self.enc_optim = torch.optim.Adam(self.enc.parameters(), lr=1e-4, betas=(beta1, 0.999), weight_decay=1e-5)
        self.main_optim = torch.optim.Adam(self.main_clf.parameters(), lr=1e-4, betas=(beta1, 0.999), weight_decay=1e-5)
        self.adv_optim = torch.optim.Adam(self.adv_clf.parameters(), 1e-4, betas=(beta1, 0.999), weight_decay=1e-5)
    

    def train(self): 
        torch.manual_seed(3407)# The magical seed
        # Training Loop

        # Lists to keep track of progress
        main_losses = []
        adv_losses = []
        main_accs = []
        adv_accs = []
        adv_accs_pre = []
        iters = 0   

        # Define the K-fold cross-validator
        kfold = KFold(n_splits=k_fold, shuffle=True)

        # Pre-train the adversary network to detect the features with high correlation to the individual differences
        for fold, (train_ids, test_ids) in enumerate(kfold.split(self.painDataset_pre)):
            # Sample elements randomly from a given list of ids, no replacement.
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            
            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(self.painDataset_pre, batch_size=10, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(self.painDataset_pre, batch_size=10, samsampler=test_subsampler)
            for epoch in range(num_epochs):
                # dataloader_pre = DataLoader(self.painDataset_pre, batch_size=batch_size, shuffle=True)
                adv_acc = 0

                for i, data in enumerate(trainloader, 0):
                    self.adv_optim.zero_grad()
                    self.enc_optim.zero_grad()
                    self.adv_clf.train()

                    # Format the batch
                    data_sample = data['data_sample'].to(device)
                    data_enc = self.enc(data_sample)
                    label = data['subject'].long().to(device)
                    output = self.adv_clf(data_enc)
                    # Loss of the adversary model
                    adv_loss = self.criterion(output, label)

                    # Optimize the losses
                    adv_loss.backward()
                    self.adv_optim.step()
                    self.enc_optim.step()

                    adv_acc += (torch.argmax(output, dim=1) == label).float().sum()

            # Evaluation for this fold
            correct, total = 0, 0
            with torch.no_grad():

                # Iterate over the test data and generate predictions
                for i, data in enumerate(testloader, 0):

                    # Get inputs
                    data_sample = data['data_sample'].to(device)
                    data_enc = self.enc(data_sample)
                    targets = data['subject'].long().to(device)

                    # Generate outputs
                    outputs = self.adv_clf(data_enc)

                    # Set total and correct
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                adv_acc_epoch = 100.0 * (correct / total)
                adv_accs_pre.append(adv_acc_epoch)
            print('Pre-trained finished, the accuracy is: ' + str(adv_acc_epoch) + ' and the loss is: ' + str(adv_loss))

        # Trainiing the main classifier with 10-fold validation
        for fold, (train_ids, test_ids) in enumerate(kfold.split(self.painDataset_train)):
            # Sample elements randomly from a given list of ids, no replacement.
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            
            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(self.painDataset_train, batch_size=10, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(self.painDataset_train, batch_size=10, samsampler=test_subsampler)
            for epoch in range(num_epochs):
                
                # For each batch in the dataloader
                # dataloader_train = DataLoader(self.painDataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
                main_acc, adv_acc = 0, 0
                
                for i, data in enumerate(trainloader, 0):
                    
                    ################
                    # Update the main classifier (minimize the loss)
                    self.main_optim.zero_grad()
                    self.enc_optim.zero_grad()
                    self.main_clf.train()
                    
                    # Format the batch
                    data_sample = data['data_sample'].to(device)
                    data_enc = self.enc(data_sample)

                    label = data['class'].long().to(device)
                    # Forward the labels through the main classifier
                    output = self.main_clf(data_enc)
                    # Calculate the loss
                    main_loss = self.criterion(output, label)
                    # Calculate the gradients for the main_clf in backward pass
                    
                    main_x = output.mean().item()
                    main_acc += (torch.argmax(output, dim=1) == label).float().sum()
                    

                

                    ########################
                    # Train the adversary classifier (maximize the loss)
                    self.adv_optim.zero_grad()
                    
                    label = data['subject'].long().to(device)
                    output = self.adv_clf(data_enc)
                    # Loss of the adversary model
                    adv_loss = self.criterion(output, label) * 0.1 * self.lam1

                    # Optimize the losses
                    mix_loss = main_loss + (-adv_loss) # For maximizing the main loss and minimize the adversary loss
                    adv_loss_control = adv_loss * self.lam2
                    adv_loss_control.backward(retain_graph=True)
                    mix_loss.backward()
                    self.adv_optim.step()
                    self.enc_optim.step()   
                    self.main_optim.step()

                    adv_x = output.mean().item()
                    adv_acc += (torch.argmax(output, dim=1) == label).float().sum()


                    # Output training stats
                    if i % 50 == 0:
                        print('[%d/%d][%d/%d]\tLoss_main: %.4f\tLoss_adv: %.4f\tLoss(x): %.4f\tD(G(x)): %.4f / %.4f'
                            % (epoch+1, num_epochs, i, len(self.painDataset_train),
                                main_loss.item(), adv_loss.item(), mix_loss.item(), main_x, adv_x))

                    # Save Losses for plotting later
                    main_losses.append(main_loss.item())
                    adv_losses.append(adv_loss.item())

                    iters += 1
                # Evaluation for this fold
                correct, total = 0, 0
                with torch.no_grad():

                    # Iterate over the test data and generate predictions
                    for i, data in enumerate(testloader, 0):

                        # Get inputs
                        data_sample = data['data_sample'].to(device)
                        data_enc = self.enc(data_sample)
                        subjects = data['subject'].long().to(device)
                        classes = data['class'].long().to(device)

                        # Generate outputs
                        outputs_main = self.main_clf(data_enc)
                        outputs_adv = self.adv_clf(data_enc)

                        # Set total and correct
                        _, predicted = torch.max(outputs_main.data, 1)
                        total_main += classes.size(0)
                        correct_main += (predicted == classes).sum().item()

                        _, predicted = torch.max(outputs_adv.data, 1)
                        total_adv += subjects.size(0)
                        correct_adv += (predicted == subjects).sum().item()

                    main_acc_epoch = 100.0 * (correct_main / total_main)
                    main_accs.append(adv_acc_epoch)
                    adv_acc_epoch = 100.0 * (correct_adv / total_adv)
                    adv_accs.append(adv_acc_epoch)

                main_accs.append(main_acc_epoch.float())
                adv_accs.append(adv_acc_epoch.float())
            
            # Test the main classifier with the testing dataset
            dataloader_test = DataLoader(self.painDataset_test, batch_size=batch_size, shuffle=True)
            data_test = next(iter(dataloader_test))
            X_test = self.enc(data_test['data_sample'])
            y_test = data_test['class']

        # Predict the classes of the testing input dataset
        with torch.no_grad():
            self.main_clf.eval() # Set the model as evaluation mode
            output_test = self.main_clf(X_test)
            preds = torch.argmax(output_test, dim=1)
            cf_matrix = confusion_matrix(y_test, preds)
            pred_acc = accuracy_score(y_test, preds)

        return main_losses, adv_losses, main_accs, adv_accs, cf_matrix, pred_acc




            
            



