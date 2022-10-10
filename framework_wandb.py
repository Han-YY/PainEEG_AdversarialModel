import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import trans_net
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from torchmetrics import ConfusionMatrix, Accuracy
import wandb


import pandas as pd
import os
import data_prep_func as prep
import random

### The basic parameters for training and testing
learning_rate = 1e-4 # Learning rate for optimizers
beta1 = 0.5 # Beta1 heperparam for Adam optimizer
k_fold = 10 # Number of folds in cross-validation
ngpu = 4 # Number of GPUs
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# The adversarial model's basic setting
class AdversarialModel:
    def __init__(self, data_samples, class_label, subject_label, exclude_idx):
        ### Parameters ###
        # data_samples: the feature trials
        # class_label: the labels of conditions
        # subject_label: the labels of participants
        # main_clf: the classifier classifying conditions
        # adv_clf: the classifier classifying participants
        # loss_func: the loss function
        # exclude_idx: the INDEX of the exclude subject used as the training set

        # Assign the values to variables
        self.data_samples = data_samples
        self.class_label = class_label
        self.subject_label = subject_label

        # Get the number of classes and subjects involved in the classification
        class_unique = np.unique(class_label)
        class_count = class_unique.shape[0]
        subject_unique = np.unique(subject_label)
        subject_count = subject_unique.shape[0]
        self.class_count = class_count

        # Creating the training set and the testing set according to the subject id
        train_idx = [i for i, e in enumerate(subject_label) if e != subject_unique[exclude_idx]]
        test_idx = [i for i, e in enumerate(subject_label) if e == subject_unique[exclude_idx]]


        # Create the training set for pre-training the adversary model and the training both models
        random.shuffle(train_idx)
        pre_idx = train_idx[int(0.5 * len(train_idx)):]
        train_idx_0 = train_idx[0:int(0.5 * len(train_idx))]

        # Create the testing set for post-testing
        random.shuffle(test_idx)
        post_idx = train_idx[int(0.9 * len(test_idx)):]
        test_idx_0 = train_idx[0:int(0.9 * len(test_idx))]

         # Create global variables to balance the ratio of different conditions in each test
        test_class_id = []
        for class_id in range(class_count):
            test_id_temp =  [i for i, e in enumerate(class_label[test_idx_0]) if e == class_id]
            test_class_id.append(test_id_temp)
        self.test_class_id = test_class_id


        # Create the dataloader
        self.painDataset_train = trans_net.PainDataset(data_samples[train_idx_0], class_label[train_idx_0], subject_label[train_idx_0])
        self.painDataset_pre = trans_net.PainDataset(data_samples[pre_idx], class_label[pre_idx], subject_label[pre_idx])
        self.painDataset_test = trans_net.PainDataset(data_samples[test_idx_0], class_label[test_idx_0], subject_label[test_idx_0])
        self.painDataset_post = trans_net.PainDataset(data_samples[post_idx], class_label[post_idx], subject_label[post_idx])

        # Assign the models
        self.main_clf = trans_net.main_clf(class_count).to(device)
        self.adv_clf = trans_net.adv_clf(subject_count-1).to(device)
        self.enc = trans_net.encoder().to(device)

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            self.main_clf = nn.DataParallel(self.main_clf, list(range(ngpu)))
            self.adv_clf = nn.DataParallel(self.adv_clf, list(range(ngpu)))
            self.enc = nn.DataParallel(self.enc, list(range(ngpu)))
        # Initialize the weights
        self.main_clf.apply(trans_net.weights_init)
        self.adv_clf.apply(trans_net.weights_init)
        self.enc.apply(trans_net.weights_init)

        self.criterion = nn.CrossEntropyLoss() 

        # Optimizers
        self.enc_optim = torch.optim.Adam(self.enc.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=1e-5)
        self.main_optim = torch.optim.Adam(self.main_clf.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=1e-5)
        self.adv_optim = torch.optim.Adam(self.adv_clf.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=1e-5)

    def train(self, config=None):
        torch.manual_seed(3407) # The magical seed

        # Define the K-fold cross-validator
        kfold = KFold(n_splits=k_fold, shuffle=True)

        # Get the config information
        wandb.init(config=config, project = "adv_test")
        config = wandb.config # Get the configuration from the wandb setting
        num_epochs = config.epochs
        batch_size = config.batch_size
        lam1 = config.lam1
        lam2 = config.lam2

        # Pre-train the adversary network
        for fold, (train_ids, test_ids) in enumerate(kfold.split(self.painDataset_pre)):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
                
            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(self.painDataset_pre, batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(self.painDataset_pre, batch_size=batch_size, sampler=test_subsampler)

            for epoch in range(num_epochs):
                for i, data in enumerate(trainloader, 0):
                    self.adv_optim.zero_grad()
                    self.enc_optim.zero_grad()

                    # Format the batch
                    data_sample = data['data_sample'].to(device)
                    data_enc = self.enc(data_sample)
                    label = data['subject'].long().to(device)
                    output = self.adv_clf(data_enc).to(device)

                    # Loss of the adversary model
                    adv_loss = self.criterion(output, label)

                    # Optimize the loss
                    adv_loss.backward()
                    self.adv_optim.step()
                    self.enc_optim.step()

            # Evaluate for this fold
            correct, total = 0.0, 0.0
            with torch.no_grad():
                for i, data in enumerate(testloader, 0):

                    # Get inputs
                    data_sample = data['data_sample'].to(device)
                    data_enc = self.enc(data_sample)
                    targets = data['subject'].long().to(device)
                    outputs = self.adv_clf(data_enc).to(device)

                    # Set total and correct
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                
                adv_acc_epoch = 100.0 * (correct / total)
                print('Pre-trained finished, the accuracy is: ' + str(adv_acc_epoch) + ' and the loss is: ' + str(adv_loss))
        
        
        # Train the main classifier and the adversary network
        for fold, (train_ids, test_ids) in enumerate(kfold.split(self.painDataset_train)):
            # Sample elements randomly from a given list of ids
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(self.painDataset_train, batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(self.painDataset_train, batch_size=batch_size, sampler=test_subsampler)

            for epoch in range(num_epochs):
                main_correct, adv_correct = 0, 0
                total_main, total_adv = 0, 0

                for i, data in enumerate(trainloader, 0):
                    # Update the main classifier
                    # Remove all the gradiants in the previous process
                    self.main_optim.zero_grad()
                    self.adv_optim.zero_grad() 
                    self.enc_optim.zero_grad()

                    # Start the training process
                    self.main_clf.train()
                    self.adv_clf.train()

                    # Get the data
                    data_sample = data['data_sample'].to(device)
                    data_enc = self.enc(data_sample)
                    label = data['class'].long().to(device)
                    
                    ## Train the main classifier
                    #Forward the labels through the main classifier
                    output = self.main_clf(data_enc).to(device)
                    # Calculate the loss 
                    main_loss = self.criterion(output, label)
                    main_correct += (torch.argmax(output, 1) == label).float().sum()
                    total_main += label.size(0)

                    ## Train the adversary classifier
                    label = data['subject'].long().to(device)
                    output = self.adv_clf(data_enc).to(device)
                    adv_loss = self.criterion(output, label)
                    adv_correct += (torch.argmax(output, 1) == label).float().sum().item()
                    total_adv += label.size(0)

                    # Optimize the losses in an adversarial progress
                    mix_loss = main_loss - lam1 * adv_loss
                    adv_loss_control = adv_loss * lam2

                    # Adversarial Progress by controling the convergence targets
                    adv_loss_control.backward(retain_graph=True)
                    mix_loss.backward()
                    self.adv_optim.step()
                    self.enc_optim.step()
                    self.main_optim.step()



                    # Submit the current losses and accs
                    log_dict = {"main_loss": main_loss.item(),
                    "adv_loss": adv_loss.item(),
                    "mix_loss": mix_loss.item()}
                    wandb.log(log_dict)
                
                main_acc  = 100.0 * (main_correct / total_main)
                adv_acc = 100.0 * (adv_correct / total_adv)

                log_dict = {"main_acc": main_acc,
                "adv_acc": adv_acc}
                wandb.log(log_dict)
                
            # Test the model with data from the testing set with random sampling
            # Load the data from the testing set
            # AWARE: It is only for finding the hyperparameters, it will be risky in the actural property test!!!
            for cl_idx in range(len(self.test_class_id)):
                if cl_idx == 0:
                    test_final_test_id = random.sample(self.test_class_id[cl_idx], 200)
                else:
                    test_final_test_id = test_final_test_id + random.sample(self.test_class_id[cl_idx], 200)

            epoch_subsampler = torch.utils.data.SubsetRandomSampler(test_final_test_id)
            epochloader = torch.utils.data.DataLoader(self.painDataset_test, batch_size=batch_size, sampler=epoch_subsampler)

            # Use the temporary small testing set to test current performance
            data_test = next(iter(epochloader))
            data_sample = data_test['data_sample'].to(device)
            X_test = self.enc(data_sample)
            y_test = data_test['class'].long().to(device)

            # Predict the classes of the testing input dataset
            with torch.no_grad():
                self.main_clf.eval() # Set the main classifier in the evaluation mode
                output_test = self.main_clf(X_test)
                preds = torch.argmax(output_test, 1).to(device)
                accuracy = Accuracy().to(device)
                pred_acc = accuracy(preds, y_test)
                # cf_matrix = wandb.plot.confusion_matrix(probs=None, y_true=y_test.cpu(), preds=preds.cpu())
                log_dict = {"pred_acc": pred_acc}
                wandb.log(log_dict)
                

               

            # Validate this fold
            correct_main, total_main = 0, 0
            correct_adv, total_adv = 0, 0

            with torch.no_grad():
                for i, data in enumerate(testloader, 0):
                    data_sample = data['data_sample'].to(device)
                    data_enc = self.enc(data_sample)
                    subjects = data['subject'].long().to(device)
                    classes = data['class'].long().to(device)

                    # Generate outputs of two nets
                    outputs_main = self.main_clf(data_enc).to(device)
                    outputs_adv = self.adv_clf(data_enc).to(device)

                    # Get total and correct
                    _, predicted = torch.max(outputs_main.data, 1)
                    total_main += classes.size(0)
                    correct_main += (predicted == classes).sum().item()

                    _, predicted = torch.max(outputs_adv.data, 1)
                    total_adv += subjects.size(0)
                    correct_adv += (predicted == subjects).sum().item()

                main_acc_epoch = 100.0 * (correct_main / total_main)
                adv_acc_epoch = 100.0 * (correct_adv / total_adv)

                log_dict = {"main_acc_epoch": main_acc_epoch,
                "adv_acc_epoch": adv_acc_epoch}
            
        # Post test after the whole training
        # Use the temporary small testing set to test current performance
        postloader = torch.utils.data.DataLoader(self.painDataset_post, batch_size=batch_size, shuffle=True)
        data_test = next(iter(postloader))
        data_sample = data_test['data_sample'].to(device)
        X_test = self.enc(data_sample)
        y_test = data_test['class'].long().to(device)

            # Predict the classes of the testing input dataset
        with torch.no_grad():
            self.main_clf.eval() # Set the main classifier in the evaluation mode
            output_test = self.main_clf(X_test)
            preds = torch.argmax(output_test, 1).to(device)
            accuracy = Accuracy().to(device)
            pred_acc = accuracy(preds, y_test)
            # cf_matrix = wandb.plot.confusion_matrix(probs=None, y_true=y_test.cpu(), preds=preds.cpu())
            log_dict = {"post_pred_acc": pred_acc}
            wandb.log(log_dict)


