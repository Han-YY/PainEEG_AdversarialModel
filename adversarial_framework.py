# Author: Yiyuan Han
# Date: 08/12/2022
# Make all the steps into function, so that the pipeline can be built as blocks
## Import the packages
import numpy as np
import random
import data_prep_func as prep
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import trans_net

from torch.optim import Adam

################ Functions for training and testing #################
################################################################ Functions for training #################
# Train a model (The main classifier or the adversary classifier)
# label_target: 'class' or 'subject'
def train(clf, train_set, label_target, epoch_num, batch_size, device, ngpu, output_state=False):
    # Initilize the classifier
    enc = trans_net.encoder()
    enc.apply(trans_net.weights_init)
    clf.apply(trans_net.weights_init)

    # Initialize the optimizers
    enc_opt = Adam(enc.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
    main_opt = Adam(clf.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)    

    if (device.type == 'cuda') and (ngpu > 1):
        enc = nn.DataParallel(enc, list(range(ngpu))).cuda()
        clf = nn.DataParallel(clf, list(range(ngpu))).cuda()

    criterion = nn.CrossEntropyLoss()

    # Load the dataset
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
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
            label = data[label_target].long().to(device)

            # Generate outputs
            output_label = clf(data_enc)
            loss = criterion(output_label, label)
            losses.append(loss.item())

            loss.backward()
            main_opt.step()
            enc_opt.step()

    if output_state:
        return enc, clf, losses
    else:
        return enc, clf
    

# Train the main and adversary models together
def train_combined(train_set, epoch_num, batch_size, device, ngpu, lam1, output_state=False):
    # Initilize the classifier
    enc = trans_net.encoder()
    main_clf = trans_net.main_clf()
    adv_clf = trans_net.adv_clf()

    enc.apply(trans_net.weights_init)
    main_clf.apply(trans_net.weights_init)
    adv_clf.apply(trans_net.weights_init)

    # Initialize the optimizers
    enc_opt = Adam(enc.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
    main_opt = Adam(main_clf.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)    
    adv_opt = Adam(adv_clf.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)    

    if (device.type == 'cuda') and (ngpu > 1):
        enc = nn.DataParallel(enc, list(range(ngpu))).cuda()
        main_clf = nn.DataParallel(main_clf, list(range(ngpu))).cuda()
        adv_clf = nn.DataParallel(adv_clf, list(range(ngpu))).cuda()

    criterion = nn.CrossEntropyLoss()

    # Load the dataset
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    losses = []

    # Train the model
    for epoch in range(epoch_num):
        for i, data in enumerate(trainloader, 0):
            enc_opt.zero_grad()
            main_opt.zero_grad()

            enc.train()
            main_clf.train()
            adv_clf.train()

            # format the batch
            data_sample = data['data_sample'].to(device)
            data_enc = enc(data_sample)
            label_class = data['class'].long().to(device)
            label_subject = data['subject'].long().to(device)

            # Generate outputs
            output_class = main_clf(data_enc)
            output_subject = adv_clf(data_enc)
            loss_class = criterion(output_class, label_class)
            loss_subject = criterion(output_subject, label_subject)

            loss = loss_class - lam1 * loss_subject
            losses.append(loss.item())

            loss.backward()
            main_opt.step()
            adv_opt.step()
            enc_opt.step()

    if output_state:
        return enc, main_clf, adv_clf, losses
    else:
        return enc, main_clf, adv_clf


################################################################ Functions for testing #########################################################
# Test the model with single samples
# Test the model with single samples
def test_sample(enc, clf, test_set, label_target, train_dict, device, val_state=False):
    # Basic properties for testing the set
    correct, total = 0, 0
    testloader = DataLoader(test_set, batch_size=test_set.__len__(), shuffle=False) # Load all the data together
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # Load the data
            data_sample = data['data_sample'].to(device)
            data_enc = enc(data_sample)
            label = data[label_target].to(device)

            output_label = clf(data_enc)

            # Evaluate the performance
            _, predicted_label = torch.max(output_label.data, 1)
            predicted_label_conv = []
            for p_label in predicted_label:
                predicted_label_conv.append(train_dict[p_label.item()])

            total += label.size(0)
            correct += len([i for i, e in enumerate(predicted_label_conv) if e == label[i]])
    accuracy = correct / total

    if val_state:
        return predicted_label_conv, accuracy
    else:
        return predicted_label_conv


# Test the model with accumulated evidence
# accu_len: the number of trials before one target trial for getting its prediction
def test_acc_evi(enc, clf, accu_len, test_set, label_target, device, val_state=False):
    # Cut the testing set according to the class and subject labels, so that the discountinuity can be removed
    testloader = DataLoader(test_set, batch_size=test_set.__len__(), shuffle=False) # Load all the data together
    # Load the data
    data = next(iter(testloader))
    data_sample = data['data_sample']
    label_subject = data['subject']
    label_class = data['class']

    # Remove the discountinuity
    subject_unique = np.unique(label_subject)
    class_unique = np.unique(label_class)
    cont_idx = []
    for sub in subject_unique:
        for cla in class_unique:
            idx_temp_s = [i for i, s in enumerate(label_subject) if s == sub]
            idx_temp_c = [i for i, c in enumerate(label_class) if c == cla]
            idx_temp = [i for i in idx_temp_c if i in idx_temp_s]
            idx_discont = prep.ranges(idx_temp)
            for idx_slice in idx_discont:
                cont_idx.append(list(range(idx_slice[0], idx_slice[1] + 1)))
    
    # Use accumulated evidence to test the model
    gold_label = []
    
    predict_orig = []
    predict_mean = []
    predict_vote = []
    with torch.no_grad():
        for idx_case in cont_idx:
            data_sample_temp = data_sample[idx_case].to(device)
            label = data[label_target][idx_case].to(device)
            data_enc = enc(data_sample_temp)
            output_score = np.array(clf(data_enc).tolist())

            for i in range(accu_len, len(idx_case)):
                
                gold_label.append(label[i])

                output_temp = output_score[i - accu_len:i]
                
                
                output_temp_mean = np.mean(output_temp, axis=0)           
                predict_mean.append(np.argmax(np.array(output_temp_mean)))

                output_temp_argmax = np.argmax(output_temp, axis=1)
                predict_vote.append(np.argmax(np.bincount(output_temp_argmax)))

                predict_orig.append(np.argmax(output_score[i]))
    
    # Evaluate the metrics with the predictions according to mean scores or voting
    acc_orig = len([i for i in range(len(gold_label)) if predict_orig[i] == gold_label[i]]) / len(gold_label)
    acc_mean = len([i for i in range(len(gold_label)) if predict_mean[i] == gold_label[i]]) / len(gold_label)
    acc_vote = len([i for i in range(len(gold_label)) if predict_vote[i] == gold_label[i]]) / len(gold_label)

    predict_results = [gold_label, predict_mean, predict_vote]

    if val_state:
        return acc_orig, acc_mean, acc_vote, predict_results
    else:
        return predict_results


################ Functions for creating and splitting the dataset #################
# Exclude one participant (subject) as the testing set and keep the other participants as training set
def sub_exclude(data_samples, class_label, subject_label, exclude_idx):
    subject_unique = np.unique(subject_label)

    train_idx = [i for i, e in enumerate(subject_label) if e != subject_unique[exclude_idx]]
    test_idx = [i for i, e in enumerate(subject_label) if e == subject_unique[exclude_idx]]

    train_dict = list(range(len(subject_unique)))
    train_dict.remove(exclude_idx)

    painDataset_train = trans_net.PainDataset(data_samples[train_idx], class_label[train_idx], subject_label[train_idx])
    painDataset_test = trans_net.PainDataset(data_samples[test_idx], class_label[test_idx], subject_label[test_idx])   

    return painDataset_train, painDataset_test, train_dict


# Combine the data from particular subjects
def sub_combine(data_samples, class_label, subject_label, sub_idxs):
    subject_unique = np.unique(subject_label)
    data_idx = [i for i, e in enumerate(subject_label) if e in subject_unique[sub_idxs]]
    return trans_net.PainDataset(data_samples[data_idx], class_label[data_idx], subject_label[data_idx])   

# When there are n participants to make a set, select 1/n data from each participant
def sub_combine_fair(data_samples, class_label, subject_label, sub_idxs):
    subject_unique = np.unique(subject_label)
    data_idx  = []
    n = len(sub_idxs)
    for sub_idx in sub_idxs:
        data_idx_temp = [i for i, e in enumerate(subject_label) if e == sub_idx]
        # Randomly select 1/n of the data
        random.shuffle(data_idx_temp)
        data_idx = data_idx + data_idx[0:int((1 / n) * len(data_idx_temp))]
    return trans_net.PainDataset(data_samples[data_idx], class_label[data_idx], subject_label[data_idx])  

# Split the training set into a training set and a validation set
def train_test(data_samples, class_label, subject_label, ratio):
    data_idx = list(range(data_samples.shape[0]))
    random.shuffle(data_idx)

    train_idx = data_idx[0:int(ratio * len(data_idx))]
    test_idx = data_idx[int(ratio * len(data_idx)):]

    painDataset_train = trans_net.PainDataset(data_samples[train_idx], class_label[train_idx], subject_label[train_idx])
    painDataset_test = trans_net.PainDataset(data_samples[test_idx], class_label[test_idx], subject_label[test_idx])   

    return painDataset_train, painDataset_test

# Select the data from particular classes
def select_data_class(data_samples, class_label, subject_label, target_classes):

    data_idx = [i for i, e in enumerate(class_label) if e in target_classes]
    return data_samples[data_idx], class_label[data_idx], subject_label[data_idx] # It was not ensembled into an object for the further processing (It is supposed to be called before creating the datasets)

# Convert the class labels into 0 and 1 for binary clasification
def bin_conv(class_label):
    class_unique = np.unique(class_label)
    class_bin_label = [0] * class_label.shape[0]

    one_idx = [i for i, e in enumerate(class_label) if e == class_unique[1]]
    for idx in one_idx:
        class_bin_label[idx] = 1
    return np.array(class_bin_label)

# Find the first n participants with the sum at least above 50%
def find_common_thre(common_sub_list, threshold):
    list_temp = sorted(common_sub_list, reverse=True) #A copy for sorting the list
    print(list_temp)
    sum_ratio = 0 # Sum of the common subjects' ratios
    n_subject = []
    k = 0
    while sum_ratio < threshold:
        sum_ratio += list_temp[k]
        n_subject.append(common_sub_list.index(list_temp[k]))
        k += 1
    
    return n_subject


