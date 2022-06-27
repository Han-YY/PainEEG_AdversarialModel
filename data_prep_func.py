import numpy as np
import pandas as pd
import scipy as sp

import matplotlib
import math
from math import e
import mne

# import tensorflow as tf
# from tensorflow import keras

# import os
# import sys

# IMPORTANT NOTES:
# Labels: 0-Hot, 1-Warm, 2-EO, 3-EC, 4-S

# Function for reading a dataset from the EEG dataset and creating its corresponding label tuple
# Input:
# -file_path: file path of the .set file
# -subject: number of the subject
# -tag: the tag got from the file name
# -ch_index: channel names used in this analysis

# Output:
# -dataset: dataset
# -dataset_labels: condition labels toward the dataset
# -dataset_subject: subject numbers along the dataset

def create_ISPC_dataset(file_path, subject, tag, ch_valid):
    # Read and filtering the data
    phase_epochs_1, data_raw_1 = eeg_preprocessing(file_path, lo_fre=7, hi_fre=9, overlap_ratio=0.5)
    phase_epochs_2, data_raw_3 = eeg_preprocessing(file_path, lo_fre=9, hi_fre=11, overlap_ratio=0.5)
    phase_epochs_3, data_raw_3 = eeg_preprocessing(file_path, lo_fre=11, hi_fre=13, overlap_ratio=0.5)
    sfreq = data_raw_1.info['sfreq']
    
    # Get the ISPC data (tensor)
    # Firstly get the indices of the valid channels
    ch_index = []
    channel_names = data_raw_1.ch_names
    for ch_valid_name in ch_valid:
        ch_index.append(channel_names.index(ch_valid_name))

    ISPCs_array_1 = get_ISPC(phase_epochs_1, 1, ch_index)
    ISPCs_array_2 = get_ISPC(phase_epochs_2, 1, ch_index)
    ISPCs_array_3 = get_ISPC(phase_epochs_3, 1, ch_index)

    print(ISPCs_array_1.shape)
    
    # Concatenate the ISPCs into an 'image' with 3 layers
    ISPCs_array = np.concatenate((ISPCs_array_1, ISPCs_array_2, ISPCs_array_3), 3)

    print(ISPCs_array.shape)

    # Concatenate the ISPC 'images' into frames with 20 images in a frame
    ISPCs_clips = np.zeros((ISPCs_array.shape[0] - 20, 20, ISPCs_array.shape[1], ISPCs_array.shape[2], 3))
    for i in range(ISPCs_array.shape[0] - 20):
        ISPCs_clips[i] = ISPCs_array[i:i+20]


    # Convert the tensor into a dataset and create the corresponding labels
    frame_labels, frame_subject = create_dataset_label(ISPCs_array = ISPCs_array, subject_number=subject, label_tag=tag)
    # clip_labels, clip_subject = create_dataset_label(ISPCs_array = ISPCs_clips, subject_number=subject, label_tag=tag)

    # return ISPCs_clips, ISPCs_array, frame_labels, frame_subject, clip_labels, clip_subject
    return ISPCs_array, frame_labels, frame_subject

# Function for reading a dataset from the EEG dataset and creating its corresponding label tuple
# Input:
# -file_path: file path of the .set file
# -subject: number of the subject
# -tag: the tag got from the file name
# -ch_index: channel names used in this analysis

# Output:
# ISPCs_matrix: the matrix containing all corresponding ISPCs, whose columns and rows represents the channels
# frame_labels: the label of each sample
# frame_subject: the subject number of each sample


def create_ISPC_matrix(file_path, subject, tag, ch_valid, lo_fre, hi_fre, epoch_time):
    # Read and filtering the data
    
    phase_epochs_1, data_raw_1 = eeg_preprocessing(file_path, lo_fre=lo_fre, hi_fre=hi_fre, overlap_ratio=0.5, epoch_time=epoch_time)
    
    
    sfreq = data_raw_1.info['sfreq']
    
    # Get the ISPC data (tensor)
    # Firstly get the indices of the valid channels
    ch_index = []
    channel_names = data_raw_1.ch_names
    for ch_valid_name in ch_valid:
        ch_index.append(channel_names.index(ch_valid_name))

    ISPCs_matrix = get_ISPC(phase_epochs_1, 1, ch_index)
    ISPCs_matrix = np.reshape(ISPCs_matrix, (ISPCs_matrix.shape[0], ISPCs_matrix.shape[1], ISPCs_matrix.shape[2]))

    print(ISPCs_matrix.shape)



    # Convert the tensor into a dataset and create the corresponding labels
    frame_labels, frame_subject = create_dataset_label(ISPCs_array = ISPCs_matrix, subject_number=subject, label_tag=tag)

    return ISPCs_matrix, frame_labels, frame_subject

# Pre-processing functions (Filtering and denoising)
# Input:
# -file_path: the file path of .set file
# Output:
# -phase_epochs: the data epochs of signal phases from Hilbert transform
# -data_raw: the raw data read from the data file
def eeg_preprocessing(file_path, lo_fre, hi_fre, overlap_ratio, epoch_time):
    ## Read the test file
    data_raw = mne.io.read_raw_eeglab(file_path,preload=True)

    ## Filter the data to remove the DC components and line noise
    # Remove line noise
    raw_removal_line = data_raw.copy().notch_filter(
        freqs=50, method='spectrum_fit', filter_length='10s')

    ## Reference with surface Laplacian (CSD)
    data_raw = raw_removal_line.pick_types(eeg=True, stim=True).load_data()
    # data_raw = data_raw.set_eeg_reference(projection=True).apply_proj()
    raw_csd = mne.preprocessing.compute_current_source_density(data_raw) # Compute the CSD


    # Filter the signals into alpha band
    data_alpha = raw_csd.copy().filter(l_freq=lo_fre, h_freq=hi_fre) # Alpha filter
    

    ## Get the signals' phase
    # Get the date from the alpha data
    data_chunk = data_alpha.get_data()
    data_phase = np.empty(shape=data_chunk.shape)
    # Use Hilbert transform to get the data's phases
    i = 0
    for data_channel in data_chunk:
        data_phase[i] =np.angle(sp.signal.hilbert(data_channel))
        i = i + 1
    
    
    # Segment the phases into epochs as 3-D array
    sfreq = data_alpha.info['sfreq']
    phase_epochs = crop_overlap(data=data_phase, length=sfreq * epoch_time, overlap=sfreq * overlap_ratio)
    
    return phase_epochs, data_raw
    


# the function for cropping data into epochs with overlap
# Input:
# -data: input data, which is a 2-D narray
# -length: length of each epoch (points)
# -overlap: overlapping length between neighbor epochs (points)
# Output:
# - data_epochs: 3-D narray
def crop_overlap(data, length, overlap_ratio):
    # Initialize the output array
    channel_number = data.shape[0]
    epoch_number = int(math.floor(((data.shape[1] - length)/overlap)))
    overlap = length * overlap_ratio

    data_epochs = np.zeros(shape=(channel_number, epoch_number, int(length))) # Create the empty array for output

    # Crop the data
    epoch_index = 0
    for i in range(0, epoch_number * int(overlap), int(overlap)):
        for ch in range(channel_number):
            data_epochs[ch][epoch_index] = data[ch][i:i + int(length)]
        epoch_index = epoch_index + 1
    
    # return the output array
    return data_epochs


from math import e

def get_ISPC(data_phase, epochs_number, channels_index):
    # Initialize an array to store all ISPCs
    ISPCs_array = np.zeros(
        shape=(data_phase.shape[1], len(channels_index), len(channels_index)))
    
    # Calculate ISPCs
    
    for ch_1 in range(len(channels_index)):
        channel_1 = channels_index[ch_1]
        
        for ch_2 in range(ch_1 + 1, len(channels_index)):
            channel_2 = channels_index[ch_2]

            for i in range(data_phase.shape[1]):
                phase_diff = data_phase[channel_1][i] - data_phase[channel_2][i]
                
                phase_diff_comp = []
                for p_diff in phase_diff:
                    phase_diff_comp.append(e ** (1j * p_diff))
                
                ISPCs_array[i][ch_1][ch_2] = abs(np.mean(phase_diff_comp))
                ISPCs_array[i][ch_2][ch_1] = abs(np.mean(phase_diff_comp))
            
    
    # Create the tuple storing the neighbor ISPCs as time series
    # ISPCs_time_series = np.zeros(shape=
    #     (ISPCs_array.shape[1] - epochs_number + 1, ISPCs_array.shape[0], epochs_number))
    
    ISPCs_time_series = np.zeros(shape=(
        ISPCs_array.shape[0] - epochs_number + 1, len(channels_index), len(channels_index), epochs_number))

    for index_epoch in range(ISPCs_array.shape[0] - epochs_number + 1):
            ISPCs_time_series[index_epoch] = np.reshape(ISPCs_array[index_epoch:index_epoch + epochs_number], (len(channels_index), len(channels_index), epochs_number))

    # Convert the tuple to tensor
    # ISPCs_array_tf = tf.constant(ISPCs_time_series)
    return ISPCs_time_series


    # Function for converting the tensor to a dataset, and create list to store the subject no. and labels
def create_dataset_label(ISPCs_array, subject_number, label_tag):
    # Convert the data from tensor to a dataset
    # dataset = tf.data.Dataset.from_tensors(ISPCs_array_tf)
    

    if label_tag == 'H':
        label = [1, 0, 0, 0, 0]
    elif label_tag == 'W':
        label = [0, 1, 0, 0, 0]
    elif label_tag == 'O':
        label = [0, 0, 1, 0, 0]
    elif label_tag == 'C':
        label = [0, 0, 0, 1, 0]
    else:
        label = [0, 0, 0, 0, 1]

    dataset_labels = label * ISPCs_array.shape[0]
    dataset_subject = [subject_number] * ISPCs_array.shape[0]

    return dataset_labels, dataset_subject

# Function to read all the files matching the requirements
def read_target_list(class_list, test_number_list, file_list):
    file_read_list = []
    for file_name in file_list:
        if file_name[0:2] in test_number_list and file_name[-3:] == 'set' and file_name[:-4] + '.fdt' in file_list and file_name[-5] in class_list:
            file_read_list.append(file_name)
    return file_read_list
