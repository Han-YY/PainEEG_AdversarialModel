## The class for all the objects used in the adversarial model
##### Adapted from https://github.com/philipph77/ACSE-Framework
import numpy as np
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalCrossentropy
import tensorflow as tf
import trans_net
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.models import  Model

import numpy as np

import pandas as pd
import os
from tqdm import tqdm, trange
from data_prep_func import get_confusion_matrix
import random
import data_prep_func as prep

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

        # Get the number of classes and subjects involved in the classeification
        class_unique = np.unique(class_label)
        class_count = class_unique.shape[0]
        subject_unique = np.unique(subject_label)
        subject_count = subject_unique.shape[0]

        # Assign the models
        self.lam = lam
        

        self.main_clf = trans_net.classifier_model(class_count=class_count)
        self.adv_clf = trans_net.adversary_model(sub_count=subject_count)
        # self.main_clf.summary()

        # Compile the model with the loss function for adapting the adversarial process
        input_shape = layers.Input(shape=(32, 32, 1))
        print(input_shape)
        output = self.main_clf(input_shape)
        leakage = self.adv_clf(input_shape)

        self.adv_clf.trainable = False # Freeze the adversary classifier

        self.frame = Model(input_shape, [output, leakage])
        self.frame.compile(loss=[lambda x, y: CategoricalCrossentropy(x, y, from_logits=True),
                                    lambda x, y: CategoricalCrossentropy(x, y, from_logits=True)], 
        loss_weights = [1., -1. * lam], optimizer=Adam(learning_rate=1e-3, decay=1e-4), metrics=[tf.keras.metrics.Accuracy()])

        self.adv_clf.trainable = True # Unfreeze the adversary classifier
        self.adv_clf.compile(loss=lambda x, y: CategoricalCrossentropy(x, y, from_logits=True), 
                                     loss_weights=self.lam, optimizer=Adam(learning_rate=1e-3, decay=1e-4), metrics=[tf.keras.metrics.Accuracy()])

    # The function for training the model with speficified loss
    
    def train(self, log, early_stopping_after_epochs=10, epochs=50, batch_size=200, run_name=''):
        self.writer = tf.summary.create_file_writer(log+'/tensorboard_logs/'+run_name+'/')
        monitoring_adv_val_acc = list()

        # Split the data into training and testing sets
        # Shuffle the indices
        dataset_indices = list(range(self.data_samples.shape[0]))
        random.shuffle(dataset_indices)
        train_idx = dataset_indices[0:int(0.75 * self.data_samples[0].shape[0])]
        test_idx = dataset_indices[int(0.75 * self.data_samples[0].shape[0]):]

        x_train, y_train, s_train = self.data_samples[train_idx], self.class_label[train_idx], self.subject_label[train_idx]
        x_test, y_test, s_test = self.data_samples[test_idx], self.class_label[test_idx], self.subject_label[test_idx]


        train_index = np.arange(y_train.shape[0])
        train_batches = [(i * batch_size, min(y_train.shape[0], (i + 1) * batch_size))
                        for i in range((y_train.shape[0] + batch_size - 1) // batch_size)]

        # Earlt stopping variables
        es_wait = 0
        es_best = np.Inf
        es_best_weights = None


        for epoch in range(1, epochs + 1):
            print('[{} - {}] Epoch {}/{}'.format(run_name, str(self.lam), epoch, epochs))
            np.random.shuffle(train_index)
            train_log = []
            for iter, (batch_start, batch_end) in enumerate(tqdm(train_batches)):
                batch_ids = train_index[batch_start:batch_end]
                x_train_batch = x_train[batch_ids]
                y_train_batch = y_train[batch_ids]
                s_train_batch = s_train[batch_ids]
                
                self.adv_clf.train_on_batch(x=x_train_batch, y=s_train_batch)
                train_log.append(self.main_clf.train_on_batch(x=x_train_batch, y=[y_train_batch, s_train_batch]))

            train_log = np.mean(train_log, axis=0)
            val_log = self.main_clf.test_on_batch(x_test, [y_test, s_test])

            [y_pred_train, s_pred_train] = self.main_clf.predict(x_train)
            [y_pred_val, s_pred_val] = self.main_clf.predict(x_test)

            monitoring = True
            if monitoring:
            
                self.adv_clf.fit(x_train, np.argmax(s_train, axis=1))
                
                monitoring_adv_val_acc.append(self.adv_clf.score(x_test, np.argmax(s_test, axis=1)))
                

            # Logging model training information per epoch
            print("[%s  - %s] Train - [Loss: %f] - [CLA loss: %f, acc: %.2f%%] - [ADV loss: %f, acc: %.2f%%]"
                % (run_name, str(self.lam), train_log[0], train_log[1], 100*train_log[3], train_log[2], 100*train_log[4]))
            print("[%s - %s] Validation - [Loss: %f] - [CLA loss: %f, acc: %.2f%%] - [ADV loss: %f, acc: %.2f%%]"
                % (run_name, str(self.lam), val_log[0], val_log[1], 100*val_log[3], val_log[2], 100*val_log[4]))
            with open(log + '/train.csv', 'a') as f:
                f.write(str(epoch) + ',' + str(train_log[0]) + ',' + str(train_log[1]) + ',' +
                        str(100*train_log[3]) + ',' + str(train_log[2]) + ',' + str(100*train_log[4]) + '\n')
            with open(log + '/validation.csv', 'a') as f:
                f.write(str(epoch) + ',' + str(val_log[0]) + ',' + str(val_log[1]) + ',' +
                        str(100*val_log[3]) + ',' + str(val_log[2]) + ',' + str(100*val_log[4]) + '\n')
        # Logging data to tensorboard
        with self.writer.as_default():
            with tf.name_scope("Encoder"):
                for layer in self.main_clf.layers[1].layers[1].layers:
                    if not(len(layer.trainable_weights)==0):
                        tf.summary.histogram("Encoder %s /weights"%layer.name, layer.get_weights()[0], step=epoch)
            with tf.name_scope("Classifier"):
                tf.summary.histogram("Classifier layer-%s /weights"%self.main_clf.layers[2].layers[1].name, self.main_clf.layers[2].layers[1].get_weights()[0], step=epoch)
            with tf.name_scope("Main Adversary"):
                tf.summary.histogram("Adversary layer-%s /weights"%self.main_clf.layers[3].layers[1].name, self.main_clf.layers[3].layers[1].get_weights()[0], step=epoch)
            with tf.name_scope("ADV Adversary"):
                tf.summary.histogram("Adversary layer-%s /weights"%self.adv_clf.layers[1].name, self.adv_clf.layers[1].get_weights()[0], step=epoch)

            with tf.name_scope('Losses'):
                tf.summary.scalar("Main Train-Loss",train_log[0], step=epoch)
                tf.summary.scalar("ADV Train-Loss",train_log[2], step=epoch)
                tf.summary.scalar("Main Validation-Loss",val_log[0], step=epoch)
                tf.summary.scalar("CLA Validation-Loss",val_log[1], step=epoch)
            with tf.name_scope('Accuracies'):
                tf.summary.scalar("Classifier-Accuracy (Train)",train_log[3], step=epoch)
                tf.summary.scalar("Adversary-Accuracy (Train)",train_log[4], step=epoch)
                tf.summary.scalar("Classifier-Accuracy (Val)",val_log[3], step=epoch)
                tf.summary.scalar("Adversary-Accuracy (Val)",val_log[4], step=epoch)

            cm_cla_train = get_confusion_matrix(y_pred_train, y_train)
            cm_adv_train = get_confusion_matrix(s_pred_train, s_train)
            cm_cla_val = get_confusion_matrix(y_pred_val, y_test)
            cm_adv_val = get_confusion_matrix(s_pred_val, s_test)
            with tf.name_scope("Classifier Train - Confusion Matrices"):
                tf.summary.image("Classifier Train", cm_cla_train, step=epoch)
            with tf.name_scope("Classifier Validation - Confusion Matrices"):
                tf.summary.image("Classifier Validation", cm_cla_val, step=epoch)
            with tf.name_scope("Adversary Train - Confusion Matrices"):
                tf.summary.image("Adversary Train", cm_adv_train, step=epoch)
            with tf.name_scope("Adversary Validation - Confusion Matrices"):
                tf.summary.image("Adversary Validation", cm_adv_val, step=epoch)

        self.writer.flush()
        
    # Function for fitting the main classifier simply
    def fit(self, epochs=500, batch_size=10, callbacks=None):
        # x_train, y_train, p_train = train_set #last two components are adversary
        # x_test, y_test, p_test = test_set
        dataset_indices = list(range(self.data_samples.shape[0]))
        random.shuffle(dataset_indices)
        train_idx = dataset_indices[0:int(0.75 * (self.data_samples.shape[0]))]
        test_idx = dataset_indices[int(0.75 * (self.data_samples.shape[0])):]

        x_train, y_train, s_train = self.data_samples[train_idx], self.class_label[train_idx], self.subject_label[train_idx]
        x_test, y_test, s_test = self.data_samples[test_idx], self.class_label[test_idx], self.subject_label[test_idx]


        self.main_clf.fit([x_train], [y_train, s_train], validation_data= ([x_test], [y_test, s_test]), 
                    epochs = epochs, batch_size= batch_size, callbacks=callbacks, verbose = 2)