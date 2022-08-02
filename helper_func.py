## The class for all the objects used in the adversarial model
import numpy as np
from tensorflow import keras
from tensorflow.keras import Model, Training

class AdversarialModel:
    def __init__(self, data_samples, class_label, subject_label, main_clf, adv_clf, loss_func):
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

        # Assign the models
        self.main_clf = main_clf
        self.adv_clf = adv_clf

        # Compile the model with 


