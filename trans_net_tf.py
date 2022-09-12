## The networks in different steps of the adversarial model
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

# The neural network for clasifying the labels with the connectivity features
# class_count: the number of classes in classification (for multiclassification)
def classifier_model(class_count):
    dim = (32, 32, 1)
    input_shape = (dim)
    Input_words = layers.Input(shape=input_shape, name='inpud_vid')
    # CNN
    x = layers.Conv2D(filters=128, kernel_size=(7,7), padding='same', activation='relu')(Input_words)
    x = layers.MaxPooling2D(pool_size=(3,3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(3,3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(3,3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout2D(0.2)(x)
    # Flatten
    x = layers.Flatten()(x)
    x = layers.Dense(100, kernel_regularizer='l2')(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    x = layers.Activation('sigmoid')(x)
    x = layers.Dense(class_count)(x)
    out = layers.Softmax()(x)
    model = keras.Model(inputs=Input_words, outputs=[out], name='main_clf')

    return model

#The neural network for classifying the participant ids
# sub_count: the amount of subjects involved in the classification
def adversary_model(sub_count):
    dim = (32, 32, 1)
    input_shape = (dim)
    Input_words = layers.Input(shape=input_shape, name='inpud_vid')
    # CNN
    x = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(Input_words)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    out = layers.Dense(sub_count)(x)
    # out = layers.Softmax()(x)
    model = keras.Model(inputs=Input_words, outputs=[out], name='adv_clf')

    return model
