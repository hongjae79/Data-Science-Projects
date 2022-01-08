# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:32:09 2021

@author: hongj
"""
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

def create_model():

    i = Input(shape=(48,48,1))
    x = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(i)
    x = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(7, activation='softmax')(x)
    model = Model(inputs=i, outputs=x)

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    return model