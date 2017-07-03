#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import json
import math
import keras.backend as K
from keras.models import load_model

import gc

try:
    import cPickle as pickle
    #import _pickle as pickle
except:
    print "exeption from impport cpickle"
    import picklepathModelJson

import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,\
    Conv2D,Convolution2D, MaxPooling2D, InputLayer, Merge, BatchNormalization, ZeroPadding2D, Dropout, UpSampling2D, Reshape
from keras import optimizers as opt
#from keras.utils.visualize_util import plot as kplot
import skimage.io as skio
import skimage.transform as sktf
import skimage.color as skolor

from keras.utils import np_utils

import numpy as np
import pandas as pd


##############################################
def readImageAndReshape(fimg):#loads only img as grey in tf shape
    timg = (skio.imread(fimg).astype(np.float) / 255.)[...,None]
    #plt.imshow(timg[:,:,0])
    return timg

##############################################

def loadModelFromH5(pathModel):
    if not os.path.isfile(pathModel):
        raise Exception('Cant find H5-file [%s]' % pathModel)
    print(pathModel," pathmodel")

    model = keras.models.load_model(pathModel)
    return model


##############################################
def getModel(defModel, inpShape =(512, 512),n_class = 2):

    if len(inpShape) < 3:
        inpShape = np.append(np.array(inpShape),1)
    #2DO: add radiobutton status check
    elif inpShape[2] == 3:

        inpShape = np.append(np.array(inpShape[:-1]),1)



    retModel = buildFCNModel(inpShape,n_class)
    for i in range(len(defModel.layers)):
        weights = defModel.layers[i].get_weights()
        retModel.layers[i].set_weights(weights)
    return retModel, defModel


##############################################


def buildFCNModel(inpShape=(512, 512, 1), n_class=2, n_layers=5,
              n_filters=8, kernel_size=(3, 3), activ_f="relu"):
    n_filt = n_filters
    model = Sequential()
    model.add(Conv2D(filters=n_filt, kernel_size=kernel_size, padding='same', input_shape=inpShape))
    ##model.add(Conv2D(filters = n_filt,kernel_size=kernel_size,padding='same'))

    model.add(BatchNormalization())
    model.add(Activation(activ_f))
    ##model.add(BatchNormalization())#
    ##model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    for i in range(n_layers):
        n_filt *= 2
        model.add(Conv2D(filters=n_filt, kernel_size=kernel_size, padding='same'))
        ##model.add(Conv2D(filters = n_filt,kernel_size=kernel_size,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation(activ_f))
        ##model.add(BatchNormalization())#

        ##model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=n_filt, kernel_size=kernel_size, padding='same'))
    ##model.add(Conv2D(filters = n_filt,kernel_size=kernel_size,padding='same'))
    kernel_size = (3, 3)
    model.add(UpSampling2D(size=(2, 2)))
    for i in range(n_layers):
        n_filt = n_filt // 2
        model.add(Conv2D(filters=n_filt, kernel_size=kernel_size, padding='same'))
        ##model.add(Conv2D(filters = n_filt,kernel_size=kernel_size,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation(activ_f))
        ##model.add(BatchNormalization())#
        model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(filters=n_class, kernel_size=kernel_size, padding='same'))

    tmpShape = model.layers[-1].get_output_shape_at(0)
    model.add(Reshape((tmpShape[1] * tmpShape[2], n_class)))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    model.add(Reshape((tmpShape[1], tmpShape[2], n_class)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


##############################################
def buildProbMap(defModel, pimg):
    model, defModel = getModel(defModel,pimg.shape)
    data_img = (pimg[:,:,0].astype(np.float) / 255.)[...,None]
    data_img = data_img[None,...]
    print(data_img.shape, "data_img.shape")
    retMapFNN = model.predict_on_batch(data_img)[0]
    inpShapeFCNN = model.layers[0].get_input_shape_at(0)[1:]
    numLblFCNN = model.layers[-1].get_output_shape_at(0)[3]#1
    nch = inpShapeFCNN[2]
    nrow = inpShapeFCNN[0]
    ncol = inpShapeFCNN[1]
    nrowCNN = retMapFNN.shape[0]
    ncolCNN = retMapFNN.shape[1]
    tretProb0 =retMapFNN
    retProb = np.zeros((nrow, ncol, numLblFCNN))
    tdr = (nrow - 1)//2
    tdc = (ncol - 1)//2
    retProb[-nrowCNN//2+nrow//2: nrowCNN//2 + nrow//2, -ncolCNN//2+ncol//2: ncolCNN//2 + ncol//2, :] = tretProb0
    return retProb, defModel
##############################################
