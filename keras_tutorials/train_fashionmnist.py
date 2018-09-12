#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 01:23:33 2018

@author: rein9
"""
# python train_fashionmnist.py --output single_gpu.png
# python train_fashionmnist.py --output multi_gpu.png --gpus 4

# set the matplotlib backend so figures can be saved in the background
# (uncomment the lines below if you are using a headless server)
# import matplotlib
# matplotlib.use("Agg")

from keras.datasets import fashion_mnist
from keras.callbacks import LearningRateScheduler
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
from pyimagesearch.minigooglenet import MiniGoogLeNet

ap = argparse.ArgumentParse()
ap.add_argument("-o", "--output", required=True,
	help="path to output plot")
ap.add_argument("-g", "--gpus", type=int, default=1,
	help="# of GPUs to use for training")
args = vars(ap.parse_args())

G = args["gpus"]
NUM_EPOCHS = 10
INIT_LR = 1

def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    # the new learning rate
    alpha = baseLR *(1-(epoch/float(maxEpochs)))**power
    return alpha

#Loading data
((trainX, trainY),(testX, testY)) = fashion_mnist.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

#Normalize the data
mean = np.mean(trainX, axis = 0)
trainX -= mean
testX -= mean

# Encoding the label
enc = LabelEncoder()
trainY = enc.fit_transform(trainY)
testY = enc.transform(testY)

#construct the generator for data augmentation and construct the callbacks
aug = ImageDataGenerator(width_shift_range=0.1,
	height_shift_range=0.1, horizontal_flip=True,
	fill_mode="nearest")
callbacks = [LearningRateScheduler(poly_decay)]

#check to see if we are compiling using just a single GPU
if G <=1:
    print("1 GPU only training")
    
    model = MiniGoogLeNet.build(width=32, height=32, depth=3,
			classes=10)
    
else: 
    with tf.device("/cpu:0"):
        model = MiniGoogLeNet.build(width=32, height=32, depth= 3, classes=10)
    model = multi_gpu_model(model, gpus=G)
    
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr = INIT_LR, momentum= 0.9)
#momentum: float >= 0. Parameter that accelerates SGD in the relevant direction and dampens oscillations.
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ['accuracy'])
hist = model.fit_generator(aug.flow(trainX, trainY, batch_size = 64*G),
                           validation_data = (testX, testY),
                           steps_per_epoch = len(trainX)//(64*G),
                           epochs = NUM_EPOCHS,
                           callbacks = callbacks, 
                           verbose = 2)
hist = hist.history

#plot the training loss
N = np.arange(0, len(hist["loss"]))
plt.figure()
plt.plot(N, hist["loss"], label="train_loss")
plt.plot(N, hist["val_loss"], label="test_loss")
plt.plot(N, hist["acc"], label="train_acc")
plt.plot(N, hist["val_acc"], label="test_acc")
plt.title("MiniGoogLeNet on FASHIONMINISR")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

# save the figure
plt.savefig(args["output"])
plt.close()