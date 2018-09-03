#!/usr/bin/python

from utils import *

import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from pymongo import MongoClient

from keras.utils.np_utils import to_categorical

import sys
from collections import Counter
import math
import os 
import pickle 
import datetime
import keras


class LossHistory(keras.callbacks.Callback):
    #self.mongo_id = mongo_id
    def __init__(self, mongo_id, mongo_res):
        self.mongo_id = mongo_id
        self.mongo_res = mongo_res

    def on_train_begin(self, logs=[]):
        self.losses = []
    def on_epoch_end(self, batch, logs=()):
        del(logs['lr'])
        self.losses.append(logs)
        self.mongo_res.update_one({"_id": self.mongo_id}, {"$set": {"history": self.losses}})

def get_class_weights(y, smooth_factor=0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}



if (len(sys.argv) > 1):
    instrument = sys.argv[1]
    mode = sys.argv[2]
    window = int(sys.argv[3])
    forecast = int(sys.argv[4])
else:
    instrument =  os.environ['INSTRUMENT']
    mode =  os.environ['MODE']
    window = int(os.environ['WINDOW'])
    forecast = int(os.environ['FORECAST'])







if mode == "dev":
    client = MongoClient('mongodb://localhost:27017')
else:
    client = MongoClient('mongodb://mongo:27017')



#db.ticks.createIndex({ time: 1})

print("Instrument is " + instrument)
db = client.inklin


run_collection = db.run_statistics


bid = []
ask = []


#ticksdb = db.ig_ticks
ticksdb = db.ticks

c_bid = np.empty(shape=[0, 1])
c_ask = np.empty(shape=[0, 1])

time_ago = datetime.datetime.utcnow() - datetime.timedelta(days=14)

max_nans = 0
functions = []
nn_inputs = []
ta_inputs = []


if (os.path.isfile(instrument + ".p")):
    print ("Loading cached data...")
    all_ticks = pickle.load( open( instrument + ".p", "rb" ))
else:
    all_ticks = []
    print ("Downloading data...")

    ticks = ticksdb.find({"instrument": instrument, "time":{"$gte": time_ago }}).sort('time', 1)

    for tick in ticks:
        all_ticks.append(tick)

    pickle.dump( all_ticks, open( instrument + ".p", "wb" ) )


print ("Structuring data...")

for tick in all_ticks:
    bid.append(float(tick['bid']))
    ask.append(float(tick['ask']))



WINDOW = window
EMB_SIZE = 2

STEP = 1
FORECAST = forecast


X, Y = [], []

ups = 0
downs = 0
neutral = 0

ta_inputs = []


ta_inputs.insert(0, bid)
ta_inputs.insert(0, ask)

bid = []
ask = []

i = 0

print ("Using " + str(len(ta_inputs)) + " features")
EMB_SIZE = len(ta_inputs)


print ("Normalising data...")

for i in range(0, len(ta_inputs[0]), STEP): 
    try:
        ta_window = []
        c = 0   
        for ta_input in ta_inputs:
            ta_input = ta_input[i:i+WINDOW]

            norm_array = (np.array(ta_input) - np.mean(ta_input)) / np.std(ta_input)

            ta_window.append(norm_array)
            c =c +1

        x_i = ta_inputs[1][i:i+WINDOW]
        y_i = ta_inputs[1][i+WINDOW+FORECAST]  

        last_close = x_i[-1]
        next_close = y_i

        spread = ta_inputs[0][i+WINDOW] - ta_inputs[1][i+WINDOW]


        if last_close < next_close:
            if (next_close - last_close) > spread:
                y_i = 1
                ups = ups + 1
            else:
                y_i = 2
                neutral = neutral + 1
        else:
            if (last_close - next_close) > spread:
                downs = downs + 1
                y_i = 0
            else:
                y_i = 2
                neutral = neutral + 1

        x_i = np.column_stack(ta_window)

    except Exception as e:
        
        print(e)
        break

    X.append(x_i)
    Y.append(y_i)

    #if y_i != 2:
    #    X.append(x_i)
    #    Y.append(y_i)
    
    #if y_i == 2 and (neutral <= ups or neutral <= downs):
    #    #print("HERE")
    #    X.append(x_i)
    #    Y.append(y_i)
    #    neutral = neutral + 1


ta_inputs = []

print(ups)
print(downs)
print(neutral)




#EMB_SIZE = 5

X, Y = np.array(X), np.array(Y)


print ("Splitting Train and Test...")
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y)

tmp_X_train = []
tmp_Y_train = []

tmp_X_test = []
tmp_Y_test = []

counter = 0

threshold = max(ups, downs)
final_neutrals = 0

for yt in Y_train:
    if yt == 1 or yt == 0:
        tmp_X_train.append(X_train[counter])
        tmp_Y_train.append(Y_train[counter])

    if yt == 2 and final_neutrals < threshold:
        tmp_X_train.append(X_train[counter])
        tmp_Y_train.append(Y_train[counter])
        final_neutrals = final_neutrals + 1

    counter = counter + 1

counter = 0

for yt in Y_test:
    if yt == 1 or yt == 0:
        tmp_X_test.append(X_test[counter])
        tmp_Y_test.append(Y_test[counter])

    counter = counter + 1

number_of_uds = int(len(Y_test)/2)

counter = 0

final_neutrals = 0

for yt in Y_test:
    if yt == 2 and final_neutrals < number_of_uds:
        tmp_X_test.append(X_test[counter])
        tmp_Y_test.append(Y_test[counter])
        final_neutrals = final_neutrals + 1

    counter = counter + 1


X_train = np.array(tmp_X_train)
Y_train = np.array(tmp_Y_train)
Y_test = np.array(tmp_Y_test)
X_test = np.array(tmp_X_test)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], EMB_SIZE))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], EMB_SIZE))


run_creation = {"instrument": instrument, "window": window, "forecast": forecast, "start": datetime.datetime.utcnow(), "examples": len(X_train), "history":[]}
id = run_collection.insert_one( run_creation )

history = LossHistory(id.inserted_id, run_collection)


model = Sequential()
model.add(Convolution1D(input_shape = (WINDOW, EMB_SIZE),
                        nb_filter=16,
                        filter_length=8,
                        border_mode='same'))

model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.25))

model.add(Convolution1D(nb_filter=8,
                        filter_length=4,
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.25))


model.add(Flatten())

model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU())


model.add(Dense(3))
model.add(Activation('softmax'))

opt = Nadam(lr=0.002)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)

if mode == "prod":
    checkpointer = ModelCheckpoint(filepath="/models/" + instrument + "-" + str(window) + ".hdf5", verbose=0, save_best_only=True)
else:
    checkpointer = ModelCheckpoint(filepath=instrument + "-" + str(window) + ".hdf5", verbose=0, save_best_only=True)
    


print ("Compiling model...")

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print ("Fitting...")

unique, counts = np.unique(Y_train, return_counts=True)
print(dict(zip(unique, counts)))


y_train = to_categorical(Y_train, 3)
y_test = to_categorical(Y_test, 3)

class_weights = get_class_weights(Y_train, 0.1)

print(class_weights)

history = model.fit(X_train, y_train, 
          epochs = 100, 
          batch_size = 128, 
          verbose=1, 
          validation_data=(X_test, y_test),
          callbacks=[reduce_lr, checkpointer, history],
          shuffle=True) #,
          #class_weight=class_weights)

model.load_weights(instrument + "-" + str(window) + ".hdf5")
pred = model.predict_classes(np.array(X_test))

run_collection.update_one({"_id": id.inserted_id}, {"$set": {"end": datetime.datetime.utcnow()}})
