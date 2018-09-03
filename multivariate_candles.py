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
db = client.fluid


run_collection = db.run_statistics


ticksdb = db.candles
fundamentalsdb = db.fundamentals

candles = ticksdb.find({"ticker": instrument}).sort('date', 1)
fundamentals = fundamentalsdb.find({"ticker": instrument}).sort('date', 1)


openp = []
highp = []
lowp = []
closep = []
volumep = []

#ticks = ticksdb.find({"instrument": instrument, "time":{"$gte": time_ago }}).sort('time', 1)

fundies = []

for candle in candles:
    fundamental = fundamentalsdb.find({"ticker": instrument, "date":{"$lte": candle['date'] }}).sort('date', -1)
    
    for f in fundamental:
        fundies.append(f)
        break

    if len(fundies) > 0:
        openp.append(float(candle['adj_open']))
        highp.append(float(candle['adj_high']))
        lowp.append(float(candle['adj_low']))
        closep.append(float(candle['adj_close']))
        volumep.append(float(candle['adj_volume']))


print len(fundies)
print(len(openp))

fundamentals = []

for key in fundies[0]:
    tmp_arr = [] 
    if key != "_id" and key != "date" and key != "Quarter end" and key != "ticker":
        
        for f in fundies:
            if f[key] == "None":
                f[key] = 0

            tmp_arr.append(float(f[key]))

    if len(tmp_arr) > 0:
        fundamentals.append(tmp_arr)



np.savetxt('test.csv', fundamentals, delimiter=',')

# data_chng = data_original.ix[:, 'Adj Close'].pct_change().dropna().tolist()

WINDOW = 30
EMB_SIZE = 5
STEP = 1
FORECAST = 1




X, Y = [], []
for i in range(0, len(closep), STEP): 
    tmp_arr = []
    try:
        o = openp[i:i+WINDOW]
        h = highp[i:i+WINDOW]
        l = lowp[i:i+WINDOW]
        c = closep[i:i+WINDOW]
        v = volumep[i:i+WINDOW]

        for f in fundamentals:
            f = f[i:i+WINDOW]
            #print f
            #f = (np.array(f) - np.mean(f)) / np.std(f)
            tmp_arr.append(f)

        o = (np.array(o) - np.mean(o)) / np.std(o)
        h = (np.array(h) - np.mean(h)) / np.std(h)
        l = (np.array(l) - np.mean(l)) / np.std(l)
        c = (np.array(c) - np.mean(c)) / np.std(c)
        v = (np.array(v) - np.mean(v)) / np.std(v)

        x_i = closep[i:i+WINDOW]
        y_i = closep[i+WINDOW+FORECAST]  

        last_close = x_i[-1]
        next_close = y_i

        if last_close < next_close:
            y_i = [1, 0]
        else:
            y_i = [0, 1] 

        tmp_arr.append(o)
        tmp_arr.append(h)
        tmp_arr.append(l)
        tmp_arr.append(c)
        tmp_arr.append(v)
        x_i = np.column_stack(tmp_arr)
        x_i = np.column_stack((o, h, l ,v, c))
    except Exception as e:
        print e
        break

    X.append(x_i)
    Y.append(y_i)


X, Y = np.array(X), np.array(Y)


X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y)
#EMB_SIZE = len(X_train[0])
print X_train.shape

#exit()


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], EMB_SIZE))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], EMB_SIZE))



run_creation = {"instrument": instrument, "window": window, "forecast": forecast, "start": datetime.datetime.utcnow(), "examples": len(openp), "history":[]}
id = run_collection.insert_one( run_creation )

history = LossHistory(id.inserted_id, run_collection)

model = Sequential()
model.add(Convolution1D(input_shape = (WINDOW, EMB_SIZE),
                        nb_filter=32,
                        filter_length=8,
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.25))

model.add(Convolution1D(nb_filter=16,
                        filter_length=4,
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


model.add(Dense(2))
model.add(Activation('softmax'))

opt = Nadam(lr=0.002)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=0)
checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=0, save_best_only=True)


model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, 
          nb_epoch = 100, 
          batch_size = 128, 
          verbose=2, 
          validation_data=(X_test, Y_test),
          callbacks=[reduce_lr, checkpointer, history],
          shuffle=True)

model.load_weights("lolkek.hdf5")
pred = model.predict(np.array(X_test))


run_collection.update_one({"_id": id.inserted_id}, {"$set": {"end": datetime.datetime.utcnow()}})
