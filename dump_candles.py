import pymongo
from pymongo import MongoClient
import csv
import urllib2
import sys
import os.path
from datetime import datetime, timedelta
from dateutil import parser



uri = 'mongodb://localhost:27017'

# Make sure unique index has been created  
# db.candles.createIndex({ date: 1, ticker: 1 }, { unique: true })

client = MongoClient(uri)

db = client.fluid
candles = db.candles
instrument = sys.argv[1]

cursor = candles.find({"ticker": instrument},{"_id": 0,"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}).sort('date', -1)

candles = []
counter = 0
row = []
target = 0

for candle in cursor:
    if counter > 1:
        if last_close < candle['close']:
            target = 1
        else:
            target = 0
        last_close = float(candle['close'])

    if counter < 1:
        last_close = float(candle['close'])

    candles.append([candle['open'], candle['high'], candle['low'], candle['close'], candle['volume'], target])
    
    counter = counter + 1

print candles[0]
      