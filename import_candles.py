import pymongo
from pymongo import MongoClient
import csv
import urllib2
import sys
import os.path
from datetime import datetime, timedelta
from dateutil import parser


QUANDL_API_KEY = "NxJJy-fpqF-GV7wwypzs"
# Are we using development or Production ?

# uri = "mongodb://USERNAME:PASSWORD@INSTANCE.documents.azure.com:10255/?ssl=true&replicaSet=globaldb"
uri = 'mongodb://localhost:27017'

# Make sure unique index has been created  
# db.candles.createIndex({ date: 1, ticker: 1 }, { unique: true })

client = MongoClient(uri)

db = client.fluid
candles = db.candles

# We don't need all candles from IPO to now - I'll start with the past few months


# Get the list of tickers from the companies collection
#companies = db.companies.find({"Exchange": "AMEX"})

url = 'https://www.quandl.com/api/v3/datatables/WIKI/PRICES.csv?api_key=' + QUANDL_API_KEY + "&ticker=" + sys.argv[1]
csvfile = urllib2.urlopen(url)
reader = csv.DictReader(csvfile, delimiter=',')

for line in reader:
    sys.stdout.write('.')
    try:
        line['date'] = parser.parse(line["date"])

        candles.insert_one(line)
    except Exception as e:
        continue

sys.stdout.write('\n')
