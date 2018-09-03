from flask import Flask, make_response
from pymongo import MongoClient
import numpy as np
import talib
from talib import abstract
from talib.abstract import *
import json
import datetime
import numpy as np
import StringIO
from flask import request

import pymongo
import random
import os


# Make sure to set the environment variable MONGODB to be the endpoint for the Fluid DB
#if "MONGODB" not in os.environ:
#    print "Please set MONGODB environment variable to point to the Fluid DB"
#    exit()

#client = MongoClient(os.environ['MONGODB'])
client = MongoClient("mongodb://localhost:27017")


db = client.fluid

candles = db.historical_prices
crypto_candles = db.historical_crypto_prices
companies = db.companies
crypto = db.cryptos

app = Flask(__name__)

def get_candles(instrument, limit=0):

    start_date = datetime.datetime.now() + datetime.timedelta(-30)

    c_high = []
    c_open = [] 
    c_close = []
    c_low = []
    c_volume = []
    c_date = []
 
#    for candle in candles.find({"ticker": instrument, "date": {'$gte': start_date} }):
    for candle in candles.find({"symbol": instrument}).sort([("date", pymongo.DESCENDING)]).limit(20):

	try:
        	c_open.insert(0, float(candle["open"])) 
        	c_high.insert(0, float(candle["high"])) 
        	c_low.insert(0, float(candle["low"])) 
        	c_close.insert(0, float(candle["close"])) 
        	c_volume.insert(0, float(candle["volume"])) 
        	c_date.insert(0, candle["date"]) 

	except Exception as e:
		print e
		continue


    inputs = {
        'open': np.asarray(c_open),
        'high': np.asarray(c_high),
        'low': np.asarray(c_low),
        'close': np.asarray(c_close),
        'volume': np.asarray(c_volume)
    }


    return inputs



def get_crypto_candles(instrument, limit=0):
    start_date = datetime.datetime.now() + datetime.timedelta(-30)

    c_high = []
    c_open = []
    c_close = []
    c_low = []
    c_volume = []
    c_date = []
    inputs = {}

#    for candle in candles.find({"ticker": instrument, "date": {'$gte': start_date} }):
    for candle in crypto_candles.find({"symbol": instrument}).sort([("time", pymongo.DESCENDING)]).limit(20):
        try:
            c_open.insert(0, float(candle["open"])) 
            c_high.insert(0, float(candle["high"])) 
            c_low.insert(0, float(candle["low"])) 
            c_close.insert(0, float(candle["close"])) 
            c_volume.insert(0, float(candle["volumefrom"])) 
            c_date.insert(0, candle["time"]) 

        except Exception as e:
            print e
            continue


        inputs = {
            'open': np.asarray(c_open),
            'high': np.asarray(c_high),
            'low': np.asarray(c_low),
            'close': np.asarray(c_close),
            'volume': np.asarray(c_volume)
        }


    return inputs


@app.route('/instrument/<instrument>/history')
def instrument_history(instrument):
    candles = get_candles(instrument)

    return json.dumps(candles['close'].tolist()[-10:])

@app.route('/instrument/<instrument>/SMA')
def calculate_sma(instrument):
    candles = get_candles(instrument)
    return json.dumps(SMA(candles).tolist()[-10:])

@app.route('/instrument/<instrument>/MACD')
def calculate_macd(instrument):
    candles = get_candles(instrument)
    macd, macdsignal, macdhist = MACD(candles)
    return json.dumps(macd.tolist()[-10:])


@app.route('/instrument/<instrument>/RSI')
def calculate_rsi(instrument):
    if request.args.get('crypto') == "1":
        candles = get_crypto_candles(instrument, 30)
    else:
        candles = get_candles(instrument, 30)

    rsi = RSI(candles, timeperiod=14)
    return json.dumps(rsi.tolist()[-10:])

@app.route('/instrument/<instrument>/patterns')
def get_patterns(instrument):

    if request.args.get('crypto') == "1":
        candles = get_crypto_candles(instrument, 30)
    else:
        candles = get_candles(instrument, 30)

    if len(candles) == 0:
        return []

    ta_funcs =  talib.get_function_groups()

    patterns = []

    human_readable = {"CDL2CROWS": "Two Crows",
    "CDL3BLACKCROWS":       "Three Black Crows",
    "CDL3INSIDE":           "Three Inside Up/Down",
    "CDL3LINESTRIKE":       "Three-Line Strike",
    "CDL3OUTSIDE":          "Three Outside Up/Down",
    "CDL3STARSINSOUTH":     "Three Stars In The South",
    "CDL3WHITESOLDIERS":    "Three Advancing White Soldiers",
    "CDLABANDONEDBABY":     "Abandoned Baby",
    "CDLADVANCEBLOCK":      "Advance Block",
    "CDLBELTHOLD":          "Belt-hold",
    "CDLBREAKAWAY":         "Breakaway",
    "CDLCLOSINGMARUBOZU":   "Closing Marubozu",
    "CDLCONCEALBABYSWALL":  "Concealing Baby Swallow",
    "CDLCOUNTERATTACK":     "Counterattack",
    "CDLDARKCLOUDCOVER":    "Dark Cloud Cover",
    "CDLDOJI":              "Doji",
    "CDLDOJISTAR":          "Doji Star",
    "CDLDRAGONFLYDOJI":     "Dragonfly Doji",
    "CDLENGULFING":         "Engulfing Pattern",
    "CDLEVENINGDOJISTAR":   "Evening Doji Star",
    "CDLEVENINGSTAR":       "Evening Star",
    "CDLGAPSIDESIDEWHITE":  "Up/Down-gap side-by-side white lines",
    "CDLGRAVESTONEDOJI":    "Gravestone Doji",
    "CDLHAMMER":            "Hammer",
    "CDLHANGINGMAN":        "Hanging Man",
    "CDLHARAMI":            "Harami Pattern",
    "CDLHARAMICROSS":       "Harami Cross Pattern",
    "CDLHIGHWAVE":          "High-Wave Candle",
    "CDLHIKKAKE":           "Hikkake Pattern",
    "CDLHIKKAKEMOD":        "Modified Hikkake Pattern",
    "CDLHOMINGPIGEON":      "Homing Pigeon",
    "CDLIDENTICAL3CROWS":   "Identical Three Crows",
    "CDLINNECK":            "In-Neck Pattern",
    "CDLINVERTEDHAMMER":    "Inverted Hammer",
    "CDLKICKING":           "Kicking",
    "CDLKICKINGBYLENGTH":   "Kicking - bull/bear determined by the longer marubozu",
    "CDLLADDERBOTTOM":      "Ladder Bottom",
    "CDLLONGLEGGEDDOJI":    "Long Legged Doji",
    "CDLLONGLINE":          "Long Line Candle",
    "CDLMARUBOZU":          "Marubozu",
    "CDLMATCHINGLOW":       "Matching Low",
    "CDLMATHOLD":           "Mat Hold",
    "CDLMORNINGDOJISTAR":   "Morning Doji Star",
    "CDLMORNINGSTAR":       "Morning Star",
    "CDLONNECK":            "On-Neck Pattern",
    "CDLPIERCING":          "Piercing Pattern",
    "CDLRICKSHAWMAN":       "Rickshaw Man",
    "CDLRISEFALL3METHODS":  "Rising/Falling Three Methods",
    "CDLSEPARATINGLINES":   "Separating Lines",
    "CDLSHOOTINGSTAR":      "Shooting Star",
    "CDLSHORTLINE":         "Short Line Candle",
    "CDLSPINNINGTOP":       "Spinning Top",
    "CDLSTALLEDPATTERN":    "Stalled Pattern",
    "CDLSTICKSANDWICH":     "Stick Sandwich",
    "CDLTAKURI":            "Takuri (Dragonfly Doji with very long lower shadow)",
    "CDLTASUKIGAP":         "Tasuki Gap",
    "CDLTHRUSTING":         "Thrusting Pattern",
    "CDLTRISTAR":           "Tristar Pattern",
    "CDLUNIQUE3RIVER":      "Unique 3 River",
    "CDLUPSIDEGAP2CROWS":   "Upside Gap Two Crows",
    "CDLXSIDEGAP3METHODS":  "Upside/Downside Gap Three Methods"}

    for ta_func in ta_funcs['Pattern Recognition']:
        ret = getattr(talib.abstract, ta_func)(candles)

        if len(np.nonzero(ret[-1:])[0]) > 0:
            if ret[-1:][0] > 0:
                patterns.append("Bullish " + human_readable[ta_func])

            if ret[-1:][0] < 0:
                patterns.append("Bearish " + human_readable[ta_func])
        
    return json.dumps(patterns)


if __name__ == '__main__':
    app.run(debug=False, port=8080)
