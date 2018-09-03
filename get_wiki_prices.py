import quandl

quandl.ApiConfig.api_key = 'NxJJy-fpqF-GV7wwypzs'
data = quandl.get_table('WIKI/PRICES', ticker='MSFT')

print data