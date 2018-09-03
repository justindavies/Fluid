import pymongo
from pymongo import MongoClient
import urllib2
import csv


# Are we using development or Production ?

# uri = "mongodb://USERNAME:PASSWORD@INSTANCE.documents.azure.com:10255/?ssl=true&replicaSet=globaldb"

uri = 'mongodb://localhost:27017'

# Connect to the instance and get a handle
client = MongoClient(uri)
db = client.fluid
companies = db.companies

# Let's pull the NASDAQ data directly

#url = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download' 
#url = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NYSE&render=download'
url = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=AMEX&render=download'
csvfile = urllib2.urlopen(url)

# Read the CSV into a dict so we can pipe it into DB
reader = csv.DictReader(csvfile, delimiter=',')

# And import - I'm not too worried about batching these as it's only a few thousand entries
for line in reader:
    line['Exchange'] = "AMEX"
    print("Inserting " + line['Name'])
    companies.insert_one(line)


