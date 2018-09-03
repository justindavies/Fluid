import pymongo
from pymongo import MongoClient
import re
import json


# uri = "mongodb://USERNAME:PASSWORD@INSTANCE.documents.azure.com:10255/?ssl=true&replicaSet=globaldb"
uri = 'mongodb://localhost:27017'

# Make sure unique index has been created  
# db.candles.createIndex({ date: 1, ticker: 1 }, { unique: true })

client = MongoClient(uri)

db = client.fluid
companies = db.companies

# Get the list of tickers from the companies collection
companies = db.companies.find({"Exchange": "NYSE"})

## format for LUIS JSON is:
# [
#  {
#	"canonicalForm": "Egypt",
#	"list": [
#	  "Cairo",
#	  "Alexandria"
#	]
#  }
# ]



luis_companies = []




for company in companies:
    # Only add if we have candles
    if db.candles.find({"ticker": company['Symbol']}).count() > 0:
        # Clean up the company name
        clean_company_name = re.sub(', Inc\.$', '', company['Name'])
        clean_company_name = re.sub(' Inc\.$', '', clean_company_name)
        clean_company_name = re.sub(' Inc$', '', clean_company_name)
        clean_company_name = re.sub(', Ltd\.$', '', clean_company_name)
        clean_company_name = re.sub(' \(The\)$', '', clean_company_name)

        if clean_company_name == company['Name']:        
           luis_companies.append({"canonicalForm": company['Symbol'], "list": [company['Name']]})    
        else:
           luis_companies.append({"canonicalForm": company['Symbol'], "list": [company['Name'], clean_company_name]})    
            
print json.dumps(luis_companies)

    