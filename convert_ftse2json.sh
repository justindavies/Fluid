#!/bin/bash
#
# Read in a CSV of Company, Ticker, Industry and output to JSON for use in Luis.ai
# 

IFS=$'\n'

echo "["

for i in `cat ftse100.csv`; do
	COMPANY=`cut -f 1 -d "," <<< ${i}`
	TICKER=`cut -f 2 -d "," <<< ${i}`
  
	# Output entity
	echo "  {
	\"canonicalForm\": \"${COMPANY}\",
	\"list\": [
	  \"${TICKER}\"
	]
  },"
done

echo "]"
