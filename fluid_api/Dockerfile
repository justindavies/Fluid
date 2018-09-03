from python:2
run pip install numpy
run apt-get update
run apt-get -y install build-essential  
run  wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
run tar zxvf ta-lib-0.4.0-src.tar.gz
run cd ta-lib && ./configure && make && make install
run pip install TA-Lib
env LD_LIBRARY_PATH /usr/local/lib
run pip install ta-lib 
run pip install flask
run pip install pymongo
run pip install matplotlib
COPY app.py /
CMD python /app.py


