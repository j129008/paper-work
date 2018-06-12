#!/bin/bash
time python3 ./lstm_basic.py -i ./data/data_lite.txt -k 10 -ts 0.7 -vec 10
time python3 ./lstm_basic.py -i ./data/data_lite.txt -k 10 -ts 0.7 -vec 50
time python3 ./lstm_basic.py -i ./data/data_lite.txt -k 10 -ts 0.7 -vec 100
