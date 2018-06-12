#!/bin/bash
time python3 ./lstm_s2s.py -i ./data/data_lite.txt -k 10 -ts 0.7
time python3 ./lstm_s2s.py -i ./data/data_lite.txt -k 10 -ts 0.7 -stack 5 -enc 5 -dec 5
