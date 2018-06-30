#!/bin/bash
echo 'basic s2s'
time python3 ./lstm_s2s.py -i $1 -k 10 -ts 0.7
echo 'coder s2s'
time python3 ./lstm_s2s.py -i $1 -k 10 -ts 0.7 -stack 0 -enc 3 -dec 2
