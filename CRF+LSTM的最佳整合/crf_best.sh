#!/bin/bash
time python3 ./crf_basic.py -i $1 -k 5 -ts 1.0 -tdiff -pmi -smod ./pickles/crf.pkl
