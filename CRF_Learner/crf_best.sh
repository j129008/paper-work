#!/bin/bash
time python3 ./crf_learner.py -i ./data/data_lite.txt -k 5 -ts 1.0 -tdiff -pmi -smod ./pickles/crf.pkl
