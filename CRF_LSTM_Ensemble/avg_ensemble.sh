#!/bin/bash
time python3 ./lstm+crf_avg_ensemble.py -i ./data/test_lite.txt -lk 10 -ck 5 -lmod ./pickles/lstm.h5 -cmod ./pickles/crf.pkl
