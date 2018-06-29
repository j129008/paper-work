#!/bin/bash
time python3 ./lstm+crf_avg_ensemble.py -i $1 -lk 10 -ck 5 -cmod ./pickles/crf.pkl -lmod ./pickles/lstm.h5 -cpmi -ctdiff
