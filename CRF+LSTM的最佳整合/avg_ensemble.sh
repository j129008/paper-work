#!/bin/bash
time python3 ./lstm+crf_avg_ensemble.py -train $1 -test $2 -pred $3  -lk 10 -ck 5 -cmod ./pickles/crf.pkl -lmod ./pickles/lstm.h5 -cpmi -ctdiff
