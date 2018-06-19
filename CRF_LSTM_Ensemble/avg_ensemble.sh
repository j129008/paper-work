#!/bin/bash
time python3 ./lstm+crf_avg_ensemble.py -i $1 -lk 10 -ck 5 -lmod $2 -cmod $3 -cpmi -ctdiff
