#!/bin/bash
time python3 ./crf_learner.py -i $1 -k 5 -ts 1.0 -tdiff -pmi -smod $2
