#!/bin/bash
END=10
for i in $(seq 1 $END);
do
   echo 'crf context k:' $i
   time python3 ./crf_learner.py -i ./data/data_lite.txt -k $i -ts 0.7
done
