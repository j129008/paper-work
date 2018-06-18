#!/bin/bash
END=6
for i in $(seq 1 $END);
do
   echo 'bagging context k:' $i
   time python3 ./crf_bagging.py -i $1 -k $i -ts 0.7
done
