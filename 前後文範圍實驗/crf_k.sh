#!/bin/bash
END=10
for i in $(seq 1 $END);
do
   echo 'crf context k:' $i
   time python3 ./crf_basic.py -i $1 -k $i -ts 0.7
done
