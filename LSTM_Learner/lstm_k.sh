#!/bin/bash
END=10
for i in $(seq 1 $END);
do
   echo 'lstm context k:' $i
   time python3 ./lstm_basic.py -i ./data/data_lite.txt -k $i -ts 0.7
done
