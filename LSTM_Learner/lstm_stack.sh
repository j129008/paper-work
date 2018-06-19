#!/bin/bash
END=10
for i in $(seq 1 $END);
do
   echo 'lstm stack :' $i
   time python3 ./lstm_basic.py -i $1 -k 10 -ts 0.7 -stack $i
done
