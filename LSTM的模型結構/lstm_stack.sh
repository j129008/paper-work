#!/bin/bash

# END: 層數從 1~10 測試, -ts: 訓練資料比例, -stack: lstm 層數設定
END=10
for i in $(seq 1 $END);
do
   echo 'lstm stack :' $i
   time python3 ./lstm_basic.py -i $1 -k 10 -ts 0.7 -stack $i
done
