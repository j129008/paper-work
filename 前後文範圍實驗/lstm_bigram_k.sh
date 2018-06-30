#!/bin/bash

# END: 前後文範圍 1~10, -ts: 訓練資料比例, -i: 斷句語料路徑
END=10
for i in $(seq 1 $END);
do
   echo 'lstm bigram context k:' $i
   time python3 ./lstm_bigram.py -i $1 -k $i -ts 0.7
done
