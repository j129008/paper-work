#!/bin/bash

# END: 前後文範圍測試 1~6
# -i: 輸入斷句語料路徑
# -ts: 訓練資料比例
END=6
for i in $(seq 1 $END);
do
   echo 'bagging context k:' $i
   time python3 ./crf_bagging.py -i $1 -k $i -ts 0.7
done
