#!/bin/bash
END=6
# END: 前後文範圍測試 1~6
# -i: 輸入斷句語料路徑
# -ts: 訓練資料比例
for i in $(seq 1 $END);
do
   echo 'boost context k:' $i
   time python3 ./crf_boost.py -i $1 -k $i -ts 0.7
done
