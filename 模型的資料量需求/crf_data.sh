#!/bin/bash

# -k: 前後文範圍, -i: 斷句語料路徑, -ts: 訓練資料佔斷句語料比例, -subtrain: 使用部分訓練資料比例
echo 'data size :' 0.1
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -subtrain 0.1
echo 'data size :' 0.2
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -subtrain 0.2
echo 'data size :' 0.3
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -subtrain 0.3
echo 'data size :' 0.4
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -subtrain 0.4
echo 'data size :' 0.5
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -subtrain 0.5
echo 'data size :' 0.6
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -subtrain 0.6
echo 'data size :' 0.7
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -subtrain 0.7
echo 'data size :' 0.8
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -subtrain 0.8
echo 'data size :' 0.9
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -subtrain 0.9
echo 'data size :' 1.0
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -subtrain 1.0
