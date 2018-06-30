#!/bin/bash

# -k: 前後文範圍, -ts: 訓練資料比例, -i: 輸入斷句資料路徑
echo 'basic s2s'
time python3 ./lstm_s2s.py -i $1 -k 10 -ts 0.7

# -stack: lstm stack 層數, -enc: encoder 層數, -dec: decoder 層數
echo 'coder s2s'
time python3 ./lstm_s2s.py -i $1 -k 10 -ts 0.7 -stack 0 -enc 3 -dec 2
