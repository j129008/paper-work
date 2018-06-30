#!/bin/bash

# -i: 斷句語料路徑 -k: 前後文範圍, -ts: 讀入資料完全用來訓練, -smod: 模型儲存路徑
time python3 ./lstm_basic.py -i $1 -k 10 -ts 1.0 -smod ./pickles/lstm.h5
