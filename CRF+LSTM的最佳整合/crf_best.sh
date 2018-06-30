#!/bin/bash

# -i: 斷句語料路徑 -k: 前後文範圍, -ts: 讀入資料完全用來訓練, -tdiff, -pmi: 使用斷詞特徵, -smod: 模型儲存路徑
time python3 ./crf_basic.py -i $1 -k 5 -ts 1.0 -tdiff -pmi -smod ./pickles/crf.pkl
