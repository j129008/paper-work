#!/bin/bash

# -i: 斷句語料路徑, -k: 前後文範圍, -vec: 字嵌入向量維度
time python3 ./lstm_basic.py -i $1 -k 10 -ts 0.7 -vec 10
time python3 ./lstm_basic.py -i $1 -k 10 -ts 0.7 -vec 50
time python3 ./lstm_basic.py -i $1 -k 10 -ts 0.7 -vec 100
