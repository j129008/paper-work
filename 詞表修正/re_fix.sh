#!/bin/bash

# -i: 自動斷句結果語料, -o: 詞表修正輸出語料, -ans: 人工標記的參考語料, -voc: 詞表位置
python3 re_fix.py -i $1 -o $2 -ans $3 -voc $4
