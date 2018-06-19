#!/bin/bash
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.3 -pmi
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.3 -tdiff
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.3 -pmi -noise
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.3 -tdiff -noise
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.3 -rhy 反切
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.3 -rhy 聲母
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.3 -rhy 韻目
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.3 -rhy 調
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.3 -rhy 等
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.3 -rhy 呼
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.3 -rhy 韻母
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.3 -list office:./ref/tang_name/tangOffice.clliu.txt
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.3 -list address:./ref/tang_name/tangAddresses.clliu.txt
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.3 -list nianhao:./ref/tang_name/tangReignperiods.clliu.txt
