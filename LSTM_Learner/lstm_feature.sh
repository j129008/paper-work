#!/bin/bash
# time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.7 -pmi
# time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.7 -tdiff
# time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.7 -pmi -noise
# time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.7 -tdiff -noise
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.7 -rhy 聲母
# time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.7 -rhy 韻目
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.7 -rhy 調
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.7 -rhy 等
time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.7 -rhy 呼
# time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.7 -rhy 韻母
# time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.7 -list office:./ref/tang_name/tangOffice.clliu.txt
# time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.7 -list address:./ref/tang_name/tangAddresses.clliu.txt
# time python3 ./lstm_feature.py -i $1 -k 10 -ts 0.7 -list nianhao:./ref/tang_name/tangReignperiods.clliu.txt
