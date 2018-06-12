#!/bin/bash
time python3 ./lstm_feature.py -i ./data/data_lite.txt -k 10 -ts 0.3 -pmi
time python3 ./lstm_feature.py -i ./data/data_lite.txt -k 10 -ts 0.3 -tdiff
time python3 ./lstm_feature.py -i ./data/data_lite.txt -k 10 -ts 0.3 -list office:./ref/tang_name/tangOffice.clliu.txt
time python3 ./lstm_feature.py -i ./data/data_lite.txt -k 10 -ts 0.3 -list address:./ref/tang_name/tangAddresses.clliu.txt
time python3 ./lstm_feature.py -i ./data/data_lite.txt -k 10 -ts 0.3 -list nianhao:./ref/tang_name/tangReignperiods.clliu.txt
