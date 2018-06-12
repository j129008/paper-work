#!/bin/bash
echo 'pmi'
time python3 ./crf_learner.py -i ./data/data_lite.txt -k 5 -ts 0.7 -pmi

echo 'tdiff'
time python3 ./crf_learner.py -i ./data/data_lite.txt -k 5 -ts 0.7 -tdiff

echo 'rhyme1'
time python3 ./crf_learner.py -i ./data/data_lite.txt -k 5 -ts 0.7 -rhy 反切

echo 'rhyme2'
time python3 ./crf_learner.py -i ./data/data_lite.txt -k 5 -ts 0.7 -rhy 聲母

echo 'rhyme3'
time python3 ./crf_learner.py -i ./data/data_lite.txt -k 5 -ts 0.7 -rhy 韻目

echo 'rhyme4'
time python3 ./crf_learner.py -i ./data/data_lite.txt -k 5 -ts 0.7 -rhy 調

echo 'rhyme5'
time python3 ./crf_learner.py -i ./data/data_lite.txt -k 5 -ts 0.7 -rhy 等

echo 'rhyme6'
time python3 ./crf_learner.py -i ./data/data_lite.txt -k 5 -ts 0.7 -rhy 呼

echo 'rhyme7'
time python3 ./crf_learner.py -i ./data/data_lite.txt -k 5 -ts 0.7 -rhy 韻母

echo 'office'
time python3 ./crf_learner.py -i ./data/data_lite.txt -k 5 -ts 0.7 -list office:./ref/tang_name/tangOffice.clliu.txt

echo 'address'
time python3 ./crf_learner.py -i ./data/data_lite.txt -k 5 -ts 0.7 -list address:./ref/tang_name/tangAddresses.clliu.txt

echo 'nianhao'
time python3 ./crf_learner.py -i ./data/data_lite.txt -k 5 -ts 0.7 -list nianhao:./ref/tang_name/tangReignperiods.clliu.txt
