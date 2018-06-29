#!/bin/bash
echo 'pmi'
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -pmi

echo 'tdiff'
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -tdiff

echo '反切'
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -rhy 反切

echo '聲母'
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -rhy 聲母

echo '韻目'
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -rhy 韻目

echo '調'
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -rhy 調

echo '等'
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -rhy 等

echo '呼'
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -rhy 呼

echo '韻母'
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -rhy 韻母

echo 'office'
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -list office:./ref/tang_name/tangOffice.clliu.txt

echo 'address'
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -list address:./ref/tang_name/tangAddresses.clliu.txt

echo 'nianhao'
time python3 ./crf_basic.py -i $1 -k 5 -ts 0.7 -list nianhao:./ref/tang_name/tangReignperiods.clliu.txt
