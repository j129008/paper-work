#!/bin/bash
INPUT=./data/data_lite.txt
W2V=./data/w2v.txt

cd ./Text_Preproc
echo 'text preproc'
./text_preproc.sh $INPUT $W2V
cd ..

cd ./CRF_Learner
echo 'crf data need'
./crf_data.sh $INPUT > crf_data.txt
echo 'crf context'
./crf_k.sh $INPUT > crf_k.txt
echo 'crf feature'
./crf_feature.sh $INPUT > crf_feature.txt
echo 'crf bagging'
./bagging_k.sh $INPUT > bagging_k.txt
echo 'crf boosting'
./boost_k.sh $INPUT > boost_k.txt
cd ..
