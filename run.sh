#!/bin/bash
INPUT=./data/data_lite.txt
W2V=./data/w2v.txt

cd ./Text_Preproc
./text_preproc.sh $INPUT $W2V
cd ..

cd ./CRF_Learner
./crf_data.sh $INPUT > crf_data.txt
./crf_k.sh $INPUT > crf_k.txt
./crf_feature.sh $INPUT > crf_feature.txt
./bagging_k.sh $INPUT > bagging_k.txt
./boost_k.sh $INPUT > boost_k.txt
cd ..
