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

cd ./LSTM_Learner
echo 'lstm data need'
./lstm_data.sh $INPUT > lstm_data.txt
echo 'lstm context'
./lstm_k.sh $INPUT > lstm_k.txt
echo 'lstm feature'
./lstm_feature.sh $INPUT > lstm_feature.txt
echo 'lstm bigram'
./lstm_bigram_k.sh $INPUT > lstm_bigram_k.txt
echo 'lstm seq2seq'
./lstm_s2s.sh $INPUT > lstm_s2s.txt
echo 'lstm w2v size'
./w2v_size.sh $INPUT > w2v_size.txt
cd ..
