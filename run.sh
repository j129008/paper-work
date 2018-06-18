#!/bin/bash
INPUT=./data/data_lite.txt
W2V=./data/w2v.txt

# cd ./Text_Preproc
# ./text_preproc.sh $INPUT $W2V
# cd ..

# cd ./CRF_Learner
# ./crf_data.sh $INPUT > crf_data.txt
# ./crf_k.sh $INPUT > crf_k.txt
# ./crf_feature.sh $INPUT > crf_feature.txt
# ./bagging_k.sh $INPUT > bagging_k.txt
# ./boost_k.sh $INPUT > boost_k.txt
# cd ..

cd ./LSTM_Learner
./lstm_k.sh $INPUT > lstm_k.txt
./lstm_data.sh $INPUT > lstm_data.txt
./lstm_feature.sh $INPUT > lstm_feature.txt
./lstm_bigram_k.sh $INPUT > lstm_bigram_k.txt
./lstm_s2s.sh $INPUT > lstm_s2s.txt
./w2v_size.sh $INPUT > w2v_size.txt
cd ..
