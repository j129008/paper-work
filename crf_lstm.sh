#!/bin/bash
TRAIN=./data/train.txt
TEST=./data/test.txt
CRF_MOD=./pickles/crf.pkl
LSTM_MOD=./pickles/lstm.h5
SEG_TEXT=./seg.txt

cd ./CRF_Learner
./crf_best.sh $TRAIN $CRF_MOD
cd ..

cd ./LSTM_Learner
./lstm_best.sh $TRAIN $LSTM_MOD
cd ..

cd ./CRF_LSTM_Ensemble
./avg_ensemble.sh $TEST $CRF_MOD $LSTM_MOD
./avg_text_seg.sh $TEST $CRF_MOD $LSTM_MOD $SEG_TEXT
cd ..
