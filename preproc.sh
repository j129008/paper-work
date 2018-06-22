#!/bin/bash
RAW=./data/epitaph_RAW.txt
DATA=./data/tang_epitaph.txt
TRAIN=./data/train.txt
TEST=./data/test.txt
SPLIT=0.7

cd ./Text_Preproc
echo 'text preproc'
./text_preproc.sh $RAW $DATA
./text_split.sh $DATA $TRAIN $TEST $SPLIT
cd ..

echo 'remove word2vec.pkl'
rm -f ./pickles/word2vec.pkl
