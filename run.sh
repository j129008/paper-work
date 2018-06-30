#!/bin/bash
RAW="./data/data_lite_RAW.txt"
PROC="$RAW.proc"
TRAIN="$PROC.train"
TEST="$PROC.test"
PRED="$PROC.pred"
RE="$PRED.re"
VOC=./ref/tang_name/tangAddresses.clliu.txt

echo '文字前處理'
cd 文字前處理/
./text_preproc.sh $RAW $PROC
./text_split.sh $PROC $TRAIN $TEST
cd ..

echo '斷句模型選擇'
cd 斷句模型選擇/
./other_models.sh $PROC
cd ..

echo '前後文範圍實驗'
cd 前後文範圍實驗/
./crf_k.sh $PROC
./lstm_k.sh $PROC
cd ..

echo '輔助特徵選擇'
cd 輔助特徵選擇/
./crf_feature.sh $PROC
./lstm_feature.sh $PROC
cd ..

echo '模型的資料量需求'
cd 模型的資料量需求/
./crf_data.sh $PROC
./lstm_data.sh $PROC
cd ..

echo 'CRF的整合學習'
cd CRF的整合學習/
./bagging_k.sh $PROC
./boost_k.sh $PROC
cd ..

echo 'LSTM的模型結構'
cd LSTM的模型結構/
./w2v_size.sh* $PROC
./lstm_s2s.sh $PROC
./lstm_stack.sh $PROC
cd ..

echo 'CRF+LSTM的最佳整合'
cd CRF+LSTM的最佳整合/
./crf_best.sh $TRAIN
./lstm_best.sh $TRAIN
./avg_ensemble.sh $TRAIN $TEST $PRED
cd ..

echo '詞表修正'
cd 詞表修正/
./re_fix.sh $PRED $RE $TEST $VOC
cd ..
