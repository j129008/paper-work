#!/bin/bash
RAW="./data/data_lite_RAW.txt"
PROC="$RAW.proc"

echo '文字前處理'
cd 文字前處理/
./text_preproc.sh $RAW $PROC
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
./lstm_s2s.sh $PROC
./lstm_stack.sh $PROC
cd ..

echo 'CRF+LSTM的最佳整合'
cd CRF+LSTM的最佳整合/
./avg_ensemble.sh $PROC
cd ..
# 詞表修正/
