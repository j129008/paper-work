#!/bin/bash
function data_linker {
   cd $1
   ln -sf ../data
   ln -sf ../lib
   ln -sf ../ref
   ln -sf ../pickles
   cd ..
}

function basic_linker {
   cd $1
   ln -sf ../basic_learner/crf_basic.py .
   ln -sf ../basic_learner/lstm_basic.py .
   cd ..
}

data_linker '斷句模型選擇/'
data_linker '前後文範圍實驗/'
data_linker '輔助特徵選擇/'
data_linker '模型的資料量需求'
data_linker 'CRF的整合學習/'
data_linker 'LSTM的模型結構/'
data_linker 'CRF+LSTM的最佳整合/'
data_linker 'basic_learner'

basic_linker '斷句模型選擇/'
basic_linker '前後文範圍實驗/'
basic_linker '輔助特徵選擇/'
basic_linker '模型的資料量需求'
basic_linker 'LSTM的模型結構/'
basic_linker 'CRF+LSTM的最佳整合/'

rm ./LSTM的模型結構/crf_basic.py
