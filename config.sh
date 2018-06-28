#!/bin/bash
function linker {
   cd $1
   pwd
   ln -sf ../data
   ln -sf ../lib
   ln -sf ../ref
   ln -sf ../pickles
   cd ..
}

linker 斷句模型選擇
linker 前後文範圍實驗
linker 輔助特徵選擇
linker 模型的資料量需求
linker CRF的整合學習
linker LSTM的模型結構
linker CRF+LSTM的最佳整合
