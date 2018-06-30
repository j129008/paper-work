# 斷句程式使用說明
初次使用本程式須先執行「config.sh」，方式如下：

若是執行新實驗，可使用「clean.sh」清除舊實驗資料以避免資料混亂。

以下是資料夾及其對應的論文實驗章節，內容以及執行方法在下文分別介紹。

| 資料夾             | 章節 |
| ------------------ | ---- |
| 文字前處理         | 4.2  |
| 斷句模型選擇       | 7.1  |
| 前後文範圍實驗     | 7.2  |
| 輔助特徵選擇       | 7.3  |
| 模型的資料量需求   | 7.4  |
| CRF的整合學習      | 7.5  |
| LSTM的模型結構     | 7.6  |
| CRF+LSTM的最佳整合 | 7.7  |

| 資料夾        | 用途         |
| ------------- | ------------ |
| basic_learner | 基礎模型     |
| data          | 語料檔案     |
| lib           | 函式庫       |
| pickles       | 儲存訓練模型 |
| ref           | 名詞表       |


## 文字前處理
### 統一符號
```
./text_preproc.sh [ 原始語料路徑 ]
```
### 語料分割
```
text_split.sh
```

## 斷句模型選擇
```
./other_models.sh [ 斷句語料路徑 ]
```

## 前後文範圍實驗
### CRF實驗
```
./crf_k.sh [ 斷句語料路徑 ]
```
### LSTM實驗
```
./lstm_k.sh [ 斷句語料路徑 ]
```

## 輔助特徵選擇
### CRF實驗
```
./crf_feature.sh
```
### LSTM實驗
```
./lstm_feature.sh [ 斷句語料路徑 ]
```

## 模型的資料量需求
### CRF實驗
```
./crf_data.sh [ 斷句語料路徑 ]
```
### LSTM實驗
```
./lstm_data.sh [ 斷句語料路徑 ]
```

## CRF的整合學習
### CRF-Bagging實驗
```
./bagging_k.sh [ 斷句語料路徑 ]
```
### CRF-Boost實驗
```
./boost_k.sh [ 斷句語料路徑 ]
```

## LSTM的模型結構
### LSTM stack實驗
```
./lstm_stack.sh [ 斷句語料路徑 ]
```
### sequence to sequence實驗
```
./lstm_s2s.sh [ 斷句語料路徑 ]
```

## CRF+LSTM的最佳整合
```
./avg_ensemble.sh [ 斷句語料路徑 ]
```

## basic_learner
crf_basic.py  : CRF模型訓練工具
lstm_basic.py : LSTM模型訓練工具

## data
### 唐代墓誌銘語料
epitaph_RAW.txt  : 原始語料
### 中國佛教寺廟志語料
buddhist_info.txt     : 原始語料
### 測試語料
data_lite_RAW.txt
### 小韻表
rhyme.txt

## lib
arg.py              : 讀取程式參數
ensumble_learner.py : 產生 CRF 整合學習模型
crf.py              : 產生 CRF 模型
learner.py          : 訓練 CRF 模型
lstmlib.py          : 產生 LSTM 模型
data.py             : 產生斷句語料與文字分類
feature.py          : 產生特徵資料
crawer.py           : 抓取「韻典網」的小韻表資訊

## pickles
rhyme_list.pkl : 小韻表 python 資料庫物件

## ref
tang_name/                           : 唐代墓誌銘名詞表資料夾
tang_name/tangAddresses.clliu.txt    : 地名
tang_name/tangOffice.clliu.txt       : 官職
tang_name/tangReignperiods.clliu.txt : 年號
tang_name/all.txt                    : 合併以上三個名詞表的檔案

budd/                                : 佛典資料夾
budd/cjkve.txt                       : 佛典詞表1
budd/ddb.txt                         : 佛典詞表2
budd/all.txt                         : 合併以上兩個名詞表的檔案
