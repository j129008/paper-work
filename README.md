# 斷句程式使用說明
本程式初次使用，須先執行「環境初始設定」，方式如下：
```
./config.sh
```

以下是資料夾及其對應論文實驗章節。

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
| data               | 4.3  |
| lib                |      |
| pickles            |      |
| ref                |      |


## 文字前處理
### 統一符號
text_preproc.py
```
./text_preproc.sh [ 原始語料路徑 ]
```
### 語料分割
text_split.sh
text_split.py

## 斷句模型選擇
other_models.py
```
./other_models.sh [ 斷句語料路徑 ]
```

## 前後文範圍實驗
### CRF實驗
crf_learner.py
```
./crf_k.sh [ 斷句語料路徑 ]
```
### LSTM實驗
lstm_basic.py
```
./lstm_k.sh [ 斷句語料路徑 ]
```

## 輔助特徵選擇
### CRF實驗
crf_learner.py
```
./crf_feature.sh
```

### LSTM實驗
lstm_feature.py
```
./lstm_feature.sh [ 斷句語料路徑 ]
```

## 模型的資料量需求
### CRF實驗
crf_learner.py
```
./crf_data.sh [ 斷句語料路徑 ]
```
### LSTM實驗
lstm_basic.py
```
./lstm_data.sh [ 斷句語料路徑 ]
```

## CRF的整合學習
### CRF-Bagging實驗
crf_bagging.py
```
./bagging_k.sh [ 斷句語料路徑 ]
```
### CRF-Boost實驗
crf_boost.py
```
./boost_k.sh [ 斷句語料路徑 ]
```

## LSTM的模型結構
### LSTM stack實驗
lstm_basic.py
```
./lstm_stack.sh [ 斷句語料路徑 ]
```
### sequence to sequence實驗
lstm_s2s.py
```
./lstm_s2s.sh [ 斷句語料路徑 ]
```

## CRF+LSTM的最佳整合
lstm+crf_avg_ensemble.py
```
./avg_ensemble.sh [ 斷句語料路徑 ]
```

## data
### 唐代墓誌銘語料
epitaph_RAW.txt  : 原始語料
tang_epitaph.txt : 斷句語料
### 中國佛教寺廟志語料
buddhist_info.txt     : 原始語料
buddhist_info_RAW.txt : 斷句語料
### 字嵌入語料
w2v.txt
### 測試語料
data_lite_RAW.txt
data_lite.txt
test_lite.txt
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
bagging.pkl    : CRF-Bagging 模型檔案
boost.pkl      : CRF-Boost 模型檔案
crf.pkl        : CRF 模型檔案
lstm.h5        : LSTM 模型檔案
s2s.h5         : sequence to sequence 模型檔案
rhyme_list.pkl : 小韻表 python 資料庫物件
word2vec.pkl   : 字嵌入模型檔案

## ref
tang_name                            : 唐代墓誌銘名詞表資料夾
tang_name/tangAddresses.clliu.txt    : 地名
tang_name/tangOffice.clliu.txt       : 官職
tang_name/tangReignperiods.clliu.txt : 年號
