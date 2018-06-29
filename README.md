# 唐代墓誌銘與中國佛教寺廟志斷句程式使用說明
## 環境初始設定
config.sh

## 文字前處理
### 統一符號
text_preproc.sh
text_preproc.py

### 資料分割
text_split.sh
text_split.py

## 斷句模型選擇
other_models.sh
other_models.py

## 前後文範圍實驗
### CRF實驗
crf_k.sh
crf_learner.py
### LSTM實驗
lstm_k.sh
lstm_basic.py

## 輔助特徵選擇
### CRF實驗
crf_feature.sh
crf_learner.py
### LSTM實驗
lstm_feature.sh
lstm_feature.py

## 模型的資料量需求
### CRF實驗
crf_data.sh
crf_learner.py
### LSTM實驗
lstm_data.sh
lstm_basic.py

## CRF的整合學習
### CRF-Bagging實驗
bagging_k.sh
crf_bagging.py
### CRF-Boost實驗
boost_k.sh
crf_boost.py

## LSTM的模型結構
### LSTM stack實驗
lstm_stack.sh
lstm_basic.py
### sequence to sequence實驗
lstm_s2s.sh
lstm_s2s.py

## CRF+LSTM的最佳整合
lstm+crf_avg_ensemble.py

## 數據集、函式庫
### data
#### 唐代墓誌銘語料
epitaph_RAW.txt  : 唐代墓誌銘原始語料
tang_epitaph.txt : 唐代墓誌銘斷句語料
#### 中國佛教寺廟志語料
buddhist_info.txt     : 中國佛教寺廟志原始語料
buddhist_info_RAW.txt : 中國佛教寺廟志斷句語料
#### 字嵌入語料
w2v.txt
#### 測試語料
data_lite_RAW.txt
data_lite.txt
test_lite.txt
#### 小韻表
rhyme.txt
### lib
arg.py              : 讀取程式參數
crawer.py           : 抓取「韻典網」的小韻表資訊
crf.py              : 產生 CRF 模型
data.py             : 產生斷句語料與文字分類
ensumble_learner.py : 產生 CRF 整合學習模型
feature.py          : 產生特徵資料
learner.py          : 產生訓練模型
lstmlib.py          : 產生 LSTM 模型
### pickles
bagging.pkl    : CRF-Bagging 模型檔案
boost.pkl      : CRF-Boost 模型檔案
crf.pkl        : CRF 模型檔案
lstm.h5        : LSTM 模型檔案
s2s.h5         : sequence to sequence 模型檔案
rhyme_list.pkl : 小韻表 python 資料庫物件
word2vec.pkl   : 字嵌入模型檔案
### ref
tang_name : 唐代墓誌銘官職、地名、年號表
