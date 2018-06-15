## CRF options

### basic options
-i     : 輸入的文字檔案
-smod  : 儲存訓練完成的模型路徑
-k     : 前後文範圍
-ts    : 訓練、測試資料的比例
-subtrain : 調整training data的大小比例

### CRF only options
-cv   : 進行K-fold的crossvalidation
-iter : 每次CV tune參數的次數
-c1   : L1 regularization 參數
-c2   : L2 regularization 參數

### feature options
-rhy   : [反切,聲母,韻目,調,等,呼,韻母]
-ngram : 使用ngram的長度, 1代表使用unigram, 2則是bigram, 3使用trigram
-pmi   : 加入pmi作為特徵
-tdiff : 加入tdiff作為特徵

### bagging
-seg : 將資料單位切割成更小塊
-voter : 投票者數量
-votersize : 投票者能得到的資料比例


## LSTM options

### basic options
-i     : 輸入的文字檔案
-smod  : 儲存訓練完成的模型路徑
-k     : 前後文範圍
-ts    : 訓練、測試資料的比例
-subtrain : 調整training data的大小比例

### LSTM options
-w2v   : w2v 的文字檔案路徑
-vec   : w2v 的向量長度
-val   : valid data 佔 training data 的比例
-stack : LSTM的模型層數
-plot  : 輸出模型架構圖

### feature options
-rhy   : [反切,聲母,韻目,調,等,呼,韻母]
-ngram : 使用ngram的長度, 1代表使用unigram, 2則是bigram, 2使用trigram
-pmi   : 加入pmi作為特徵
-tdiff : 加入tdiff作為特徵

## LSTM + CRF options
-i    : 輸入的文字檔案
-lk   : LSTM的前後文範圍
-ck   : CRF的前後文範圍
-lmod : 讀取LSTM的模型路徑
-cmod : 讀取CRF的模型路徑
-w2v  : w2v 的文字檔案路徑
-vec  : w2v 的向量長度
