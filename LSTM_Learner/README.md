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
