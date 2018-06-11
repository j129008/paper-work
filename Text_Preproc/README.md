# 標點處理

## 指令
python3 text_preproc.py -i [input file] -o [output file] --hold [hold punctuation]

## 非斷句符號
```
bracket = ['「', '」', '『', '』', '“', '”', '‘', '’', '《', '》']
```

## 斷句符號
```
pause_punc = ['，', '。', '；','：', '！', '？', '、']
```

## 去掉英文、注音、全形數字等文言文不會出現的字元
```
chap_proc = re.sub(r'[A-z]', '', chap_proc)
chap_proc = re.sub(r'[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ]', '', chap_proc)
chap_proc = re.sub(r'[０１２３４５６７８９]', '', chap_proc)
chap_proc = re.sub(r'[■□╮●△↓Δ＊　]', '', chap_proc)
chap_proc = re.sub(r'[\x00-\x7F]', '', chap_proc)
chap_proc = re.sub(r'（[^）]*）', '', chap_proc)
chap_proc = re.sub(r'【[^】]*】', '', chap_proc)
chap_proc = re.sub(r'〔[^〕]*〕', '', chap_proc)
chap_proc = re.sub(r'［[^］]*］', '', chap_proc)
chap_proc = re.sub(r'〈[^〉]*〉', '', chap_proc)
chap_proc = re.sub(r'^，', '', chap_proc)
```
