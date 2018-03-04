import re

class TextPreproc:
    def __init__(self, input_path='./data/data.csv', output_path='./data/data_proc.txt'):
        f = open(input_path, 'r')
        f_proc = open(output_path, 'w')
        bracket = ['「', '」', '『', '』', '“', '”', '‘', '’', '《', '》']
        pause_punc = ['，', '。', '；','：', '！', '？', '、']

        for chap in f:
            chap_proc = chap.strip()
            for b in bracket:
                chap_proc = chap_proc.replace(b, '')
            chap_proc = re.sub(r'[A-z]', '', chap_proc)
            chap_proc = re.sub(r'[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ]', '', chap_proc)
            chap_proc = re.sub(r'[０１２３４５６７８９]', '', chap_proc)
            chap_proc = re.sub(r'[■□╮●△↓Δ＊　]', '', chap_proc)
            chap_proc = re.sub(r'[\x00-\x7F]', '', chap_proc)
            chap_proc = re.sub(r'（[^）]*）', '', chap_proc)
            chap_proc = re.sub(r'【[^】]*】', '', chap_proc)
            if len(chap_proc) < 30:
                continue
            if chap_proc[-1] not in pause_punc:
                chap_proc += '，'
            for p in pause_punc:
                chap_proc = chap_proc.replace(p, '，')
            f_proc.write(chap_proc+'\n')
