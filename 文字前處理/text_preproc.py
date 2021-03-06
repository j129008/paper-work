import re
from argparse import ArgumentParser

class TextPreproc:
    def __init__(self, input_path='./data/data.csv', output_path='./data/data_proc.txt', hold=None):
        f = open(input_path, 'r')
        f_proc = open(output_path, 'w')
        bracket = ['「', '」', '『', '』', '“', '”', '‘', '’', '《', '》']
        pause_punc = ['，', '。', '；','：', '！', '？', '、']
        if hold != None:
            pause_punc.remove(hold)

        for chap in f:
            chap_proc = chap.strip()
            for b in bracket:
                chap_proc = chap_proc.replace(b, '')
            chap_proc = re.sub(r'[A-z]', '', chap_proc)
            chap_proc = re.sub(r'[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ]', '', chap_proc)
            chap_proc = re.sub(r'[０１２３４５６７８９]', '', chap_proc)
            chap_proc = re.sub(r'[■╮●△↓Δ＊　]', '', chap_proc)
            chap_proc = re.sub(r'[\x00-\x7F]', '', chap_proc)
            chap_proc = re.sub(r'（[^）]*）', '', chap_proc)
            chap_proc = re.sub(r'【[^】]*】', '', chap_proc)
            chap_proc = re.sub(r'〔[^〕]*〕', '', chap_proc)
            chap_proc = re.sub(r'［[^］]*］', '', chap_proc)
            chap_proc = re.sub(r'〈[^〉]*〉', '', chap_proc)
            chap_proc = re.sub(r'^，', '', chap_proc)
            if len(chap_proc) < 30:
                continue
            if chap_proc[-1] not in pause_punc + [hold]:
                chap_proc += '，'
            for p in pause_punc:
                chap_proc = chap_proc.replace(p, '，')
            f_proc.write(chap_proc+'\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', dest='input', help='input file')
    parser.add_argument('-o', dest='output', help='output file')
    parser.add_argument('--hold', dest='hold', default=None, help='hold punctuation')
    args = parser.parse_args()
    TextPreproc(input_path=args.input, output_path=args.output, hold=args.hold)
