#!/bin/bash

# -i: 參考語料路徑, -o: 斷句語料路徑, -hold: 保留符號
python3 text_preproc.py -i $1 -o $2
python3 text_preproc.py -i $1 -o ./data/w2v.txt --hold 。
