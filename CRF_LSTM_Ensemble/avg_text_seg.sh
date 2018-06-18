#!/bin/bash
python3 avg_text_seg.py -i ./data/data_lite.txt -cmod ./pickles/crf.pkl -lmod ./pickles/lstm.h5 -lk 10 -ck 5 > avg_seg.txt
