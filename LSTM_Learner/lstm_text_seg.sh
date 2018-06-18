#!/bin/bash
python3 lstm_text_seg.py -i ./data/data_lite.txt -smod ./pickles/lstm.h5 -k 10 > seg.txt
