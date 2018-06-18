#!/bin/bash
python3 text_preproc.py -i ./data/epitaph_RAW.txt -o ./data/tang_epitaph.txt
python3 text_preproc.py -i ./data/epitaph_RAW.txt -o ./data/w2v.txt --hold 。
