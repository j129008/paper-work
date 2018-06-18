#!/bin/bash
python3 text_preproc.py -i $1 -o $2
python3 text_preproc.py -i $1 -o ./data/w2v.txt --hold 。
