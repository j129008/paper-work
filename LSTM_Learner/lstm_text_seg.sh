#!/bin/bash
python3 lstm_text_seg.py -i $1 -smod $2 -k 10 > $3
