#!/bin/bash
python3 avg_text_seg.py -i $1 -cmod $2 -lmod $3 -lk 10 -ck 5 -cpmi -ctdiff > $4
