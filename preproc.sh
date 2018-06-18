#!/bin/bash
INPUT=./data/epitaph_RAW.txt
OUTPUT=./data/tang_epitaph.txt

cd ./Text_Preproc
echo 'text preproc'
./text_preproc.sh $INPUT $OUTPUT
cd ..
