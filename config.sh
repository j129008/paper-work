#!/bin/bash
function linker {
   cd $1
   pwd
   ln -sf ../data
   ln -sf ../lib
   ln -sf ../ref
   ln -sf ../pickles
   cd ..
}

linker CRF_Learner
linker LSTM_Learner
linker CRF_LSTM_Ensemble
linker Text2Feature
linker Text_Classify
linker Text_Preproc
linker Other_Model
linker RE_Fix
