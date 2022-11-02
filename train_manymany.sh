#!/bin/bash
CONFIGS=('klue_roberta_large_e_df' 'klue_roberta_large_nof')

config_length=${#CONFIGS[@]}


for (( i=0; i<${config_length}; i++ ))
do

    echo ${CONFIGS[$i]}
    python3 rob_large_train.py \
        --config ${CONFIGS[$i]}
done