#!/bin/bash
CONFIGS=('roberta_large' 'XLM_roberta' 'XLM_roberta1' 'XLM_roberta2')

config_length=${#CONFIGS[@]}


for (( i=0; i<${config_length}; i++ ))
do

    echo ${CONFIGS[$i]}
    python3 rob_large_train.py \
        --config ${CONFIGS[$i]}
done