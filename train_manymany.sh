#!/bin/bash
CONFIGS=('xlm_roberta_with_aug')

config_length=${#CONFIGS[@]}


for (( i=0; i<${config_length}; i++ ))
do

    echo ${CONFIGS[$i]}
    python3 rob_large_train.py \
        --config ${CONFIGS[$i]}
done