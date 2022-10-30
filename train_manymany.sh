#!/bin/bash
CONFIGS=("roberta_back05_config" "roberta_back033_config" "roberta_back025_config" "roberta_back066_config" "roberta_back075_config")

config_length=${#CONFIGS[@]}


for (( i=0; i<${config_length}; i++ ))
do

    echo ${CONFIGS[$i]}
    python3 train_y.py \
        --config ${CONFIGS[$i]}
done