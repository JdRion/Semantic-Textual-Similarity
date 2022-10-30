#!/bin/bash
CONFIGS=("base_config" "roberta-base-e_df" "roberta-base-e_df-pororo" "roberta-base-e_df-pororo21")

config_length=${#CONFIGS[@]}


for (( i=0; i<${config_length}; i++ ));
do

    echo ${CONFIGS[$i]}
    python3 train_y.py \
        --config ${CONFIGS[$i]}
done