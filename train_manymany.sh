#!/bin/bash
CONFIGS=("krelectra_nof_16_3_seed1")

config_length=${#CONFIGS[@]}


for (( i=0; i<${config_length}; i++ ));
do

    echo ${CONFIGS[$i]}
    python3 train_y.py \
        --config ${CONFIGS[$i]}
done