#!/bin/bash
CONFIGS=("monologg_edf_16_1 copy")

config_length=${#CONFIGS[@]}


for (( i=0; i<${config_length}; i++ ));
do

    echo ${CONFIGS[$i]}
    python3 train_y.py \
        --config ${CONFIGS[$i]}
done