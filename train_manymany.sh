#!/bin/bash
CONFIGS=("electra_config_nof" "electra_config_16" "electra_config_16_nof" )

config_length=${#CONFIGS[@]}


for (( i=0; i<${config_length}; i++ ));
do

    echo ${CONFIGS[$i]}
    python3 train_y.py \
        --config ${CONFIGS[$i]}
done