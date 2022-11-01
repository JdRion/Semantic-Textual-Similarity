#!/bin/bash
CONFIGS=("electra_e_no_c_05_df_config" "electra_r02_e_no_c_05_df_config")

config_length=${#CONFIGS[@]}


for (( i=0; i<${config_length}; i++ ));
do

    echo ${CONFIGS[$i]}
    python3 train_y.py \
        --config ${CONFIGS[$i]}
done