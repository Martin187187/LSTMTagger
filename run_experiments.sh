#!/bin/bash

# Define the parameter values
SEED=42
BATCH_SIZES=(1)
DROPOUT_RATES=(0.0)
LEARNING_RATES=(0.000016 0.000032 0.000064 0.000128 0.000256 0.000512 0.001 0.002)

# Activate the virtual environment
source /path/to/your/env/bin/activate

# Log file
LOG_FILE="experiment_log.txt"
echo "Starting experiments..." > ${LOG_FILE}

# Run the script for each combination of parameters
for BATCH_SIZE in "${BATCH_SIZES[@]}"
do
    for DROPOUT_RATE in "${DROPOUT_RATES[@]}"
    do
        for LR in "${LEARNING_RATES[@]}"
        do
            echo "Running sequence_tagger.py with batch_size=${BATCH_SIZE}, dropout_rate=${DROPOUT_RATE}, learning_rate=${LR}" | tee -a ${LOG_FILE}
            python -u sequence_tagger.py --seed ${SEED} --batch_size ${BATCH_SIZE} --dropout_rate ${DROPOUT_RATE} --learning_rate ${LR} | tee -a ${LOG_FILE}
            wait
        done
    done
done

# Deactivate the virtual environment
deactivate
