#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
if [[ $4 == *"--"* ]]; 
then
    $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
        $(dirname "$0")/test_noc.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
else
    PORT=$4
    $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port $PORT \
        $(dirname "$0")/test_noc.py $CONFIG $CHECKPOINT --launcher pytorch ${@:5}
fi