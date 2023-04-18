#!/bin/bash

TASK=UDSS
VERSION=v4
PATH_OUT=/home/data2/kbh/${TASK}/${VERSION}
MIC=4ch
ROOM=v1


# eval
VERSION=v4
PATH_OUT=/home/data2/kbh/${TASK}/${VERSION}/test
python3 ./src/mix_for_sep.py -o ${PATH_OUT} -c config/${TASK}/${VERSION}_test.yaml -d config/default.yaml -m  config/mic/${MIC}.yaml -r config/room/${ROOM}.yaml -n 5000

# 2023-04-14 : v4 train
PATH_OUT=/home/data2/kbh/${TASK}/${VERSION}/train
python3 ./src/mix_for_sep.py -o ${PATH_OUT} -c config/${TASK}/${VERSION}.yaml -d config/default.yaml -m  config/mic/${MIC}.yaml -r config/room/${ROOM}.yaml -n 100000

