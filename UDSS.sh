#!/bin/bash

TASK=UDSS
PATH_OUT=/home/data2/kbh/${TASK}/${VERSION}
MIC=4ch
ROOM=v1

# eval
VERSION=v4
PATH_OUT=/home/data2/kbh/${TASK}/${VERSION}/test
#python3 ./src/mix_for_sep.py -o ${PATH_OUT} -c config/${TASK}/${VERSION}_test.yaml -d config/default.yaml -m  config/mic/${MIC}.yaml -r config/room/${ROOM}.yaml -n 1000

# 2023-04-14 : v4 train
PATH_OUT=/home/data2/kbh/${TASK}/${VERSION}/train
#python3 ./src/mix_for_sep.py -o ${PATH_OUT} -c config/${TASK}/${VERSION}.yaml -d config/default.yaml -m  config/mic/${MIC}.yaml -r config/room/${ROOM}.yaml -n 10000



# 2023-04-19 : v5

MIC=4ch_cross
VERSION=v5

PATH_OUT=/home/data2/kbh/${TASK}/${VERSION}/test
#python3 ./src/mix_for_sep.py -o ${PATH_OUT} -c config/${TASK}/${VERSION}_test.yaml -d config/default.yaml -m  config/mic/${MIC}.yaml -r config/room/${ROOM}.yaml -n 1000

PATH_OUT=/home/data2/kbh/${TASK}/${VERSION}/train
#python3 ./src/mix_for_sep.py -o ${PATH_OUT} -c config/${TASK}/${VERSION}.yaml -d config/default.yaml -m  config/mic/${MIC}.yaml -r config/room/${ROOM}.yaml -n 10000


# 2023-04-20

MIC=4ch_linear
VERSION=v6

VERSION=v7
MIC=4ch_linear_even

# 2023-04-21
VERSION=v8

# 2022-04-22
VERSION=v9

# 2022-04-24
VERSION=v10

# 2023-04-27
VERSION=v11

# 2023-05-01
VERSION=v12
MIC=4ch_cross_even

# 2023-05-09
TASK=UDSS
MIC=UMA-8
VERSION=v13
PATH_OUT=/home/data2/kbh/${TASK}/${VERSION}/train
#python3 ./src/mix_for_sep.py -o ${PATH_OUT} -c config/${TASK}/${VERSION}.yaml -d config/default.yaml -m  config/mic/${MIC}.yaml -r config/room/${ROOM}.yaml -n 10000

VERSION=v14
PATH_OUT=/home/data2/kbh/${TASK}/${VERSION}/train
#python3 ./src/mix_for_sep.py -o ${PATH_OUT} -c config/${TASK}/${VERSION}.yaml -d config/default.yaml -m  config/mic/${MIC}.yaml -r config/room/${ROOM}.yaml -n 10000


VERSION=sample
PATH_OUT=/home/data2/kbh/${TASK}/${VERSION}/
python3 ./src/mix_for_sep.py -o ${PATH_OUT} -c config/${TASK}/${VERSION}.yaml -d config/default.yaml -m  config/mic/${MIC}.yaml -r config/room/${ROOM}.yaml -n 10
