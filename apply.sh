#!/bin/bash



DIR_LRS=/home/data/kbh/LRS3/trainval
DIR_RIR=/home/data/kbh/UMA8_data/RIR/R705
DIR_OUT=/home/data/kbh/RIR/LRS3/trainval_R705
python src/apply_RIR.py -i ${DIR_LRS} -r ${DIR_RIR} -o ${DIR_OUT}

DIR_LRS=/home/data/kbh/LRS3/test
DIR_RIR=/home/data/kbh/UMA8_data/RIR/R907
DIR_OUT=/home/data/kbh/RIR/LRS3/test_R907
python src/apply_RIR.py -i ${DIR_LRS} -r ${DIR_RIR} -o ${DIR_OUT}
