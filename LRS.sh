ROOT=/home/data/kbh/DSS
TASK=LRS
VERSION=v8

N=10000
#python ./src/LRS_mixing.py -o ${ROOT}/${TASK}/${VERSION}/train -c ./config/${TASK}/${VERSION}.yaml -d ./config/${TASK}/default.yaml -n ${N}


N=100
python ./src/LRS_mixing.py -o ${ROOT}/${TASK}/${VERSION}/test -c ./config/${TASK}/${VERSION}_test.yaml -d ./config/${TASK}/default.yaml -n ${N}
