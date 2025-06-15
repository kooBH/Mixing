
ROOT=/home/data2/kbh/DSS/UMA8
VERSION=v11
N=1000

ROOT=/home/data2/kbh/DSS/UMA8/
N=10000
python ./src/RIR_mat_mixing.py -o ${ROOT}/${VERSION}/train -c ./config/UMA8/${VERSION}.yaml -d ./config/UMA8/default.yaml -n ${N}


N=1000
python ./src/RIR_mat_mixing.py -o ${ROOT}/${VERSION}/test -c ./config/UMA8/${VERSION}_test.yaml -d ./config/UMA8/default.yaml -n ${N}
