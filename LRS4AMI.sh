ROOT=/home/data/kbh/AVCSS
TASK=LRS4AMI
VERSION=v0

N=10000
python ./src/LRS4AMI.py -o ${ROOT}/${TASK}/${VERSION} -c ./config/${TASK}/${VERSION}.yaml -d ./config/${TASK}/default.yaml -n ${N}
python ./src/LRS4AMI_noise.py -o ${ROOT}/${TASK}/${VERSION} -c ./config/${TASK}/${VERSION}.yaml -d ./config/${TASK}/default.yaml -n ${N}
