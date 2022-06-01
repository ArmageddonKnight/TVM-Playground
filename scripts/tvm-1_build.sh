#!/bin/bash -e

CWD=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)

case $1 in
        --parallel)
                nthreads=$2
                ;;
        *)
                printf "Unrecognized argument: $1\n"
                exit -1
                ;;
esac

TVM_ROOT_DIR=${CWD}/../tvm
cd ${TVM_ROOT_DIR}

mkdir -p build && cd build

if [ ! -f Makefile ]
then
        cmake -DUSE_CUDA=/usr/local/cuda/ \
              -DUSE_LLVM=/usr/lib/llvm-14/bin/llvm-config \
              -DUSE_CUBLAS=1 \
              -DUSE_CUDNN=1 ..
fi

if [ -z ${nthreads} ]
then
        nthreads=$(nproc)
fi
printf "Building with nthreads=${nthreads}\n"
make -j ${nthreads}

cd ${TVM_ROOT_DIR}/python

if [ ! -d requirements ]
then
        python3 gen_requirements.py
        cd requirements
        pip3 install -r core.txt
        pip3 install -r xgboost.txt
        cd ..
fi

python3 setup.py build
