#!/bin/sh

MODEL=bvlc_googlenet

python ./caffe/scripts/download_model_binary.py "./caffe/models/${MODEL}"

mkdir -p "./models/${MODEL}"

cp "./caffe/models/${MODEL}/${MODEL}.caffemodel" "./models/${MODEL}/"
