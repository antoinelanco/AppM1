#!/bin/bash
if ! [ -d "./res" ]; then
  mkdir res
fi
cd res
if ! [ -d "./mnist" ]; then
    mkdir mnist
fi
cd mnist
if ! [ -f "train-images-idx3-ubyte" ]; then
  wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
  gzip -d train-images-idx3-ubyte.gz
  rm -f train-images-idx3-ubyte.gz
fi
if ! [ -f "train-labels-idx1-ubyte" ]; then
  wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
  gzip -d train-labels-idx1-ubyte.gz
  rm -f train-labels-idx1-ubyte.gz
fi
if ! [ -f "t10k-images-idx3-ubyte" ]; then
  wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
  gzip -d t10k-images-idx3-ubyte.gz
  rm -f t10k-images-idx3-ubyte.gz
fi
if ! [ -f "t10k-labels-idx1-ubyte" ]; then
  wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
  gzip -d t10k-labels-idx1-ubyte.gz
  rm -f t10k-labels-idx1-ubyte.gz
fi
cd ../..
