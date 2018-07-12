#!/bin/bash

python train_cifar10.py -a=wideresnet -o=adam --ckpt=ckpt --gpu=0,1 --lr=0.003 --boundary=0 &&


ls
