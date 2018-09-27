# ssl (semi-supervised learning)
This repository contains code to reproduce "[Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://arxiv.org/abs/1804.09170)" in pytorch. Currently, only supervised baseline, PI-model[2] and Mean-Teacher[3] are implemented. We attempted to follow the description in the paper, but there are several differences made intentionally. There may be other differences made accidentally from experiments in the paper. 

* The training code is under modification.

# Prerequisites
Tested on 
* python 2.7
* pytorch 0.4.0

Download ZCA preprocessed CIFAR-10 dataset
* As described in the paper, global contrast normalize (GCN) and ZCA are important steps for the performance. We preprocess CIFAR-10 dataset using the code implemented in [Mean-Teacher repository](https://github.com/CuriousAI/mean-teacher). The code is in tensorflow/dataset folder.
Place the preprocessed file (e.g. cifar10_gcn_zca_v2.npz) into a subfolder (e.g. cifar10_zca).

# Experiment detail
 

# To Run
For basline 
    
    python train.py -a=wideresnet -m=baseline -o=adam -b=225 --dataset=cifar10_zca --gpu=0,1 --lr=0.003 --boundary=0

 For Pi model

    python train.py -a=wideresnet -m=pi -o=adam -b=225 --dataset=cifar10_zca --gpu=0,1 --lr=0.0003 --boundary=0
 For Mean Teacher

    python train.py -a=wideresnet -m=mt -o=adam -b=225 --dataset=cifar10_zca --gpu=0,1 --lr=0.0004 --boundary=0
    
* boundary option is for different label/unlabel division [0, 9].
    
You can check the average error rates for `n` runs using `check_result.py`. For example, you trained baseline model on 10 different boundary,

    python check_result.py --fdir ckpt_cifar10_zca_wideresnet_baseline_adam_e1200/ --fname wideresnet --nckpt 10 
    
# Result (CIFAR-10)
|Method       |WideResnet28x2 [1]    |WideResnet28x3 w/ dropout (ours)   |
|-------------|----------------------|-----------------------------------|
|Supervised   |20.26 (0.38)          |                                   |
|PI Model     |16.37 (0.63)          |                                   |
|Mean Teacher |15.87 (0.28)          |                                   |
|VAT          |13.86 (0.27)          |-                                  |
|VAT + EM     |13.13 (0.39)          |-                                  |


# References
[1] Oliver, Avital, et al. "Realistic Evaluation of Deep Semi-Supervised Learning Algorithms." arXiv preprint arXiv:1804.09170 (2018).

[2] Laine, Samuli, and Timo Aila. "Temporal ensembling for semi-supervised learning." arXiv preprint arXiv:1610.02242 (2016).

[3] Tarvainen, Antti, and Harri Valpola. "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results." Advances in neural information processing systems. 2017.

[4] https://github.com/CuriousAI/mean-teacher

[5] https://github.com/facebookresearch/odin
