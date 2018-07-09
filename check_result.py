import torch
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('--fdir', default='ckpt', type=str, metavar='PATH',
                    help='path to load checkpoint (default: ckpt)')
parser.add_argument('--fname', default='preresnet', type=str, metavar='PATH',
                    help='checkpoint filename (default: preresnet)')
parser.add_argument('--nckpt',default=1, type=int, help='num of checkpoints')
parser.add_argument('--plot',default=False, type=bool, help='num of checkpoints')
args = parser.parse_args()

fdir = args.fdir
fname = args.fname
nckpt = args.nckpt

best_prec1s = []
for i in range(nckpt):
    path = os.path.join(fdir,fname+str(i)+'_latest.pth.tar')
    checkpoint = torch.load(path)
    print(path)
    best_prec1 = checkpoint['best_test_prec1']
    best_prec1_val = checkpoint['best_prec1']
    print(best_prec1)
    print(best_prec1_val)
    best_prec1s.append(best_prec1)

best_prec1s = np.array(best_prec1s)
bmean = np.around(np.mean(best_prec1s), decimals=2)
bstd = np.around(np.std(best_prec1s), decimals=2)
print('Best precision: %.2f(%.2f)'%(bmean,bstd))
#print('Best precision: ',bmean,'(',bstd,')')

#for key, val in checkpoint.iteritems():
#  print(key)

fname_acc = os.path.join(fdir,'accuracy.png')
fname_loss = os.path.join(fdir,'losses.png')
acc1_tr = checkpoint['acc1_tr']
acc1_val = checkpoint['acc1_val']
acc1_te = checkpoint['acc1_test']
losses_tr = checkpoint['losses_tr']
losses_val = checkpoint['losses_val']
losses_te = checkpoint['losses_test']

if(args.plot):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.subplot(2,1,1) 
    ax.plot(acc1_tr, label='train_acc1')
    ax.plot(acc1_val, label='val_acc1')
    ax.plot(acc1_te, label='test_acc1')
    ax.legend()
    ax.grid(linestyle='--')
    ax = plt.subplot(2,1,2) 
    ax.plot(losses_tr, label='train_loss')
    ax.plot(losses_val, label='val_loss')
    ax.plot(losses_te, label='test_loss')
    ax.legend()
    ax.grid(linestyle='--')
    plt.savefig(fname_acc)
    #plt.show()
    plt.clf()

    #plt.show()
