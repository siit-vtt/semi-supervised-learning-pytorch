import torch
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('--fdir', default='ckpt', type=str, metavar='PATH',
                    help='path to load checkpoint (default: ckpt)')
parser.add_argument('--fname', default='wideresnet', type=str, metavar='PATH',
                    help='checkpoint filename (default: wideresnet)')
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
    if 'best_test_prec1_t' in checkpoint:
        print("Teacher precision")
        best_prec1 = 100.0 - checkpoint['best_test_prec1_t']
    else:
        best_prec1 = 100.0 - checkpoint['best_test_prec1']
    best_prec1_val = 100.0 - checkpoint['best_prec1']
    print('Test Error: ',best_prec1)
    print('Val. Error: ',best_prec1_val)
    best_prec1s.append(best_prec1)

    fname_acc = os.path.join(fdir,'accuracy%d.png'%i)
    fname_lr = os.path.join(fdir,'lr%d.png'%i)
    fname_loss = os.path.join(fdir,'losses%d.png'%i)
    acc1_tr = checkpoint['acc1_tr']
    acc1_val = checkpoint['acc1_val']
    acc1_te = checkpoint['acc1_test']
    losses_tr = checkpoint['losses_tr']
    losses_val = checkpoint['losses_val']
    losses_te = checkpoint['losses_test']
    weights_cl = checkpoint['weights_cl']
    learning_rate = checkpoint['learning_rate']
    losses_cl_tr = []
    if 'losses_cl_tr' in checkpoint:
        losses_cl_tr = checkpoint['losses_cl_tr']

    if(args.plot):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = plt.subplot(1,1,1) 
        ax.plot(acc1_tr, label='train_acc1')
        ax.plot(acc1_val, label='val_acc1')
        ax.plot(acc1_te, label='test_acc1')
        ax.legend()
        ax.grid(linestyle='--')
        plt.savefig(fname_acc)
        #plt.show()
        plt.clf()

        fig = plt.figure()
        ax = plt.subplot(2,1,1) 
        ax.plot(learning_rate, label='lr')
        ax.legend()
        ax.grid(linestyle='--')
        ax = plt.subplot(2,1,2) 
        ax.plot(weights_cl, label='w_cl')
        ax.legend()
        ax.grid(linestyle='--')
        plt.savefig(fname_lr)
        #plt.show()
        plt.clf()


        fig = plt.figure()
        ax = plt.subplot(2,1,1) 
        ax.plot(losses_tr, label='train_loss')
        ax.plot(losses_val, label='val_loss')
        ax.plot(losses_te, label='test_loss')
        ax.legend()
        ax.grid(linestyle='--')
        ax = plt.subplot(2,1,2) 
        ax.plot(losses_cl_tr, label='train_loss_cl')
        ax.legend()
        ax.grid(linestyle='--')
        plt.savefig(fname_loss)
        #plt.show()
        plt.clf()


    #plt.show()
best_prec1s = np.array(best_prec1s)
bmean = np.around(np.mean(best_prec1s), decimals=2)
bstd = np.around(np.std(best_prec1s), decimals=2)
print('Best error rate: %.2f(%.2f)'%(bmean,bstd))
#print('Best precision: ',bmean,'(',bstd,')')

#for key, val in checkpoint.iteritems():
#  print(key)


