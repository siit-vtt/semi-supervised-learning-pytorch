# this code is modified from the pytorch code: https://github.com/CSAILVision/places365
# JH Kim
#  

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import preresnet_sd_cifar as preresnet_cifar
import wideresnet
import pdb
import bisect

import loader_cifar as cifar
import loader_cifar_zca as cifar_zca
import math
from math import ceil
import torch.nn.functional as F
from methods import train_sup, train_pi, train_mt, validate


parser = argparse.ArgumentParser(description='PyTorch Semi-supervised learning Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='wideresnet',
                    help='model architecture: '+ ' (default: wideresnet)')
parser.add_argument('--model', '-m', metavar='MODEL', default='baseline',
                    help='model: '+' (default: baseline)', choices=['baseline', 'pi', 'mt'])
parser.add_argument('--optim', '-o', metavar='OPTIM', default='adam',
                    help='optimizer: '+' (default: adam)', choices=['adam', 'sgd'])
parser.add_argument('--preproc', '-pre', metavar='PRE', default='zca',
                    help='image pre-processing: '+' (default: zca)', choices=['zca', 'meanstd'])
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=225, type=int,
                    metavar='N', help='mini-batch size (default: 225)')
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--weight_l1', '--l1', default=1e-3, type=float,
                    metavar='W1', help='l1 regularization (default: 1e-3)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num_classes',default=10, type=int, help='number of classes in the model')
parser.add_argument('--ckpt', default='ckpt', type=str, metavar='PATH',
                    help='path to save checkpoint (default: ckpt)')
parser.add_argument('--boundary',default=0, type=int, help='different label/unlabel division [0,9]')
parser.add_argument('--gpu',default=0, type=str, help='cuda_visible_devices')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

best_prec1 = 0
best_test_prec1 = 0
acc1_tr, losses_tr = [], []
losses_cl_tr = []
acc1_val, losses_val, losses_et_val = [], [], []
acc1_test, losses_test, losses_et_test = [], [], []
acc1_t_tr, acc1_t_val, acc1_t_test = [], [], []

def main():
    global args, best_prec1, best_test_prec1
    global acc1_tr, losses_tr 
    global losses_cl_tr
    global acc1_val, losses_val, losses_et_val
    global acc1_test, losses_test, losses_et_test
    args = parser.parse_args()
    print args
    # create model
    if args.arch == 'preresnet':
        print("Model: %s"%args.arch)
        model = preresnet_cifar.resnet(depth=32, num_classes=args.num_classes)
    elif args.arch == 'wideresnet':
        print("Model: %s"%args.arch)
        model = wideresnet.WideResNet(28, args.num_classes, widen_factor=3, dropRate=0.3, leakyRate=0.1)
    else:
        assert(False)
    
    if args.model == 'mt':
        import copy  
        model_teacher = copy.deepcopy(model)
        model_teacher = torch.nn.DataParallel(model_teacher).cuda()

    model = torch.nn.DataParallel(model).cuda()
    print model
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']

            model.load_state_dict(checkpoint['state_dict'])
            if args.model=='mt': model_teacher.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.optim == 'sgd' or args.optim == 'adam':
        pass
    else:
        print('Not Implemented Optimizer')
        assert(False)
 
    if args.preproc == 'meanstd' or args.preproc == 'zca':
        pass
    else:
        print('Not Implemented Pre-processing')
        assert(False)
        
    ckpt_dir = args.ckpt+'_'+args.arch+'_'+args.model+'_'+args.optim
    if args.preproc == 'zca': ckpt_dir = ckpt_dir + '_zca'
    ckpt_dir = ckpt_dir + '_e%d'%(args.epochs)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    print(ckpt_dir)
    cudnn.benchmark = True

    # Data loading code
    if args.preproc == 'meanstd':
        dataloader = cifar.CIFAR10
        num_classes = 10
 
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
 
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])    
        labelset = dataloader(root='/tmp/', split='label', download=True, transform=transform_train, boundary=args.boundary)
        unlabelset = dataloader(root='/tmp/', split='unlabel', download=True, transform=transform_train, boundary=args.boundary)

    elif args.preproc == 'zca':
        dataloader = cifar_zca.CIFAR10
        num_classes = 10

        # transform is implemented inside zca dataloader 
        transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
 
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        labelset = dataloader(root='cifar10_zca/cifar10_gcn_zca_v2.npz', split='label', download=True, transform=transform_train, boundary=args.boundary)
        unlabelset = dataloader(root='cifar10_zca/cifar10_gcn_zca_v2.npz', split='unlabel', download=True, transform=transform_train, boundary=args.boundary)
    
    label_size = len(labelset)
    unlabel_size = len(unlabelset)
    iter_per_epoch = int(ceil(float(label_size + unlabel_size)/args.batch_size))
    batch_size_label = int(ceil(float(label_size) / iter_per_epoch))
    batch_size_unlabel = int(ceil(float(unlabel_size) / iter_per_epoch))
    if args.model == 'baseline': batch_size_label=args.batch_size

    label_loader = data.DataLoader(labelset, 
        batch_size=batch_size_label, 
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=True)
    label_iter = iter(label_loader) 

    unlabel_loader = data.DataLoader(unlabelset, 
        batch_size=batch_size_unlabel, 
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=True)
    unlabel_iter = iter(unlabel_loader) 

    if args.model is not 'baseline':
        if len(label_iter) != len(unlabel_iter):
            print('Number of label and unlabel iteration is not match, %d, %d'%(len(label_iter) ,len(unlabel_iter)))
            assert(False)
    print("Batch size (label): ", batch_size_label)
    print("Batch size (unlabel): ", batch_size_unlabel)


    validset = dataloader(root='cifar10_zca/cifar10_gcn_zca_v2.npz', split='valid', download=False, transform=transform_test, boundary=args.boundary)
    val_loader = data.DataLoader(validset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True)

    testset = dataloader(root='cifar10_zca/cifar10_gcn_zca_v2.npz', split='test', download=False, transform=transform_test)
    test_loader = data.DataLoader(testset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True)

    # deifine loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_mse = nn.MSELoss().cuda()
    criterion_kl = nn.KLDivLoss().cuda()    
    criterion_l1 = nn.L1Loss(size_average=False).cuda()
   
    criterions = (criterion, criterion_mse, criterion_kl, criterion_l1)

    if args.optim == 'adam':
        print('Using Adam optimizer')
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    betas=(0.9,0.999),
                                    weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        print('Using SGD optimizer')
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        if args.optim == 'adam':
            print('Learning rate schedule for Adam')
            adjust_learning_rate_adam(optimizer, epoch)
        elif args.optim == 'sgd':
            print('Learning rate schedule for SGD')
            adjust_learning_rate(optimizer, epoch)
        
        # train for one epoch
        if args.model == 'baseline':
            print('Supervised Training')
            for i in range(10): #baseline repeat 10 times since small number of training set 
                prec1_tr, loss_tr = train_sup(label_loader, model, criterions, optimizer, epoch, args)
        elif args.model == 'pi':
            print('Pi model')
            prec1_tr, loss_tr, loss_cl_tr = train_pi(label_loader, unlabel_loader, model, criterions, optimizer, epoch, args)
        elif args.model == 'mt':
            print('Mean Teacher model')
            prec1_tr, loss_tr, loss_cl_tr, prec1_t_tr = train_mt(label_loader, unlabel_loader, model, model_teacher, criterions, optimizer, epoch, args)
        else:
            print("Not Implemented ", args.model)
            assert(False)
        
        # evaluate on validation set        
        prec1_val, loss_val = validate(val_loader, model, criterions, args, 'valid')
        prec1_test, loss_test = validate(test_loader, model, criterions, args, 'test')
        if args.model=='mt':
            prec1_t_val, loss_t_val = validate(val_loader, model_teacher, criterions, args, 'valid')
            prec1_t_test, loss_t_test = validate(test_loader, model_teacher, criterions, args, 'test')

        # append values
        acc1_tr.append(prec1_tr)
        losses_tr.append(loss_tr)
        acc1_val.append(prec1_val)
        losses_val.append(loss_val)
        acc1_test.append(prec1_test)
        losses_test.append(loss_test)
        if loss_cl_tr is not None: losses_cl_tr.append(loss_cl_tr)
        elif args.model=='mt':
            acc1_t_tr.append(prec1_t_tr)
            acc1_t_val.append(prec1_t_val)
            acc1_t_test.append(prec1_t_test)

        # remember best prec@1 and save checkpoint
        if args.model == 'mt': 
            is_best = prec1_t_val > best_prec1
            if is_best:
                best_test_prec1_t = prec1_t_test
                best_test_prec1 = prec1_test
            print("Best test precision: %.3f"%best_test_prec1_t)
            best_prec1 = max(prec1_t_val, best_prec1)
            dict_checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_test_prec1' : best_test_prec1,
                'acc1_tr': acc1_tr,
                'losses_tr': losses_tr,
                'losses_cl_tr': losses_cl_tr,
                'acc1_val': acc1_val,
                'losses_val': losses_val,
                'acc1_test' : acc1_test,
                'losses_test' : losses_test,
                'acc1_t_tr': acc1_t_tr,
                'acc1_t_val': acc1_t_val,
                'acc1_t_test': acc1_t_test,
                'state_dict_teacher': model_teacher.state_dict(),
                'best_test_prec1_t' : best_test_prec1_t,
            }
       
        else:
            is_best = prec1_val > best_prec1
            if is_best:
                best_test_prec1 = prec1_test
            print("Best test precision: %.3f"%best_test_prec1)
            best_prec1 = max(prec1_val, best_prec1)
            dict_checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_test_prec1' : best_test_prec1,
                'acc1_tr': acc1_tr,
                'losses_tr': losses_tr,
                'losses_cl_tr': losses_cl_tr,
                'acc1_val': acc1_val,
                'losses_val': losses_val,
                'acc1_test' : acc1_test,
                'losses_test' : losses_test,
            }

        save_checkpoint(dict_checkpoint, is_best, args.arch.lower()+str(args.boundary), dirname=ckpt_dir)
        
        
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', dirname='.'):
    fpath = os.path.join(dirname, filename + '_latest.pth.tar')
    torch.save(state, fpath)
    if is_best:
        bpath = os.path.join(dirname, filename + '_best.pth.tar')
        shutil.copyfile(fpath, bpath)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 at [150, 225, 300] epochs"""
    
    boundary = [args.epochs//2,args.epochs//4*3,args.epochs]
    lr = args.lr * 0.1 ** int(bisect.bisect_left(boundary, epoch))
    print('Learning rate: %f'%lr)
    #print(epoch, lr, bisect.bisect_left(boundary, epoch))
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_adam(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 5 at [240] epochs"""
    
    boundary = [args.epochs//5*4]
    lr = args.lr * 0.2 ** int(bisect.bisect_left(boundary, epoch))
    print('Learning rate: %f'%lr)
    #print(epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

  

if __name__ == '__main__':
    main()
