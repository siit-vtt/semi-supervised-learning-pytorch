# this code is modified from the pytorch example code: https://github.com/pytorch/examples/blob/master/imagenet/main.py
# after the model is trained, you might use convert_model.py to remove the data parallel module to make the model as standalone weight.
#
# Bolei Zhou

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
import math
from math import ceil
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='wideresnet',
                    help='model architecture: '+ ' (default: wideresnet)')
parser.add_argument('--model', '-m', metavar='MODEL', default='baseline',
                    help='model: '+
                        ' (default: baseline)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=225, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num_classes',default=10, type=int, help='num of class in the model')
parser.add_argument('--ckpt', default='ckpt', type=str, metavar='PATH',
                    help='path to save checkpoint (default: ckpt)')
parser.add_argument('--boundary',default=0, type=int, help='num of class in the model')
parser.add_argument('--gpu',default=0, type=str, help='cuda_visible_devices')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

best_prec1 = 0
best_test_prec1 = 0
acc1_tr, losses_tr, losses_et_tr, losses_kl_tr, losses_klf_tr = [], [], [], [], []
acc1_val, losses_val, losses_et_val = [], [], []
acc1_test, losses_test, losses_et_test = [], [], []

def main():
    global args, best_prec1, best_test_prec1
    global acc1_tr, losses_tr, losses_et_tr, losses_kl_tr, losses_klf_tr
    global acc1_val, losses_val, losses_et_val
    global acc1_test, losses_test, losses_et_test
    args = parser.parse_args()
    print args
    # create model
    if args.arch == 'preresnet':
        model = preresnet_cifar.resnet(depth=32, num_classes=args.num_classes)
    elif args.arch == 'wideresnet':
        model = wideresnet.WideResNet(28, args.num_classes, widen_factor=2, dropRate=0.3)
    else:
        assert(False)
        
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
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ckpt_dir = args.ckpt+'_'+args.arch+'_'+args.model
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    cudnn.benchmark = True

    # Data loading code
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

    labelset = dataloader(root='/tmp', split='label', download=True, transform=transform_train, boundary=args.boundary)
    unlabelset = dataloader(root='/tmp', split='unlabel', download=True, transform=transform_train, boundary=args.boundary)
    
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


    validset = dataloader(root='/tmp', split='valid', download=False, transform=transform_test, boundary=args.boundary)
    val_loader = data.DataLoader(validset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True)

    testset = dataloader(root='/tmp', split='test', download=False, transform=transform_test)
    test_loader = data.DataLoader(testset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True)

    # deifine loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_mse = nn.MSELoss().cuda()
    criterions = (criterion, criterion_mse)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        
        # train for one epoch
        if args.model == 'baseline':
            for i in range(10):
                prec1_tr, loss_tr = train_sup(label_loader, model, criterions, optimizer, epoch)
        else:
            print("Not Implemented ", args.model)
            assert(False)

        # evaluate on validation set
        
        prec1_val, loss_val = validate(val_loader, model, criterions, 'valid')
        prec1_test, loss_test = validate(test_loader, model, criterions, 'test')

        # append values
        acc1_tr.append(prec1_tr)
        losses_tr.append(loss_tr)
        acc1_val.append(prec1_val)
        losses_val.append(loss_val)
        acc1_test.append(prec1_test)
        losses_test.append(loss_test)

        # remember best prec@1 and save checkpoint
        is_best = prec1_val > best_prec1
        if is_best:
            best_test_prec1 = prec1_test
        best_prec1 = max(prec1_val, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_test_prec1' : best_test_prec1,
            'acc1_tr': acc1_tr,
            'losses_tr': losses_tr,
            'acc1_val': acc1_val,
            'losses_val': losses_val,
            'acc1_test' : acc1_test,
            'losses_test' : losses_test,
        }, is_best, args.arch.lower()+str(args.boundary), dirname=ckpt_dir)

        

def train(train_loader, model, criterions, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_et = AverageMeter()
    losses_kl = AverageMeter()
    losses_klf = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # weight anneal
    '''
    init_ep, end_ep, init_w, end_w = 40, 120, args.weight_mse, args.weight_mse
    if epoch > end_ep:
        weight_mse = end_w
    elif epoch < init_ep:
        weight_mse = init_w
    else:
        T = float(epoch - init_ep)/float(end_ep - init_ep)
        #weight_mse = T * (end_w - init_w) + init_w #linear
        weight_mse = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w #exp
    #print(epoch, weight_mse)
    '''
    # switch to train mode
    model.train()

    criterion, criterion_mse = criterions
    train_loader.dataset.midx=epoch%2

    end = time.time()
     
    for i, (input, target, inputu, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        
        loss_ce = criterion(output, target_var)

        loss = loss_ce

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss_ce.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    
    return top1.avg , losses.avg

def train_sup(label_loader, model, criterions, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_et = AverageMeter()
    losses_kl = AverageMeter()
    losses_klf = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    criterion, criterion_mse = criterions

    end = time.time()

    label_iter = iter(label_loader)     
    for i in range(len(label_iter)):
        input, target, _ = next(label_iter)
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        
        loss_ce = criterion(output, target_var)

        loss = loss_ce

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss_ce.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(label_iter), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    
    return top1.avg , losses.avg


def validate(val_loader, model, criterions, mode = 'valid'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    criterion, criterion_mse = criterions

    end = time.time()
    with torch.no_grad():
        for i, (input, target, _) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
 
            # compute output
            output = model(input_var)
            softmax = torch.nn.LogSoftmax(dim=1)(output)
            loss = criterion(output, target_var)
 
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
 
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
 
            if i % args.print_freq == 0:
                if mode == 'test':
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           i, len(val_loader), batch_time=batch_time, loss=losses,
                           top1=top1, top5=top5))
                else:
                    print('Valid: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           i, len(val_loader), batch_time=batch_time, loss=losses,
                           top1=top1, top5=top5))

    print(' ****** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.3f} '
          .format(top1=top1, top5=top5, loss=losses))

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', dirname='.'):
    fpath = os.path.join(dirname, filename + '_latest.pth.tar')
    torch.save(state, fpath)
    if is_best:
        bpath = os.path.join(dirname, filename + '_best.pth.tar')
        shutil.copyfile(fpath, bpath)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 at [100, 150, 200] epochs"""
    
    boundary = [150,225,300]
    lr = args.lr * 0.1 ** int(bisect.bisect_left(boundary, epoch))
    #print(epoch, lr, bisect.bisect_left(boundary, epoch))
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
