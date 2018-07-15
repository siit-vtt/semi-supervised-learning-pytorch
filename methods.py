import time
import math
import torch

def train_sup(label_loader, model, criterions, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    criterion, _, _, criterion_l1 = criterions

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
        
        reg_l1 = cal_reg_l1(model, criterion_l1)

        loss = loss_ce + args.weight_l1 * reg_l1

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

def train_pi(label_loader, unlabel_loader, model, criterions, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_pi = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    criterion, criterion_mse, _, criterion_l1 = criterions
    
    end = time.time()

    label_iter = iter(label_loader)     
    unlabel_iter = iter(unlabel_loader)     
    len_iter = len(label_iter)
    for i in range(len(label_iter)):
        # set weights for the consistency loss
        weight_cl = cal_consistency_weight(epoch*len_iter+i, end_ep=(args.epochs//2)*len_iter)
        
        input, target, input1 = next(label_iter)
        input_ul, _, input1_ul = next(unlabel_iter)
        sl = input.shape
        su = input_ul.shape
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        input1_var = torch.autograd.Variable(input1)
        input_ul_var = torch.autograd.Variable(input_ul)
        input1_ul_var = torch.autograd.Variable(input1_ul)
        input_concat_var = torch.cat([input_var, input_ul_var])
        input1_concat_var = torch.cat([input1_var, input1_ul_var])
       
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_concat_var)
        with torch.no_grad():
            output1 = model(input1_concat_var)

        output_label = output[:sl[0]]
        
        loss_ce = criterion(output_label, target_var)
        loss_pi = criterion_mse(output, output1)

        reg_l1 = cal_reg_l1(model, criterion_l1)

        loss = loss_ce + args.weight_l1 * reg_l1 + weight_cl * loss_pi

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output_label.data, target, topk=(1, 5))
        losses.update(loss_ce.item(), input.size(0))
        losses_pi.update(loss_pi.item(), input.size(0))
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
                  'LossPi {loss_pi.val:.4f} ({loss_pi.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(label_iter), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss_pi=losses_pi,
                   top1=top1, top5=top5))
    
    return top1.avg , losses.avg, losses_pi.avg


def validate(val_loader, model, criterions, args, mode = 'valid'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    criterion, criterion_mse, _, _ = criterions

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

def cal_consistency_weight(epoch, init_ep=0, end_ep=150, init_w=0.0, end_w=20.0):
    """Sets the weights for the consistency loss"""
    if epoch > end_ep:
        weight_cl = end_w
    elif epoch < init_ep:
        weight_cl = init_w
    else:
        T = float(epoch - init_ep)/float(end_ep - init_ep)
        #weight_mse = T * (end_w - init_w) + init_w #linear
        weight_cl = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w #exp
    #print('Consistency weight: %f'%weight_cl)
    return weight_cl

def cal_reg_l1(model, criterion_l1):
    reg_loss = 0
    for param in model.parameters():
        reg_loss += criterion_l1(param, torch.zeros_like(param))
    return reg_loss
 
 


