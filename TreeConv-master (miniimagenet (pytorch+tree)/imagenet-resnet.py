#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet-resnet.py

import argparse
import os

# from tensorpack import QueueInput, TFDatasetInput, logger
# from tensorpack.callbacks import *
# from tensorpack.dataflow import FakeData
# from tensorpack.models import *
# from tensorpack.tfutils import argscope, SmartInit
# from tensorpack.train import SyncMultiGPUTrainerReplicated, TrainConfig, launch_train_with_config
# from tensorpack.utils.gpu import get_num_gpu

# from imagenet_utils import ImageNetModel, eval_classification, get_imagenet_dataflow, get_imagenet_tfdata
import resnet_model
#from resnet_model import preact_group, resnet_backbone, resnet_group
from resnet_model import resnet50, resnet18
from  conv import Conv

import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import time
from tqdm import tqdm

###
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.gamma ** (epoch // 30))
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

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(tqdm(train_loader)):
        # Initial warming-up
        if epoch == 0 and i == 0:
            print("Temporarily lowered LR")
            adjust_learning_rate(optimizer, 120)
        if epoch == 0 and i == 1000:
            print("Recover LR")
            adjust_learning_rate(optimizer, 0)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.cuda()
        
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    print('Epoch: [{0}][{1}/{2}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        epoch, i, len(train_loader), batch_time=batch_time,
        data_time=data_time, loss=losses, top1=top1, top5=top5))


@torch.no_grad()
def validate(val_loader, model, criterion, multiple_crops=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if multiple_crops:
        # XXX: Loss outputs are not valid (due to duplicated softmax)
        model = MultiCropEnsemble(model, 224, act=nn.functional.softmax)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(tqdm(val_loader)):
        target = target.cuda()
        input = input.cuda()
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        # losses.update(loss.data[0], input.size(0))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #            i, len(val_loader), batch_time=batch_time, loss=losses,
        #            top1=top1, top5=top5))
    print('Test: [{0}/{1}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        i, len(val_loader), batch_time=batch_time, loss=losses,
        top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

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

class MultiCropEnsemble(nn.Module):
    def __init__(self, module, cropsize, act=nn.functional.softmax, flipping=True):
        super(MultiCropEnsemble, self).__init__()
        self.cropsize = cropsize
        self.flipping = flipping
        self.internal_module = module
        self.act = act

    # Naive code
    def forward(self, x):
        # H, W >= cropsize
        assert(x.size()[2] >= self.cropsize)
        assert(x.size()[3] >= self.cropsize)

        cs = self.cropsize
        x1 = 0
        x2 = x.size()[2] - self.cropsize
        cx = x.size()[2] // 2 - self.cropsize // 2
        y1 = 0
        y2 = x.size()[3] - self.cropsize
        cy = x.size()[3] // 2 - self.cropsize // 2

        get_output = lambda x: self.act(self.internal_module.forward(x))

        _y = get_output(x[:, :, x1:x1+cs, y1:y1+cs])
        _y = get_output(x[:, :, x1:x1+cs, y2:y2+cs]) + _y
        _y = get_output(x[:, :, x2:x2+cs, y1:y1+cs]) + _y
        _y = get_output(x[:, :, x2:x2+cs, y2:y2+cs]) + _y
        _y = get_output(x[:, :, cx:cx+cs, cy:cy+cs]) + _y

        if self.flipping == True:
            # Naive flipping

            arr = (x.data).cpu().numpy()                        # Bring back to cpu
            arr = arr[:,:,:, ::-1]                              # Flip
            x.data = type(x.data)(np.ascontiguousarray(arr))    # Store

            _y = get_output(x[:, :, x1:x1 + cs, y1:y1 + cs]) + _y
            _y = get_output(x[:, :, x1:x1 + cs, y2:y2 + cs]) + _y
            _y = get_output(x[:, :, x2:x2 + cs, y1:y1 + cs]) + _y
            _y = get_output(x[:, :, x2:x2 + cs, y2:y2 + cs]) + _y
            _y = get_output(x[:, :, cx:cx + cs, cy:cy + cs]) + _y

            _y = _y / 10.0
        else:
            _y = _y / 5.0

        return _y


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # generic:
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use. Default to use all available ones')
    parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')
    parser.add_argument('--load', help='load a model for training or evaluation')

    # data:
    parser.add_argument('--data', default = '/media/dm/d/mini_imagenet', help='ILSVRC dataset dir')
    parser.add_argument('--fake', help='use FakeData to debug or benchmark this model', action='store_true')
    parser.add_argument('--symbolic', help='use symbolic data loader', action='store_true')

    # model:
    parser.add_argument('--data-format', help='the image data layout used by the model',
                        default='NHWC', choices=['NCHW', 'NHWC']) #NCHW
    parser.add_argument('-d', '--depth', help='ResNet depth',
                        type=int, default=50, choices=[18, 34, 50, 101, 152])
    parser.add_argument('--weight-decay-norm', action='store_true',
                        help="apply weight decay on normalization layers (gamma & beta)."
                             "This is used in torch/pytorch, and slightly "
                             "improves validation accuracy of large models.")
    parser.add_argument('--batch_size', default=32, type=int, #256
                        help="total batch size. "
                        "Note that it's best to keep per-GPU batch size in [32, 64] to obtain the best accuracy."
                        "Pretrained models listed in README were trained with batch=32x8.")
    parser.add_argument('--mode', choices=['resnet', 'preact', 'se', 'resnext32x4d'],
                        help='variants of resnet to use', default='resnet')
    ###
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--print_freq',default=10, type=int,help='print_freq')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='LR decay rate')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    help='model architecture: ' +
                        ' | '.join('model_name') +
                        ' (default: resnet50)')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ###
    # model = Model(args.depth, args.mode)
    model = resnet50()
    model.cuda()
    # define loss function (criterion) and optimizer
    #optimizer = Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    #scheduler = StepLR
    #criterion = CrossEtropyLoss()
    criterion = nn.CrossEntropyLoss().cuda()
    # data_loader = Dataset()
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    best_prec1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)