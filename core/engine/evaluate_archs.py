import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset

from core.model.ofa_mbv3 import OFAMobileNetV3
from utils.metrics import EstimatedTimeArrival
from utils.log import logger, output_directory
from core.model.utils import set_running_statistics
from core.controller import str2arch, arch2str, Arch


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        # from pprint import pprint
        # pprint(res)
        return res


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


def validate(net, val_loader):
    net.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for j, (images, labels) in enumerate(val_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            # print(outputs[0])
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))
    return top1.avg, top5.avg


def evaluate_archs(archs, width, imagenet_path, batch_size=256, num_workers=8):
    net = OFAMobileNetV3(
        dropout_rate=0,
        width_mult_list=width,
        ks_list=[3, 5, 7],
        expand_ratio_list=[3, 4, 6],
        depth_list=[2, 3, 4],
    )
    pretrained_model = os.path.join("assets", "ofa_nets", f"ofa_mbv3_d234_e346_k357_w{width}.pth")
    net.load_state_dict(torch.load(pretrained_model, map_location="cpu")['state_dict'])
    net.eval()
    net.cuda()

    traindir = os.path.join(imagenet_path, "train")
    valdir = os.path.join(imagenet_path, "val")
    trainset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))

    n_samples = len(trainset)
    g = torch.Generator()
    g.manual_seed(937162211)  # seed from OFA code
    index = torch.randperm(n_samples, generator=g).tolist()

    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(index[:2000])
    sub_train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=100, sampler=sub_sampler,
        num_workers=num_workers, pin_memory=False)

    valset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]))
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    if len(archs) > 1:
        cache_tensors = []
        logger.info(f"Loading val dataset into memory to accelerate the evaluation.")
        for i, data in enumerate(val_loader):
            cache_tensors.append(data)
    else:
        cache_tensors = val_loader

    eta = EstimatedTimeArrival(len(archs))
    for arch in archs:
        if arch["arch"] == "":
            continue
        architecture: Arch = str2arch(arch["arch"])
        logger.info(f"start to test arch {architecture}")
        net.set_active_subnet(ks=architecture.ks, e=architecture.ratios, d=architecture.depths)
        subnet = net.get_active_subnet(preserve_weight=True)
        set_running_statistics(subnet, sub_train_loader)
        top1, top5 = validate(subnet, cache_tensors)
        eta.step()
        logger.info(f"top-1 accuracy={top1:.2f}%, top-5 accuracy={top5:.5f}%, eta={eta.remaining_time}, arrival={eta.arrival_time}")

        arch["top1"] = top1/100.0
        arch["top5"] = top5/100.0
