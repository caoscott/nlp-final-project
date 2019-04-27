import math
import os
import argparse
import logging
import sys
import time
from typing import List, Callable, Union, Tuple, Iterable, Generator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import init
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--num_channels', type=int, default=512)
parser.add_argument('--num_blocks', type=int, default=32)
parser.add_argument('--norm', type=str, choices=['batch', 'group'], default='group')

parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--load_model', type=str, default='')
# parser.add_argument('--boundary_epochs', type=int, nargs='+', default=[40, 90, 140])
args, _ = parser.parse_known_args()


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.modules.conv.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.modules.conv.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim: int) -> nn.Module:
    if args.norm == 'batch':
        return nn.BatchNorm2d(dim)
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: nn.Module = None) -> None:
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class Flatten(nn.Module):

    def __init__(self) -> None:
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum: float = 0.99) -> None:
        self.momentum = momentum
        self.val = None
        self.avg = 0

    def reset(self) -> None:
        self.val = None
        self.avg = 0

    def update(self, val: Union[int, float]) -> None:
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_cifar10_loaders(data_aug: bool = False, batch_size: int = 32,
                        test_batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) if data_aug else transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = DataLoader(
        datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=False
    )

    test_loader = DataLoader(
        datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=False
    )

    return train_loader, test_loader, train_eval_loader


def inf_generator(iterable: Iterable) -> Generator:
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size: int, batch_denom: int,
                             batches_per_epoch: int, boundary_epochs: List[int],
                             decay_rates: List[float]) -> Callable[[int], int]:
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr: int) -> int:
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x: np.ndarray, K: int) -> np.ndarray:
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model: nn.Module, dataset_loader: DataLoader) -> float:
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = y.to(device)
        _, predicted = torch.max(model(x), dim=1)
        total_correct += (predicted == y).sum().item()
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname: str) -> None:
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath: str, filepath: str, package_files: List = [],
               displaying: bool = True, saving: bool = True,
               debug: bool = False) -> logging.Logger:
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


class LoggerWrapper(object):
    def __init__(self, logger: logging.Logger, log_err: bool = False) -> None:
        self.logger = logger
        self.log_err = log_err

    def write(self, message: str) -> None:
        if self.log_err:
            self.logger.error(message.rstrip('\n'))
        else:
            self.logger.info(message.rstrip('\n'))

    def flush(self) -> None:
        pass


def main() -> None:
    makedirs(args.save)
    sys_logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    sys_logger.info(args)
    sys.stdout = LoggerWrapper(sys_logger, log_err=False)
    sys.stderr = LoggerWrapper(sys_logger, log_err=True)

    if args.downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(3, 64, 3, 1),
            norm(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 256, 4, 2, 1),
            norm(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, args.num_channels, 4, 2, 1),
        ]
    elif args.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(3, args.num_channels, 3, 1),
            ResBlock(args.num_channels, args.num_channels, stride=2,
                     downsample=conv1x1(args.num_channels, args.num_channels, 2)),
            ResBlock(args.num_channels, args.num_channels, stride=2,
                     downsample=conv1x1(args.num_channels, args.num_channels, 2)),
        ]

    feature_layers = [ODEBlock(ODEfunc(args.num_channels), args.tol, args.ode_solver)] if args.network == 'odenet' \
        else [ResBlock(args.num_channels, args.num_channels)
              for _ in range(args.num_blocks)] if args.network == 'resnet' \
        else [RepeatBlock(NotODEfunc(args.num_channels), args.num_blocks)]
    fc_layers = [norm(args.num_channels), nn.LeakyReLU(inplace=True),
                 nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(args.num_channels, 10)]

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
    if len(args.load_model) > 0:
        model.load_state_dict(torch.load(args.load_model)['state_dict'])
        sys_logger.info('Loaded model from {} successfully'.format(args.load_model))

    sys_logger.info(model)
    sys_logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader, train_eval_loader = get_cifar10_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )
    if args.nepochs <= 0:
        del train_loader
        model.eval()
        with torch.no_grad():
            train_acc = accuracy(model, train_eval_loader)
            test_acc = accuracy(model, test_loader)
            torch.save({'state_dict': model.state_dict(), 'args': args, 'test_acc': test_acc},
                       os.path.join(args.save, 'model.pth'))
            sys_logger.info(
                "Train Acc {:.4f} | Test Acc {:.4f}".format(
                    train_acc, test_acc
                )
            )
    else:
        del train_eval_loader
        data_gen = inf_generator(train_loader)
        batches_per_epoch = len(train_loader)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=7, verbose=True)

        best_acc = 0
        train_correct = 0
        batch_time_meter = RunningAverageMeter()
        f_nfe_meter = RunningAverageMeter()
        b_nfe_meter = RunningAverageMeter()
        train_loss_meter = RunningAverageMeter()
        end = time.time()

        for itr in range(args.nepochs * batches_per_epoch):

            optimizer.zero_grad()
            x, y = data_gen.__next__()
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            # for name, param in model.named_parameters():
            #     if args.network == 'repeatresnet' and 'intervals' in name:
            #         loss += args.weight_decay * param.norm(1)
            #     else:
            #         loss += args.weight_decay * param.norm(2)

            train_loss_meter.update(loss.item())

            _, predicted = torch.max(logits, dim=1)
            train_correct += (predicted == y).sum().item()

            if args.network == 'odenet':
                nfe_forward = feature_layers[0].nfe
                feature_layers[0].nfe = 0

            loss.backward()
            optimizer.step()

            if args.network == 'odenet':
                nfe_backward = feature_layers[0].nfe
                feature_layers[0].nfe = 0

            batch_time_meter.update(time.time() - end)
            if args.network == 'odenet':
                f_nfe_meter.update(nfe_forward)
                b_nfe_meter.update(nfe_backward)
            end = time.time()

            train_acc = train_correct / (args.batch_size * batches_per_epoch)
            if (itr + 1) % (batches_per_epoch * 4) == 0 or \
                    (itr + 1) % batches_per_epoch == 0 and (train_acc >= 0.98 or best_acc >= 0.9):
                train_correct = 0
                scheduler.step(train_acc)
                del x, y
                model.eval()
                with torch.no_grad():
                    # train_acc = accuracy(model, train_eval_loader)
                    val_acc = accuracy(model, test_loader)
                    if val_acc > best_acc:
                        torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                        best_acc = val_acc
                    sys_logger.info(
                        "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                        "Train Loss {:.6f} ({:.6f}) | Train Acc {:.4f} | Test Acc {:.4f}".format(
                            itr // batches_per_epoch, batch_time_meter.val,
                            batch_time_meter.avg, f_nfe_meter.avg,
                            b_nfe_meter.avg, train_loss_meter.val,
                            train_loss_meter.avg, train_acc, val_acc
                        )
                    )
                model.train()
            elif (itr + 1) % batches_per_epoch == 0:
                train_acc = train_correct / (args.batch_size * batches_per_epoch)
                scheduler.step(train_acc)
                train_correct = 0
                sys_logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Loss {:.6f} ({:.6f}) | Train Acc {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val,
                        batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_loss_meter.val,
                        train_loss_meter.avg, train_acc
                    )
                )

        sys_logger.info('Best test acc: {:.4f}'.format(best_acc))


if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    main()
