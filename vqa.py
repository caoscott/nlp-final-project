import json
from collections import Counter, defaultdict

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
from PIL import Image
from torch import optim
from torch.nn import init
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import embedding


def norm(dim: int) -> nn.Module:
    if args.norm == 'batch':
        return nn.BatchNorm2d(dim)
    return nn.GroupNorm(min(32, dim), dim)


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


class VQADataset(data.Dataset):

    def __init__(self, dataset_path: str, transform, mode: str = 'train',
                 word_embedding_file: str = 'glove.6B.300d-relativized.txt'):
        self.word_embeddings = embedding.read_word_embeddings(word_embedding_file)
        answer_frequency = defaultdict(int)
        questions_dict = \
        json.loads(open(os.path.join(dataset_path, 'v2_OpenEnded_mscoco_{}2014_questions.json'.format(mode))))[
            'questions']
        annotations_dict = \
        json.loads(open(os.path.join(dataset_path, 'v2_mscoco_{}2014_annotations.json'.format(mode))))['annotations']

        dataset_dict = {}
        for question in questions_dict:
            dataset_dict[question['question_id']] = {'question': question}
        for annotation in annotations_dict:
            dataset_dict[annotation['question_id']]['annotation'] = annotation
            answer_frequency[annotation['multiple_choice_answer']] += 1

        top_answers = sorted([(v, k) for k, v in answer_frequency.items()], reversed=True)[:1000]
        self.answer_to_idx = {ans: idx for idx, (_, ans) in enumerate(top_answers.items())}
        for _, data in dataset_dict.items():
            data['answer_index'] = self.answer_to_idx[data['annotation']['multiple_choice_answer']]
            data['question_embedding'] = torch.tensor([self.word_embeddings.get_embedding(word)
                                                       for word in data['question']['question']])
        self.dataset = [(k, v) for k, v in dataset_dict.items()]
        self.mode = mode
        self.dataset_path = dataset_path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        question_id, data = self.dataset[idx]
        image_id = '{:012d}'.format(data['image_id'])
        year = '2015' if self.mode == 'test' else '2014'
        with open(os.path.join(self.dataset_path, self.mode, year, image_id), 'rb') as f:
            img = Image.open(f).convert('RGB')
        return self.transform(img), data['question_embedding'], torch.tensor(data['answer_index'])

def get_loaders(data_aug: bool = False, batch_size: int = 32,
                test_batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_path = ''
    test_path = ''
    word_embedding_file = 'glove.6B.300d-relativized.txt'
    vqa_train = VQADataset(train_path, transform_train, 'train',  word_embedding_file)
    vqa_test = VQADataset(test_path, transform_test, 'test', word_embedding_file)

    train_loader = DataLoader(
        vqa_train,
        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        transform_test,
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=False
    )

    return train_loader, test_loader


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


def main() -> None:
    pass

if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    main()
