import json
from collections import Counter, defaultdict

import math
import os
import argparse
import logging
import sys
import time
import copy
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
import cv2

import embedding


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

    def __init__(self, dataset_path: str, transform, 
            word_embeddings: embedding.WordEmbeddings, mode: str = 'train'):
        self.word_embeddings = word_embeddings
        answer_frequency = defaultdict(int)
        questions_dict = \
        json.loads(open('v2_OpenEnded_mscoco_{}2014_questions.json'.format(mode)).read())['questions']
        annotations_dict = \
        json.loads(open('v2_mscoco_{}2014_annotations.json'.format(mode)).read())['annotations']
        print("JSON loaded.")

        dataset_dict = {}
        while questions_dict:
            question = questions_dict.pop()
            dataset_dict[question['question_id']] = {'question': copy.deepcopy(question['question']), 
                                                        'image_id': copy.deepcopy(question['image_id'])}
        while annotations_dict:
            annotation = annotations_dict.pop()
            dataset_dict[annotation['question_id']]['multiple_choice_answer'] = copy.deepcopy(annotation['multiple_choice_answer'])
            answer_frequency[annotation['multiple_choice_answer']] += 1
        print("Combined questions and answers.")
        del questions_dict, annotations_dict

        top_answers = sorted([(v, k) for k, v in answer_frequency.items()], reverse=True)[:1000]
        print("Done sorting.")
        self.answer_to_idx = {ans: idx for idx, (_, ans) in enumerate(top_answers)}
        del top_answers, answer_frequency
        self.dataset = []
        
        while dataset_dict:
            k, data = dataset_dict.popitem()
            if data['multiple_choice_answer'] in self.answer_to_idx:
                data['answer_index'] = self.answer_to_idx[data['multiple_choice_answer']]
                self.dataset.append((k, data))

        del dataset_dict
        self.mode = mode
        self.dataset_path = dataset_path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        question_id, data = self.dataset[idx]
        question_embedding = torch.tensor([self.word_embeddings.get_embedding(word) for word in data['question']], dtype=torch.float)
        image_id = '{:012d}'.format(data['image_id'])
        year = '2015' if self.mode == 'test' else '2014'
        image_name = "COCO_" + self.mode + year + "_" + image_id + ".jpg"
        # with open(os.path.join(self.dataset_path, image_name), 'rb') as f:
        f = os.path.join(self.dataset_path, image_name)
        img = Image.fromarray(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))

        question_embedding = F.pad(question_embedding, pad=(0, 0, 60-question_embedding.shape[0], 0))
        return self.transform(img), question_embedding, torch.tensor(data['answer_index'])

def get_loaders(train_path: str, test_path:str, batch_size: int = 32,
        test_batch_size: int = 32, num_train_workers: int = 6,
        num_test_workers: int = 8) -> Tuple[DataLoader, DataLoader]:
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

    word_embedding_file = 'glove.6B.300d-relativized.txt'
    word_embeddings = embedding.read_word_embeddings(word_embedding_file)
    vqa_train = VQADataset(train_path, transform_train, word_embeddings, 'train')
    vqa_test = VQADataset(test_path, transform_test, word_embeddings, 'val')

    train_loader = DataLoader(
        vqa_train,
        batch_size=batch_size, shuffle=True, 
        num_workers=num_train_workers, drop_last=True
    )

    test_loader = DataLoader(
        vqa_test,
        batch_size=test_batch_size, shuffle=False, 
        num_workers=num_test_workers, drop_last=False
    )

    return train_loader, test_loader


def accuracy(model: nn.Module, dataset_loader: DataLoader, device) -> float:
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = y.to(device)
        _, predicted = torch.max(model(x), dim=1)
        total_correct += (predicted == y).sum().item()
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

