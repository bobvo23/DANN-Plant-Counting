
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from base import BaseDataLoader
from utils import MNISTM
from utils import *


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Mnist_M_DataLoader(BaseDataLoader):
    """
    MNIST_M data loading using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.29730626, 0.29918741, 0.27534935),
                                                         (0.32780124, 0.32292358, 0.32056796))])

        self.data_dir = data_dir
        self.dataset = MNISTM(root='data/MNIST-M', train=training, download=True,
                              transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CVPPP_DataLoader(BaseDataLoader):
    """
    CVPPP data loading using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        transform = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(
        ), transforms.Resize((256, 256)), transforms.ToTensor()])
        self.data_dir = data_dir
        self.dataset = CVPPP(
            root='data/CVPPP', train=training, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class KOMATSUNA_DataLoader(BaseDataLoader):
    """
    KOMATSUNA data loading using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        transform = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(
        ), transforms.Resize((256, 256)), transforms.ToTensor()])
        self.data_dir = data_dir
        self.dataset = KOMATSUNA(
            root='data/KOMATSUNA', train=training, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
