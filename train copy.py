#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from models import SegmentationModel
from dataset import get_dataloaders, SegmentationTransform
from utils import dice_coefficient, bce_dice_loss, calculate_metrics, save_model, visualize_prediction





if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
        transforms.HorizontalFlip(p=0.5),
        transforms.RandomBrightnessContrast(p=0.2),
        transforms.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        transforms.GridDistortion(p=0.2),
    ])
    
    
    