#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
import random


class ISICDataset(Dataset):
    """ISIC皮肤病分割数据集"""
    
    def __init__(self, root_dir, split='train', transform=None, target_size=(256, 256)):
        """
        参数:
            root_dir (str): 数据集根目录路径
            split (str): 'train' 或 'test' 用于指定使用训练集还是测试集
            transform (callable, optional): 可选的数据增强操作
            target_size (tuple): 调整图像大小到指定尺寸
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        if split == 'train':
            self.img_dir = os.path.join(root_dir, 'ISIC_Challenge_2016', 'Part1', 'ISBI2016_ISIC_Part1_Training_Data')
            self.mask_dir = os.path.join(root_dir, 'ISIC_Challenge_2016', 'Part1', 'ISBI2016_ISIC_Part1_Training_GroundTruth')
        else:
            self.img_dir = os.path.join(root_dir, 'ISIC_Challenge_2016', 'Part1', 'ISBI2016_ISIC_Part1_Test_Data')
            self.mask_dir = os.path.join(root_dir, 'ISIC_Challenge_2016', 'Part1', 'ISBI2016_ISIC_Part1_Test_GroundTruth')
        
        # 获取图像和掩码的路径
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        # 确保每张图像都有对应的mask
        self.valid_img_paths = []
        self.mask_paths = []
        
        for img_path in self.img_paths:
            img_id = os.path.basename(img_path).split('.')[0]
            mask_path = os.path.join(self.mask_dir, f"{img_id}_Segmentation.png")
            if os.path.exists(mask_path):
                self.valid_img_paths.append(img_path)
                self.mask_paths.append(mask_path)
                
        # 基本的图像变换
        self.basic_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.valid_img_paths)
    
    def __getitem__(self, idx):
        # 读取图像和掩码
        img_path = self.valid_img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # 数据增强（仅用于训练）
        if self.transform and self.split == 'train':
            # 应用自定义的数据增强
            image, mask = self.transform(image, mask)
        
        # 应用基本变换
        image = self.basic_transform(image)
        mask = self.mask_transform(mask)
        
        # 二值化掩码 (0 or 1)
        mask = (mask > 0.5).float()
        
        return image, mask


class SegmentationTransform:
    """用于分割任务的数据增强类"""
    
    def __init__(self, flip_prob=0.5, rotate_prob=0.3, rotate_degree=20):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotate_degree = rotate_degree
    
    def __call__(self, image, mask):
        # 随机水平翻转
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)
        
        # 随机垂直翻转
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            mask = F.vflip(mask)
        
        # 随机旋转
        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.rotate_degree, self.rotate_degree)
            image = F.rotate(image, angle, fill=(0,))
            mask = F.rotate(mask, angle, fill=(0,), interpolation=InterpolationMode.NEAREST)
        
        return image, mask


def get_dataloaders(data_dir, batch_size=8, train_transform=None, target_size=(256, 256), num_workers=4):
    """获取训练和测试数据加载器
    
    参数:
        data_dir (str): 数据集根目录
        batch_size (int): 批次大小
        train_transform (callable, optional): 训练数据的增强操作
        target_size (tuple): 图像大小
        num_workers (int): 数据加载的工作线程数
        
    返回:
        train_loader, val_loader: 训练和验证数据加载器
    """
    from torch.utils.data import DataLoader, random_split
    
    # 默认数据增强
    if train_transform is None:
        train_transform = SegmentationTransform()
    
    # 创建训练数据集
    full_train_dataset = ISICDataset(
        root_dir=data_dir,
        split='train',
        transform=train_transform,
        target_size=target_size
    )
    
    # 划分训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # 重置验证集的transform
    val_dataset.dataset.transform = None
    
    # 创建测试数据集
    test_dataset = ISICDataset(
        root_dir=data_dir,
        split='test',
        transform=None,
        target_size=target_size
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集加载
    import matplotlib.pyplot as plt
    
    # 显示图像和掩码的函数
    def show_image_mask(image, mask, title=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # 显示图像
        img = image.permute(1, 2, 0).cpu().numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        ax1.imshow(img)
        ax1.set_title('Image')
        ax1.axis('off')
        
        # 显示掩码
        ax2.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
        ax2.set_title('Mask')
        ax2.axis('off')
        
        if title:
            fig.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..')
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=4)
    
    # 显示一个批次
    images, masks = next(iter(train_loader))
    for i in range(len(images)):
        show_image_mask(images[i], masks[i], f"Sample {i+1}") 