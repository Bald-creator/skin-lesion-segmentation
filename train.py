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
from torch.cuda.amp import GradScaler, autocast

from models import SegmentationModel
from dataset import get_dataloaders, SegmentationTransform
from utils import dice_coefficient, bce_dice_loss, calculate_metrics, save_model, visualize_prediction


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='皮肤病图像分割训练脚本')
    parser.add_argument('--model', type=str, default='fcn', choices=['fcn', 'deeplabv3'],
                        help='使用哪个模型 (fcn 或 deeplabv3)')
    parser.add_argument('--data_path', type=str, default='/home/chentingyu/course/DL_App_Dev',
                        help='数据集根目录路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--img_size', type=int, default=512, help='输入图像大小')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='模型保存目录')
    parser.add_argument('--results_dir', type=str, default='./results', help='结果保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载的工作线程数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用的设备(cpu 或 cuda)')
    parser.add_argument('--amp', action='store_true', help='是否使用混合精度训练')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_one_epoch(model, dataloader, optimizer, criterion, device, clip_value=1.0, use_amp=False, scaler=None):
    """
    训练一个epoch
    
    参数:
        model (nn.Module): 要训练的模型
        dataloader (DataLoader): 训练数据加载器
        optimizer (Optimizer): 优化器
        criterion (callable): 损失函数
        device (str): 设备
        clip_value (float): 梯度裁剪值
        use_amp (bool): 是否使用混合精度训练
        scaler (GradScaler): 混合精度训练的梯度缩放器
    
    返回:
        平均损失和指标
    """
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    # 记录第一个batch的信息帮助debug
    first_batch = True
    
    for images, masks in progress_bar:
        # 将数据移动到指定设备
        images = images.to(device)
        masks = masks.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        
        # 使用混合精度训练
        if use_amp:
            with autocast():
                outputs = model(images)
                
                # 在第一个batch输出诊断信息
                if first_batch:
                    print(f"\nDiagnostic info for first batch:")
                    print(f"Images shape: {images.shape}, dtype: {images.dtype}")
                    print(f"Masks shape: {masks.shape}, dtype: {masks.dtype}")
                    print(f"Outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
                    print(f"Outputs min: {outputs.min().item()}, max: {outputs.max().item()}")
                    print(f"Using AMP: {use_amp}")
                    first_batch = False
                
                loss = criterion(outputs, masks)
                
            # 使用scaler放大损失并进行反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪，防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            # 使用scaler更新参数
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            
            # 在第一个batch输出诊断信息
            if first_batch:
                print(f"\nDiagnostic info for first batch:")
                print(f"Images shape: {images.shape}, dtype: {images.dtype}")
                print(f"Masks shape: {masks.shape}, dtype: {masks.dtype}")
                print(f"Outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
                print(f"Outputs min: {outputs.min().item()}, max: {outputs.max().item()}")
                print(f"Using AMP: {use_amp}")
                first_batch = False
            
            # 计算损失
            loss = criterion(outputs, masks)
            
            # 检查损失是否为NaN
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
        
        # 计算度量
        with torch.no_grad():
            # 应用sigmoid用于计算度量
            predictions = torch.sigmoid(outputs)
            dice = dice_coefficient(masks, (predictions > 0.5).float())
        
        # 更新统计信息
        epoch_loss += loss.item()
        epoch_dice += dice.item()
        
        # 更新进度条
        progress_bar.set_postfix(loss=loss.item(), dice=dice.item())
    
    # 计算平均值
    num_batches = len(dataloader)
    avg_loss = epoch_loss / num_batches
    avg_dice = epoch_dice / num_batches
    
    return avg_loss, avg_dice


def validate(model, dataloader, criterion, device):
    """
    在验证集上验证模型
    
    参数:
        model (nn.Module): 要验证的模型
        dataloader (DataLoader): 验证数据加载器
        criterion (callable): 损失函数
        device (str): 设备
    
    返回:
        平均损失和指标
    """
    model.eval()
    val_loss = 0
    val_metrics = {'dice': 0, 'hausdorff': 0}
    num_samples = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            # 将数据移动到指定设备
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播获取logits
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
            
            # 应用sigmoid获取预测
            predictions = torch.sigmoid(outputs) 
            
            # 计算指标
            batch_metrics = calculate_metrics(masks, predictions)
            val_metrics['dice'] += batch_metrics['dice'] * images.size(0)
            
            # 对于Hausdorff距离，我们只考虑有效值
            if not np.isnan(batch_metrics['hausdorff']):
                val_metrics['hausdorff'] += batch_metrics['hausdorff'] * images.size(0)
            
            num_samples += images.size(0)
    
    # 计算平均值
    avg_loss = val_loss / num_samples
    avg_metrics = {k: v / num_samples for k, v in val_metrics.items()}
    
    return avg_loss, avg_metrics


def train_model(args):
    """
    训练分割模型的主函数
    
    参数:
        args: 命令行参数
    """
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 初始化混合精度训练的scaler
    scaler = GradScaler() if args.amp and torch.cuda.is_available() else None
    if args.amp and torch.cuda.is_available():
        print("Using Automatic Mixed Precision (AMP) training")
    
    # 获取数据加载器
    train_transform = SegmentationTransform()
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        train_transform=train_transform,
        target_size=(args.img_size, args.img_size),
        num_workers=args.num_workers
    )
    print(f"Training set size: {len(train_loader.dataset)}, Validation set size: {len(val_loader.dataset)}, Test set size: {len(test_loader.dataset)}")
    
    # 创建模型
    model = SegmentationModel(args.model, num_classes=1, pretrained=True)
    model = model.to(device)
    print(f"Using model: {args.model}")
    
    # 使用较低的初始学习率
    initial_lr = args.lr if args.lr <= 1e-4 else 1e-4
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    criterion = bce_dice_loss
    
    # 用于记录训练过程
    train_losses = []
    train_dices = []
    val_losses = []
    val_dices = []
    val_hausdorffs = []
    best_val_dice = 0
    
    # 训练循环
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # 训练一个epoch
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device, 
            clip_value=1.0, use_amp=args.amp, scaler=scaler
        )
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        
        # 在验证集上验证
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_dices.append(val_metrics['dice'])
        val_hausdorffs.append(val_metrics['hausdorff'])
        
        # 调整学习率
        scheduler.step(val_metrics['dice'])
        
        # 打印结果
        print(f"Training Loss: {train_loss:.4f}, Training Dice: {train_dice:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Dice: {val_metrics['dice']:.4f}, Validation Hausdorff: {val_metrics['hausdorff']:.4f}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            save_model(model, args.save_dir, f"{args.model}_best")
            print(f"Saved best model, Validation Dice: {best_val_dice:.4f}")
    
    # 保存最终模型
    save_model(model, args.save_dir, f"{args.model}_final")
    
    # 绘制训练过程
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(2, 2, 2)
    plt.plot(train_dices, label='Training Dice')
    plt.plot(val_dices, label='Validation Dice')
    plt.legend()
    plt.title('Dice Coefficient Curve')
    
    plt.subplot(2, 2, 3)
    plt.plot(val_hausdorffs, label='Validation Hausdorff')
    plt.legend()
    plt.title('Hausdorff Distance Curve')
    
    plt.tight_layout()
    os.makedirs(args.results_dir, exist_ok=True)  # 确保目录存在
    plt.savefig(os.path.join(args.results_dir, f"{args.model}_training_curves.png"))
    
    # 可视化一些验证集结果
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(val_loader))
        images = images.to(device)
        
        # 获取原始logits
        outputs = model(images)
        
        # 应用sigmoid获取预测
        predictions = torch.sigmoid(outputs)
        
        for i in range(min(3, len(images))):
            visualize_prediction(
                images[i], masks[i], 
                (predictions[i] > 0.5).float(),
                save_path=os.path.join(args.results_dir, f"{args.model}_sample_{i}.png")
            )
    
    print(f"Training completed! Best Validation Dice: {best_val_dice:.4f}")
    return model


if __name__ == "__main__":
    args = parse_args()
    model = train_model(args) 