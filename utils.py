#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from medpy.metric.binary import hd, dc
from skimage.measure import label, regionprops


def dice_coefficient(y_true, y_pred, smooth=1e-5):
    """
    计算Dice系数
    
    参数:
        y_true (torch.Tensor): 真实掩码
        y_pred (torch.Tensor): 预测掩码
        smooth (float): 用于避免零除的平滑值
    
    返回:
        dice系数值
    """
    # 展平张量
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    
    # 检查是否都是零张量
    if torch.sum(y_true) == 0 and torch.sum(y_pred) == 0:
        return torch.tensor(1.0, device=y_true.device)
    
    # 计算交集
    intersection = (y_true * y_pred).sum()
    
    # 计算dice系数
    dice = (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)
    
    # 处理可能的NaN值
    if torch.isnan(dice):
        return torch.tensor(0.0, device=y_true.device)
    
    return dice


def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    计算Dice损失函数
    
    参数:
        y_true (torch.Tensor): 真实掩码
        y_pred (torch.Tensor): 预测掩码
        smooth (float): 用于避免零除的平滑值
    
    返回:
        dice损失值
    """
    return 1 - dice_coefficient(y_true, y_pred, smooth)


def bce_dice_loss(y_true, y_pred, dice_weight=0.5):
    """
    结合二元交叉熵损失和Dice损失，使用安全的binary_cross_entropy_with_logits实现
    
    参数:
        y_true (torch.Tensor): 真实掩码
        y_pred (torch.Tensor): 预测掩码（在训练时为logits）
        dice_weight (float): Dice损失的权重
    
    返回:
        组合损失值
    """
    # 在混合精度训练中，必须使用binary_cross_entropy_with_logits
    # 对于logits输入应用sigmoid
    y_pred_sigmoid = torch.sigmoid(y_pred)
    
    # 计算BCE损失和DICE损失
    bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
    d_loss = dice_loss(y_true, y_pred_sigmoid)
    
    # 组合损失
    loss = bce_loss * (1 - dice_weight) + d_loss * dice_weight
    
    # 防止NaN值
    if torch.isnan(loss):
        # 如果组合损失为NaN，只返回BCE损失，它通常更稳定
        return bce_loss
    
    return loss


def hausdorff_distance(y_true, y_pred, max_dist=100.0):
    """
    计算Hausdorff距离
    
    参数:
        y_true (torch.Tensor): 真实掩码 [batch_size, 1, H, W]
        y_pred (torch.Tensor): 预测掩码 [batch_size, 1, H, W]
        max_dist (float): 最大距离限制，防止极端值影响平均结果
    
    返回:
        平均hausdorff距离
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # 初始化结果列表
    batch_hd = []
    
    # 对批次中的每个样本计算Hausdorff距离
    for i in range(y_true.shape[0]):
        true_mask = y_true[i].squeeze() > 0.5
        pred_mask = y_pred[i].squeeze() > 0.5
        
        # 检查掩码是否包含前景像素
        if np.sum(true_mask) > 0 and np.sum(pred_mask) > 0:
            # 使用medpy的hd计算Hausdorff距离
            try:
                distance = hd(pred_mask, true_mask)
                # 限制极端值
                distance = min(distance, max_dist)
                batch_hd.append(distance)
            except Exception as e:
                print(f"Computing Hausdorff distance error: {e}")
                # 如果计算失败，使用一个保守的最大值
                batch_hd.append(max_dist)
        else:
            # 如果有一个掩码是空的，另一个不是，那么给予较大的惩罚
            if np.sum(true_mask) > 0 or np.sum(pred_mask) > 0:
                batch_hd.append(max_dist)
            # 如果两个掩码都是空的，那么距离为0（完全匹配）
            else:
                batch_hd.append(0.0)
    
    # 如果无法计算任何距离，返回最大距离
    if len(batch_hd) == 0:
        return max_dist
    
    return np.mean(batch_hd)


def calculate_metrics(y_true, y_pred):
    """
    计算各种分割评估指标
    
    参数:
        y_true (torch.Tensor): 真实掩码 [batch_size, 1, H, W]
        y_pred (torch.Tensor): 预测掩码 [batch_size, 1, H, W]
    
    返回:
        指标字典
    """
    # 将预测值二值化
    y_pred_binary = (y_pred > 0.5).float()
    
    # 计算DICE系数
    dice = dice_coefficient(y_true, y_pred_binary).item()
    
    # 计算Hausdorff距离
    hausdorff = hausdorff_distance(y_true, y_pred_binary)
    
    return {
        'dice': dice,
        'hausdorff': hausdorff
    }


def save_model(model, save_path, model_name):
    """
    保存模型
    
    参数:
        model (nn.Module): 要保存的模型
        save_path (str): 保存目录
        model_name (str): 模型名称
    """
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}.pth"))
    print(f"Model saved to {os.path.join(save_path, model_name)}.pth")


def load_model(model, load_path):
    """
    加载模型
    
    参数:
        model (nn.Module): 要加载权重的模型
        load_path (str): 模型权重路径
    
    返回:
        加载了权重的模型
    """
    model.load_state_dict(torch.load(load_path))
    return model


def visualize_prediction(image, true_mask, pred_mask, save_path=None):
    """
    可视化分割结果
    
    参数:
        image (torch.Tensor): 输入图像 [3, H, W]
        true_mask (torch.Tensor): 真实掩码 [1, H, W]
        pred_mask (torch.Tensor): 预测掩码 [1, H, W]
        save_path (str, optional): 保存路径，如果指定，则保存图像
    """
    # 将张量转为numpy数组
    img = image.permute(1, 2, 0).cpu().numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    
    true_mask = true_mask.squeeze().cpu().numpy()
    pred_mask = pred_mask.squeeze().cpu().numpy()
    
    # 绘制图像和掩码
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def get_largest_connected_component(binary_mask):
    """
    从二值掩码中获取最大的连通区域
    
    参数:
        binary_mask (np.ndarray): 二值掩码图像
    
    返回:
        包含最大连通区域的二值掩码
    """
    labeled_mask = label(binary_mask)
    if labeled_mask.max() == 0:  # 如果没有连通区域
        return binary_mask
    
    regions = regionprops(labeled_mask)
    areas = [region.area for region in regions]
    
    if not areas:  # 如果没有区域
        return binary_mask
    
    max_idx = np.argmax(areas)
    largest_mask = (labeled_mask == (max_idx + 1)).astype(np.uint8)
    
    return largest_mask 