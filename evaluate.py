#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import time
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models import SegmentationModel
from dataset import get_dataloaders
from utils import calculate_metrics, load_model, visualize_prediction, get_largest_connected_component


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='皮肤病图像分割评估脚本')
    parser.add_argument('--model', type=str, default='fcn', choices=['fcn', 'deeplabv3'],
                        help='使用哪个模型进行评估 (fcn 或 deeplabv3)')
    parser.add_argument('--data_path', type=str, default='/home/chentingyu/course/DL_App_Dev',
                        help='数据集根目录路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小')
    parser.add_argument('--img_size', type=int, default=256, help='输入图像大小')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型权重文件路径，如果未指定则使用最佳模型')
    parser.add_argument('--output_dir', type=str, default='./results', help='结果输出目录')
    parser.add_argument('--post_process', action='store_true', help='是否使用后处理步骤')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用的设备(cpu 或 cuda)')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的工作线程数')
    
    return parser.parse_args()


def evaluate_model(model, test_loader, device, post_process=False):
    """
    评估模型在测试集上的性能
    
    参数:
        model (nn.Module): 要评估的模型
        test_loader (DataLoader): 测试数据加载器
        device (str): 使用的设备
        post_process (bool): 是否进行后处理
        
    返回:
        评估指标字典
    """
    model.eval()
    all_metrics = []
    all_images = []
    all_true_masks = []
    all_pred_masks = []
    
    # 记录推理时间
    total_time = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            # 测量推理时间
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            
            # 应用sigmoid并二值化预测
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()
            
            # 后处理（可选）
            if post_process:
                for i in range(pred_masks.shape[0]):
                    mask_np = pred_masks[i, 0].cpu().numpy()
                    mask_np = get_largest_connected_component(mask_np)
                    pred_masks[i, 0] = torch.from_numpy(mask_np).to(device)
            
            # 计算每个样本的指标
            for i in range(images.shape[0]):
                metrics = calculate_metrics(masks[i:i+1], pred_masks[i:i+1])
                all_metrics.append(metrics)
                
                # 保存一些样本用于可视化
                if len(all_images) < 10:  # 只保存前10个样本
                    all_images.append(images[i].cpu())
                    all_true_masks.append(masks[i].cpu())
                    all_pred_masks.append(pred_masks[i].cpu())
            
            total_time += inference_time
            total_samples += images.shape[0]
    
    # 计算平均指标
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        values = [m[metric] for m in all_metrics if not np.isnan(m[metric])]
        avg_metrics[metric] = np.mean(values)
    
    # 计算平均推理时间
    avg_metrics['inference_time_per_image'] = total_time / total_samples
    
    return avg_metrics, all_images, all_true_masks, all_pred_masks


def compare_models(args):
    """
    比较FCN和DeepLabV3模型的性能
    
    参数:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取数据加载器
    _, _, test_loader = get_dataloaders(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        target_size=(args.img_size, args.img_size),
        num_workers=args.num_workers
    )
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # 评估结果
    results = []
    
    # 评估FCN模型
    print("\nEvaluating FCN model...")
    fcn_model = SegmentationModel('fcn', num_classes=1)
    
    # 更新模型路径：优先检查当前目录/outputs_fcn/models/，如果不存在再尝试其他路径
    possible_fcn_paths = [
        os.path.join('outputs_fcn', 'models', 'fcn_best.pth'),  # 根据run.sh中的输出目录
        os.path.join(args.output_dir, '..', 'models', 'fcn_best.pth'),  # 相对于results目录的models目录
        os.path.join('saved_models', 'fcn_best.pth'),  # 原始路径
    ]
    
    fcn_model_path = args.model_path
    if args.model == 'fcn' and args.model_path:
        fcn_model_path = args.model_path
    else:
        # 尝试所有可能的路径
        for path in possible_fcn_paths:
            if os.path.exists(path):
                fcn_model_path = path
                break
    
    print(f"Looking for FCN model at: {fcn_model_path}")
    
    if fcn_model_path and os.path.exists(fcn_model_path):
        fcn_model = load_model(fcn_model, fcn_model_path)
        fcn_model = fcn_model.to(device)
        
        fcn_metrics, fcn_images, fcn_true_masks, fcn_pred_masks = evaluate_model(
            fcn_model, test_loader, device, args.post_process
        )
        
        results.append({
            'model': 'FCN',
            **fcn_metrics
        })
        
        # 可视化一些FCN结果
        for i in range(min(5, len(fcn_images))):
            visualize_prediction(
                fcn_images[i], fcn_true_masks[i], fcn_pred_masks[i],
                save_path=os.path.join(args.output_dir, f"fcn_test_sample_{i}.png")
            )
        
        print(f"FCN Dice Coefficient: {fcn_metrics['dice']:.4f}")
        print(f"FCN Hausdorff Distance: {fcn_metrics['hausdorff']:.4f}")
        print(f"FCN Average Inference Time: {fcn_metrics['inference_time_per_image']*1000:.2f} ms/image")
    else:
        print(f"FCN model file does not exist: {fcn_model_path}")
    
    # 评估DeepLabV3模型
    print("\nEvaluating DeepLabV3 model...")
    deeplabv3_model = SegmentationModel('deeplabv3', num_classes=1)
    
    # 更新模型路径：优先检查当前目录/outputs_deeplabv3/models/，如果不存在再尝试其他路径
    possible_deeplabv3_paths = [
        os.path.join('outputs_deeplabv3', 'models', 'deeplabv3_best.pth'),  # 根据run.sh中的输出目录
        os.path.join(args.output_dir, '..', 'models', 'deeplabv3_best.pth'),  # 相对于results目录的models目录
        os.path.join('saved_models', 'deeplabv3_best.pth'),  # 原始路径
    ]
    
    deeplabv3_model_path = args.model_path
    if args.model == 'deeplabv3' and args.model_path:
        deeplabv3_model_path = args.model_path
    else:
        # 尝试所有可能的路径
        for path in possible_deeplabv3_paths:
            if os.path.exists(path):
                deeplabv3_model_path = path
                break
    
    print(f"Looking for DeepLabV3 model at: {deeplabv3_model_path}")
    
    if deeplabv3_model_path and os.path.exists(deeplabv3_model_path):
        deeplabv3_model = load_model(deeplabv3_model, deeplabv3_model_path)
        deeplabv3_model = deeplabv3_model.to(device)
        
        deeplabv3_metrics, deeplabv3_images, deeplabv3_true_masks, deeplabv3_pred_masks = evaluate_model(
            deeplabv3_model, test_loader, device, args.post_process
        )
        
        results.append({
            'model': 'DeepLabV3',
            **deeplabv3_metrics
        })
        
        # 可视化一些DeepLabV3结果
        for i in range(min(5, len(deeplabv3_images))):
            visualize_prediction(
                deeplabv3_images[i], deeplabv3_true_masks[i], deeplabv3_pred_masks[i],
                save_path=os.path.join(args.output_dir, f"deeplabv3_test_sample_{i}.png")
            )
        
        print(f"DeepLabV3 Dice Coefficient: {deeplabv3_metrics['dice']:.4f}")
        print(f"DeepLabV3 Hausdorff Distance: {deeplabv3_metrics['hausdorff']:.4f}")
        print(f"DeepLabV3 Average Inference Time: {deeplabv3_metrics['inference_time_per_image']*1000:.2f} ms/image")
    else:
        print(f"DeepLabV3 model file does not exist: {deeplabv3_model_path}")
    
    # 比较结果并绘制
    if len(results) > 0:
        # 创建DataFrame并保存结果
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(args.output_dir, 'model_comparison.csv'), index=False)
        
        # 打印比较表格
        print("\nModel Performance Comparison:")
        print(df.to_string(index=False))
        
        # 绘制结果比较图
        if len(results) > 1:
            models = [r['model'] for r in results]
            dice_scores = [r['dice'] for r in results]
            hausdorff_distances = [r['hausdorff'] for r in results]
            inference_times = [r['inference_time_per_image']*1000 for r in results]  # ms
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.bar(models, dice_scores, color=['blue', 'green'])
            plt.ylabel('Dice Coefficient')
            plt.title('Dice Coefficient Comparison (Higher is Better)')
            plt.ylim(0, 1)
            for i, v in enumerate(dice_scores):
                plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
            
            plt.subplot(1, 3, 2)
            plt.bar(models, hausdorff_distances, color=['blue', 'green'])
            plt.ylabel('Hausdorff Distance')
            plt.title('Hausdorff Distance Comparison (Lower is Better)')
            for i, v in enumerate(hausdorff_distances):
                plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
            
            plt.subplot(1, 3, 3)
            plt.bar(models, inference_times, color=['blue', 'green'])
            plt.ylabel('Inference Time (ms/image)')
            plt.title('Inference Time Comparison (Lower is Better)')
            for i, v in enumerate(inference_times):
                plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'model_comparison.png'))
            
            print(f"\nComparison chart saved to {os.path.join(args.output_dir, 'model_comparison.png')}")


if __name__ == "__main__":
    args = parse_args()
    compare_models(args) 