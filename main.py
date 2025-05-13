#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import matplotlib.pyplot as plt

from train import train_model
from evaluate import compare_models


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Skin Lesion Segmentation Project')
    
    # General parameters
    parser.add_argument('--data_path', type=str, default='/home/chentingyu/course/DL_App_Dev',
                        help='Path to dataset root directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    
    # Training parameters
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'evaluate', 'all'],
                        help='Run mode: train, evaluate, or all (train and evaluate)')
    parser.add_argument('--models', type=str, default='all', choices=['fcn', 'deeplabv3', 'all'],
                        help='Models to train/evaluate: fcn, deeplabv3, or all (both)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision training')
    
    # Evaluation parameters
    parser.add_argument('--post_process', action='store_true', help='Whether to use post-processing')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, 'models')
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Training mode
    if args.mode in ['train', 'all']:
        print("\n======= Starting Model Training =======")
        
        # Train FCN
        if args.models in ['fcn', 'all']:
            print("\nTraining FCN model...")
            print(f"FCN model will be saved to: {model_dir}")
            fcn_args = argparse.Namespace(
                model='fcn',
                data_path=args.data_path,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                img_size=args.img_size,
                save_dir=model_dir,
                results_dir=results_dir,
                seed=args.seed,
                num_workers=args.num_workers,
                device=args.device,
                amp=args.amp
            )
            train_model(fcn_args)
            
            # 检查模型是否已保存
            fcn_model_path = os.path.join(model_dir, 'fcn_best.pth')
            if os.path.exists(fcn_model_path):
                print(f"FCN model successfully saved to {fcn_model_path}")
            else:
                print(f"WARNING: FCN model was not saved to {fcn_model_path}")
        
        # Train DeepLabV3
        if args.models in ['deeplabv3', 'all']:
            print("\nTraining DeepLabV3 model...")
            print(f"DeepLabV3 model will be saved to: {model_dir}")
            deeplabv3_args = argparse.Namespace(
                model='deeplabv3',
                data_path=args.data_path,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                img_size=args.img_size,
                save_dir=model_dir,
                results_dir=results_dir,
                seed=args.seed,
                num_workers=args.num_workers,
                device=args.device,
                amp=args.amp
            )
            train_model(deeplabv3_args)
            
            # 检查模型是否已保存
            deeplabv3_model_path = os.path.join(model_dir, 'deeplabv3_best.pth')
            if os.path.exists(deeplabv3_model_path):
                print(f"DeepLabV3 model successfully saved to {deeplabv3_model_path}")
            else:
                print(f"WARNING: DeepLabV3 model was not saved to {deeplabv3_model_path}")
    
    # Evaluation mode
    if args.mode in ['evaluate', 'all']:
        print("\n======= Starting Model Evaluation =======")
        
        # 检查模型文件是否存在
        fcn_model_path = os.path.join(model_dir, 'fcn_best.pth')
        deeplabv3_model_path = os.path.join(model_dir, 'deeplabv3_best.pth')
        
        print(f"Checking for models:")
        if os.path.exists(fcn_model_path):
            print(f"FCN model found at {fcn_model_path}")
        else:
            print(f"WARNING: FCN model not found at {fcn_model_path}")
            
        if os.path.exists(deeplabv3_model_path):
            print(f"DeepLabV3 model found at {deeplabv3_model_path}")
        else:
            print(f"WARNING: DeepLabV3 model not found at {deeplabv3_model_path}")
        
        # 如果至少有一个模型存在，则进行评估
        if os.path.exists(fcn_model_path) or os.path.exists(deeplabv3_model_path):
            eval_args = argparse.Namespace(
                model='fcn' if args.models == 'fcn' else 'deeplabv3',  # Only used for single model evaluation
                data_path=args.data_path,
                batch_size=args.batch_size,
                img_size=args.img_size,
                model_path=None,  # Use best model
                output_dir=results_dir,
                post_process=args.post_process,
                device=args.device,
                num_workers=args.num_workers
            )
            compare_models(eval_args)
        else:
            print("WARNING: No models found for evaluation. Please train models first.")
    
    print("\n======= Execution Completed! =======")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main() 