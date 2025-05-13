#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation
from torchvision.models.segmentation.fcn import FCN_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights


def get_fcn_model(num_classes=1, pretrained=True):
    """
    获取预训练的FCN模型
    
    参数:
        num_classes (int): 分割类别数量，对于二分类分割任务，为1
        pretrained (bool): 是否使用预训练权重
    
    返回:
        FCN模型
    """
    # 加载预训练的FCN ResNet50模型，使用新的weights参数
    if pretrained:
        weights = FCN_ResNet50_Weights.DEFAULT
    else:
        weights = None
    
    model = segmentation.fcn_resnet50(weights=weights)
    
    # 修改最后一层以适应我们的分割任务
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    # 如果是二分类，使用Sigmoid激活函数
    if num_classes == 1:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    return model


def get_deeplabv3_model(num_classes=1, pretrained=True):
    """
    获取预训练的DeepLabV3模型
    
    参数:
        num_classes (int): 分割类别数量，对于二分类分割任务，为1
        pretrained (bool): 是否使用预训练权重
    
    返回:
        DeepLabV3模型
    """
    # 加载预训练的DeepLabV3 ResNet50模型，使用新的weights参数
    if pretrained:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
    else:
        weights = None
    
    model = segmentation.deeplabv3_resnet50(weights=weights)
    
    # 修改最后一层以适应我们的分割任务
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    # 如果是二分类，修改辅助分类器
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    return model


class SegmentationModel(nn.Module):
    """分割模型包装类，用于输出原始logits以便与BCE with logits一起使用"""
    
    def __init__(self, model_name='fcn', num_classes=1, pretrained=True):
        """
        参数:
            model_name (str): 'fcn' 或 'deeplabv3'
            num_classes (int): 分割类别数量，对于二分类分割任务，为1
            pretrained (bool): 是否使用预训练权重
        """
        super(SegmentationModel, self).__init__()
        
        if model_name.lower() == 'fcn':
            self.model = get_fcn_model(num_classes, pretrained)
        elif model_name.lower() == 'deeplabv3':
            self.model = get_deeplabv3_model(num_classes, pretrained)
        else:
            raise ValueError(f"不支持的模型名称: {model_name}. 请使用 'fcn' 或 'deeplabv3'")
        
        # 用于评估和可视化时应用sigmoid
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """返回模型的原始logits输出"""
        output = self.model(x)
        
        # 对于TorchVision的分割模型，输出是一个字典，包含 'out' 键
        if isinstance(output, dict) and 'out' in output:
            logits = output['out']
            return logits
        else:
            return output
    
    def predict(self, x):
        """使用sigmoid进行预测，用于可视化和评估"""
        with torch.no_grad():
            logits = self.forward(x)
            return self.sigmoid(logits)


if __name__ == "__main__":
    # 测试模型
    model_fcn = SegmentationModel('fcn', num_classes=1, pretrained=True)
    model_deeplabv3 = SegmentationModel('deeplabv3', num_classes=1, pretrained=True)
    
    # 创建随机输入张量
    x = torch.randn(2, 3, 256, 256)
    
    # 前向传播测试
    output_fcn = model_fcn(x)
    output_deeplabv3 = model_deeplabv3(x)
    
    print(f"FCN输出形状: {output_fcn.shape}")
    print(f"DeepLabV3输出形状: {output_deeplabv3.shape}") 