# 皮肤病图像分割项目

本项目实现了基于深度学习的皮肤病图像分割，使用了FCN和DeepLabV3两种分割网络。

## 项目结构

- `dataset.py`: 自定义数据集类
- `models.py`: FCN和DeepLabV3模型定义
- `train.py`: 训练脚本
- `evaluate.py`: 评估脚本
- `utils.py`: 工具函数
- `requirements.txt`: 项目依赖

## 数据集

使用ISIC 2016挑战赛数据集进行皮肤病图像分割。

## 评估指标

- DICE系数: 测量分割结果与真值之间的重叠度
- Hausdorff Distance: 测量分割边缘与真值边缘之间的最大距离

## 使用方法

1. 安装依赖:
```
pip install -r requirements.txt
```

2. 训练模型:
```
python train.py --model fcn --data_path path/to/dataset
```

3. 评估模型:
```
python evaluate.py --model fcn --model_path path/to/model --data_path path/to/test_data
```

## 结果

比较了FCN和DeepLabV3两种模型在ISIC 2016数据集上的分割效果，详细结果可参见评估结果部分。 