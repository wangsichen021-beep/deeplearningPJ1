[README.md](https://github.com/user-attachments/files/28131508/README.md)
# deeplearningPJ1# Project 1 实验报告：基于 NumPy 的 MNIST 分类

姓名：待填写  
学号：待填写  
代码链接：待填写  
模型权重链接：待填写；本地 checkpoint 位于 `codes/best_models/`

## 1. 实验目标与实现概述

本项目要求仅使用给定 MNIST 数据集，用 NumPy 实现神经网络的基本组件，并完成 MLP 基线、CNN 模型、两项附加方向和可视化分析。代码主要修改在 `mynn/op.py`、`mynn/models.py`、`mynn/optimizer.py`、`mynn/lr_scheduler.py`、`mynn/runner.py`、`test_train.py`、`test_model.py` 和 `weight_visualization.py`。

核心实现如下：

- `Linear.forward/backward`：实现仿射变换 `XW+b`，并计算 `W`、`b` 和输入的梯度。
- `MultiCrossEntropyLoss`：内部包含数值稳定的 softmax，使用批平均交叉熵，反向传播梯度为 `(p-y)/N`。
- `conv2D`：实现 NCHW 格式二维卷积，前向使用 `im2col`，反向计算卷积核、偏置和输入梯度，没有调用深度学习框架。
- `MaxPool2D` 和 `Flatten`：用于构建 CNN。
- `MomentGD`、`MultiStepLR`、`ExponentialLR`：用于优化方向实验。
- `RunnerM`：修正空 batch 问题，按小批量验证并在每次验证后保存最佳模型。

## 2. 数据集与训练设置

使用提供的 MNIST 数据，固定随机种子为 309。训练集按随机排列划分为 50,000 张训练图像和 10,000 张验证图像，测试集为官方 10,000 张测试图像。输入像素统一归一化到 `[0,1]`。

共同设置：

- batch size：64
- 权重衰减：`1e-4`
- 学习率调度：`MultiStepLR`，milestones 为 800 和 1600，`gamma=0.5`
- 验证间隔：每 200 个 iteration
- 训练轮数：3 epoch

各模型的主要差异：

- MLP baseline：`784 -> 600 -> 10`，ReLU，SGD，学习率 0.06。
- CNN baseline：`Conv(1,8,3,pad=1) -> ReLU -> MaxPool(2) -> Conv(8,16,3,pad=1) -> ReLU -> MaxPool(2) -> Flatten -> Linear(16*7*7,64) -> ReLU -> Linear(64,10)`，SGD，学习率 0.02。
- MLP + Momentum：结构与 MLP baseline 相同，优化器改为 MomentumGD，`mu=0.9`，学习率 0.03。

## 3. MLP Baseline

MLP 在 3 个 epoch 后达到较稳定的性能。最佳验证准确率为 93.60%，测试准确率为 93.96%。学习曲线保存在：

`best_models/mlp_baseline/learning_curve.png`

MLP 的优点是实现简单、训练速度较快；缺点是直接把 28x28 图像展平成向量，破坏了局部空间结构，也没有参数共享，因此对笔画平移、局部形变的利用不如 CNN。

## 4. CNN 模型与 MLP 对比

CNN baseline 的最佳验证准确率为 96.24%，测试准确率为 96.64%，高于同样训练轮数下的 MLP baseline。学习曲线保存在：

`best_models/cnn_baseline/learning_curve.png`

CNN 表现更好的主要原因是：

- 卷积核只连接局部邻域，更符合手写数字由局部笔画组成的结构。
- 参数共享使模型在不同位置检测相似笔画，减少了参数量并提升泛化能力。
- 池化降低空间分辨率，增强了对轻微平移和形变的稳定性。

虽然本实验中的 CNN 参数量比 MLP 更少，但仍取得更高精度，说明归纳偏置比单纯增大参数量更重要。

## 5. 两项附加方向

### 5.1 优化方向：MomentumGD

在 MLP 上将 SGD 替换为 MomentumGD 后，最佳验证准确率从 93.60% 提升到 96.81%，测试准确率从 93.96% 提升到 97.10%。动量项累积历史梯度方向，使参数更新在稳定下降方向上更快，在震荡方向上更平滑，因此前期收敛速度和最终效果都有改善。

学习曲线保存在：

`best_models/mlp_momentum/learning_curve.png`

### 5.2 错误分析与可视化

使用 CNN baseline 在测试集上生成混淆矩阵、错误样例和卷积核可视化，输出在：

- `analysis_outputs/cnn_baseline/confusion_matrix.png`
- `analysis_outputs/cnn_baseline/misclassified_examples.png`
- `analysis_outputs/cnn_baseline/weights_or_kernels.png`

从混淆矩阵看，类别 0、1、6 的识别最稳定；类别 8、9 相对更难。主要混淆包括 `7 -> 2`、`5 -> 3`、`7 -> 9`、`4 -> 9`、`2 -> 8` 等。这些错误通常来自笔画连接方式相似、书写倾斜或局部笔画缺失。卷积核可视化显示第一层卷积核已经学习到不同方向和局部明暗变化的响应模式，可解释为基础笔画检测器。

## 6. 主要结果表

| 模型 | 优化器 | 验证最好准确率 | 测试 loss | 测试准确率 | checkpoint |
| --- | --- | ---: | ---: | ---: | --- |
| MLP baseline | SGD | 93.60% | 0.213552 | 93.96% | `best_models/mlp_baseline/best_model.pickle` |
| CNN baseline | SGD | 96.24% | 0.111818 | 96.64% | `best_models/cnn_baseline/best_model.pickle` |
| MLP + Momentum | MomentumGD | 96.81% | 0.097426 | 97.10% | `best_models/mlp_momentum/best_model.pickle` |

## 7. 讨论

本实验完成了从基础 MLP 到 CNN 的完整 NumPy 实现。MLP 可以作为可靠基线，但它忽略图像二维结构，泛化能力受限；CNN 通过局部连接、参数共享和池化更适合图像分类，即使参数更少也能获得更高测试精度。

优化方向中，MomentumGD 是最有信息量的改动。它没有改变模型表达能力，只改变更新方式，却显著提升了 MLP 的收敛速度和最终准确率，说明优化器对同一模型的训练质量影响很大。

错误分析显示，模型仍然容易混淆书写形状接近的数字，例如 7/2、5/3、4/9 和 2/8。后续可以继续尝试轻量数据增强，如小角度旋转和平移，来提升对书写变形的鲁棒性。

## 8. 复现实验命令

MLP baseline：

```bash
python test_train.py --model mlp --epochs 3 --batch-size 64 --eval-batch-size 512 --log-iters 200 --eval-iters 200 --save-dir ./best_models/mlp_baseline --lr 0.06 --optimizer sgd --scheduler multistep --milestones 800,1600 --gamma 0.5 --weight-decay 0.0001
```

CNN baseline：

```bash
python test_train.py --model cnn --epochs 3 --batch-size 64 --eval-batch-size 512 --log-iters 200 --eval-iters 200 --save-dir ./best_models/cnn_baseline --lr 0.02 --optimizer sgd --scheduler multistep --milestones 800,1600 --gamma 0.5 --weight-decay 0.0001
```

MLP + Momentum：

```bash
python test_train.py --model mlp --epochs 3 --batch-size 64 --eval-batch-size 512 --log-iters 200 --eval-iters 200 --save-dir ./best_models/mlp_momentum --lr 0.03 --optimizer momentum --scheduler multistep --milestones 800,1600 --gamma 0.5 --weight-decay 0.0001
```

测试模型：

```bash
python test_model.py --model-path ./best_models/cnn_baseline/best_model.pickle --batch-size 512
```

生成分析图：

```bash
python weight_visualization.py --model-path ./best_models/cnn_baseline/best_model.pickle --batch-size 512 --output-dir ./analysis_outputs/cnn_baseline
```
