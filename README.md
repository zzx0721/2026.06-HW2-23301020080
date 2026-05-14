# 2026.06-HW2-23301020080
计算机视觉Homework2
微调在ImageNet上预训练的卷积神经网络实现宠物识别


## 项目简介
本项目为计算机视觉课程作业。基于 
**Oxford-IIIT Pet 数据集**（包含 37 类宠物图像），通过 PyTorch 深度学习框架实现了宠物图像分类任务。

项目中包含了完整的模型训练、验证与评估流程。不仅对比了经典卷积神经网络（ResNet-18）与引入自注意力机制的视觉变换器（ViT-Tiny）在小规模数据集上的迁移学习表现，还通过控制变量法进行了深度的超参数分析与预训练消融实验。


**最佳模型在验证集上达到了 94.70% 的准确率。**

---


## 文件结构

* `train.py`：Baseline 模型脚本（ResNet-18，加载 ImageNet 预训练权重，差异化学习率）。

* `train_tune.py`：超参数优化脚本（引入 StepLR 学习率衰减，测试不同 Batch Size，产出全局最佳模型）。

* `train_scratch.py`：消融实验脚本（ResNet-18，随机初始化无预训练权重，统一学习率）。

* `train_vit.py`：进阶架构实验脚本（使用 timm 库加载轻量级 ViT-Tiny 进行统一微调）。

* `requirements.txt`：项目运行所需的环境依赖清单。

---


## 🛠️ 环境配置

**安装依赖包**

```bash
pip install -r requirements.txt
```

*注：本项目使用 WandB (Weights & Biases) 进行训练可视化。首次运行代码时，终端可能会提示输入 WandB 的 API Key，请提前前往 [wandb.ai](https://wandb.ai) 注册、登录并获取 Key。*

---


##  训练与测试

运行baseline模型

```bash
python train.py
```

 运行最优模型 (ResNet-18 最佳超参数组)

```bash
python train_tune.py
```


 运行消融实验（无预训练，观察对比效果）

```bash
python train_scratch.py
```

 运行 ViT 模型

```bash
python train_vit.py
```


---
