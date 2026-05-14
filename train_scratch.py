import os
import random
import numpy as np

# PyTorch 核心库
import torch
import torch.nn as nn
import torch.optim as optim

# Torchvision 视觉工具库
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset

# 实验记录与可视化工具
import wandb
import timm

# 检查是否有可用的 GPU 进行加速，如果没有则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备是: {device}")

def set_seed(seed=42):
    """
    固定所有的随机种子，确保实验结果完全可以复现
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多GPU
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法一致
    torch.backends.cudnn.benchmark = False

# 运行固定种子的函数
set_seed(42)
print("随机种子已固定为 42")

# 1. 定义数据预处理（Data Augmentation & Normalization）
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# 2. 加载 Oxford-IIIT Pet 数据集
full_train_dataset = datasets.OxfordIIITPet(
    root='./data', split='trainval', download=True, transform=data_transforms['train']
)
full_val_dataset = datasets.OxfordIIITPet(
    root='./data', split='trainval', download=False, transform=data_transforms['val']
)

# 3. 划分训练集和验证集 (利用索引)
dataset_size = len(full_train_dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices) # 因为前面固定了随机种子，这里的打乱每次都一样，保证可复现

train_size = int(0.8 * dataset_size)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# 使用 Subset 获取对应的数据，现在它们各自拥有正确的 transform 了！
train_data = Subset(full_train_dataset, train_indices)
val_data = Subset(full_val_dataset, val_indices)

# 4. 创建 DataLoader (批处理加载器)
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")
print(f"类别数量: 37 (Oxford-IIIT Pet 标准类别数)")



# ================= 第三阶段：构建模型与优化器 =================

# 1. 加载预训练模型 (Baseline)
print("正在加载随机初始化的 ResNet-18 模型 (消融实验)...")
model = models.resnet18(weights=None)

# 2. 修改输出层 (全连接层)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 37)

model = model.to(device)

# 3. 差异化学习率设置
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. 定义损失函数：交叉熵损失 (CrossEntropyLoss)
criterion = nn.CrossEntropyLoss()

print("模型构建完成，差异化优化器已设置！")

# ================= 第四阶段：训练循环与可视化 =================

# 1. 初始化 WandB
wandb.init(
    project="pet-classification",
    name="resnet18-scratch",
    config={
        "learning_rate": 1e-3,
        "epochs": 10,
        "batch_size": batch_size,
        "architecture": "ResNet-18 (Scratch)",
        "dataset": "Oxford-IIIT Pet"
    }
)

num_epochs = wandb.config.epochs
best_val_acc = 0.0

# 开始循环训练
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    #训练阶段
    model.train()
    train_loss = 0.0
    train_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1) # 获取预测概率最高的类别
        loss = criterion(outputs, labels)

        # 反向传播 与 参数更新
        loss.backward()
        optimizer.step()

        # 统计数据
        train_loss += loss.item() * inputs.size(0)
        train_corrects += torch.sum(preds == labels.data)

    epoch_train_loss = train_loss / len(train_data)
    epoch_train_acc = train_corrects.double() / len(train_data)

    #验证阶段
    model.eval()
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    epoch_val_loss = val_loss / len(val_data)
    epoch_val_acc = val_corrects.double() / len(val_data)

    print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')
    print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')

    # 2. 将数据同步到 WandB 网页端
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": epoch_train_loss,
        "train_acc": epoch_train_acc,
        "val_loss": epoch_val_loss,
        "val_acc": epoch_val_acc
    })

    # 3. 保存最佳模型
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), 'best_resnet18_scratch.pth')
        print(f"🌟 发现更好的模型！已保存权重到 best_resnet18_scratch.pth")

# 训练结束
wandb.finish()
print(f"训练完成！最佳验证集准确率为: {best_val_acc:.4f}")