import os
import random
import numpy as np

# PyTorch 核心库
import torch
import torch.nn as nn
import torch.optim as optim

# Torchvision 视觉工具库
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset

# 实验记录与可视化工具
import wandb

# 引入 timm 库用于加载 ViT 模型
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 运行固定种子的函数
set_seed(42)
print("随机种子已固定为 42")

# ================= 第一阶段 & 第二阶段：数据预处理与加载 =================

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

full_train_dataset = datasets.OxfordIIITPet(
    root='./data', split='trainval', download=True, transform=data_transforms['train']
)
full_val_dataset = datasets.OxfordIIITPet(
    root='./data', split='trainval', download=False, transform=data_transforms['val']
)

dataset_size = len(full_train_dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)

train_size = int(0.8 * dataset_size)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_data = Subset(full_train_dataset, train_indices)
val_data = Subset(full_val_dataset, val_indices)

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")
print(f"类别数量: 37 (Oxford-IIIT Pet 标准类别数)")

# ================= 第三阶段：构建模型与优化器 (ViT 专属替换部分) =================

print("正在使用 timm 加载预训练的 ViT-Tiny 模型...")
# vit_tiny_patch16_224 代表：Tiny 版本的 ViT，patch 大小为 16x16，输入分辨率 224
# timm 会自动把输出类别改为 37
model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=37)
model = model.to(device)

# 给整个模型设置一个统一且较小的学习率 (1e-4)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 保持使用学习率衰减
from torch.optim import lr_scheduler

scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 多分类任务的标准损失函数
criterion = nn.CrossEntropyLoss()

print("ViT 模型构建完成，优化器已设置！")

# ================= 第四阶段：训练循环与可视化 =================

wandb.init(
    project="pet-classification",
    name="vit-tiny-pretrained",  # 明确标记这是 ViT 实验
    config={
        "learning_rate": 1e-4,
        "epochs": 20,
        "batch_size": batch_size,
        "architecture": "ViT-Tiny",
        "dataset": "Oxford-IIIT Pet"
    }
)

num_epochs = wandb.config.epochs
best_val_acc = 0.0

# 开始循环训练
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch + 1}/{num_epochs}')
    print('-' * 10)

    # ------------------ 训练阶段 ------------------
    model.train()
    train_loss = 0.0
    train_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        train_corrects += torch.sum(preds == labels.data)

    epoch_train_loss = train_loss / len(train_data)
    epoch_train_acc = train_corrects.double() / len(train_data)

    # ------------------ 验证阶段 ------------------
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

    # 更新调度器
    scheduler.step()
    # 因为现在只有一个参数组，所以索引是 0
    current_lr = optimizer.param_groups[0]['lr']
    print(f"当前统一学习率: {current_lr:.6f}")

    # 将数据同步到 WandB 网页端
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": epoch_train_loss,
        "train_acc": epoch_train_acc,
        "val_loss": epoch_val_loss,
        "val_acc": epoch_val_acc
    })

    # 保存最佳模型
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), 'best_vit_pet.pth')
        print(f"🌟 发现更好的模型！已保存权重到 best_vit_pet.pth")

# 训练结束
wandb.finish()
print(f"训练完成！ViT-Tiny 最佳验证集准确率为: {best_val_acc:.4f}")