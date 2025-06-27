import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from data import DatasetMerger, load_mnist, load_notmnist, load_cifar10
from model.ood_detector import EnergyDetector
from torchvision.models import resnet18


def main():
    # 1. 数据准备
    # 加载MNIST并分割
    mnist_all = load_mnist(normalize=True)[:2]
    mnist_train, mnist_val = DatasetMerger.split_dataset(mnist_all, ratio=0.8)

    # 加载其他数据集
    notmnist = load_notmnist(normalize=True, max_samples=2000)[:2]
    cifar10 = load_cifar10(normalize=True, max_samples=2000)[:2]

    # 合并数据集（参考dataset_manager.py的合并逻辑）
    merged, labels = DatasetMerger.merge_datasets(
        mnist_train, DatasetMerger.merge_datasets(notmnist, cifar10, shift_mode=False), shift_mode=True, shuffle=True
    )

    # 2. 模型准备
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # MNIST 10分类

    # 预训练（简化版）
    train_loader = DataLoader(list(zip(merged, labels)), batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        for imgs, lbls in train_loader:
            imgs = imgs.permute(0, 3, 1, 2)  # NHWC -> NCHW
            outputs = model(imgs)
            loss = criterion(outputs, torch.LongTensor(lbls))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 3. OOD检测
    detector = EnergyDetector(model)
    test_data = np.concatenate([mnist_val[0], notmnist[0], cifar10[0]])
    test_labels = np.concatenate(
        [np.zeros(len(mnist_val[0])), np.ones(len(notmnist[0]) + len(cifar10[0]))]  # ID数据标记为0  # OOD数据标记为1
    )

    # 转换为Tensor
    test_tensor = torch.stack([torch.Tensor(x).permute(2, 0, 1) for x in test_data])

    # 计算能量分数
    with torch.no_grad():
        energy_scores = detector.calculate(test_tensor).numpy()

    # 4. 可视化
    plt.figure(figsize=(10, 6))

    # ID数据分布
    plt.hist(energy_scores[test_labels == 0], bins=50, alpha=0.5, label="ID (MNIST)")
    # OOD数据分布
    plt.hist(energy_scores[test_labels == 1], bins=50, alpha=0.5, label="OOD")

    plt.xlabel("Energy Score")
    plt.ylabel("Frequency")
    plt.title("OOD Detection Performance (Energy Method)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
