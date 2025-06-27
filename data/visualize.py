import numpy as np
import matplotlib.pyplot as plt

# ==================== 可视化函数 ====================


def visualize_dataset(dataset: tuple[list[np.ndarray], np.ndarray], title: str, num_samples: int = 10):
    """可视化数据集样本"""
    images, labels = dataset

    plt.figure(figsize=(15, 3))
    plt.suptitle(f"{title} (First {num_samples} samples | Total: {len(images)} images)", fontsize=14)

    for i in range(min(num_samples, len(images))):
        plt.subplot(1, num_samples, i + 1)

        # 处理不同通道数的图像
        if images[i].ndim == 2:  # 灰度图
            plt.imshow(images[i], cmap="gray")
        else:  # 彩色图
            plt.imshow(images[i])

        plt.title(f"Label: {labels[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
