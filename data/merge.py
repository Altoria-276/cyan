from pathlib import Path
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from .load_parquet import load_mnist, load_notmnist, load_cifar10
from .visualize import visualize_dataset
from torch.utils.data import TensorDataset

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parent


class DatasetMerger:
    """
    数据集合并工具类，支持两种B数据集标签转换模式：
    1. 标签偏移 (True) - 将B数据集标签偏移到新范围
    2. 单类归并 (False) - 将B数据集所有标签归为单一新类

    输入数据集格式: tuple[list[np.ndarray], np.ndarray] (图像列表, 标签数组)
    """

    @staticmethod
    def merge_datasets(
        dataset_a: tuple[list[np.ndarray], np.ndarray],
        dataset_b: tuple[list[np.ndarray], np.ndarray],
        shift_mode: bool,  # True: 标签偏移, False: 单类归并
        a_label_range: int | None = None,
        shuffle: bool = False,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        合并两个数据集，根据布尔参数转换B数据集标签

        Args:
            dataset_a: 数据集A (图像列表, 标签数组)
            dataset_b: 数据集B (图像列表, 标签数组)
            shift_mode: 合并模式 -
                True: 标签偏移模式 (B标签 = 原始值 + A_max+1)
                False: 单类归并模式 (所有B标签设为 A_max+1)
            a_label_range: A数据集的最大标签值。如果为None，则自动从dataset_a中计算

        Returns:
            合并后的数据集 (图像列表, 标签数组)
        """
        images_a, labels_a = dataset_a
        images_b, labels_b = dataset_b

        # 验证输入格式
        DatasetMerger._validate_dataset(dataset_a, "A")
        DatasetMerger._validate_dataset(dataset_b, "B")

        # 计算A数据集的最大标签值
        max_label_a = np.max(labels_a) if a_label_range is None else a_label_range

        # 根据布尔参数转换B数据集标签
        if shift_mode:  # 标签偏移模式
            labels_b = DatasetMerger._shift_labels(labels_b, max_label_a)
        else:  # 单类归并模式
            labels_b = DatasetMerger._convert_to_one_class(labels_b, max_label_a)

        # 合并数据集
        merged_images = images_a + images_b
        merged_labels = np.concatenate((labels_a, labels_b))

        # 新增打乱逻辑
        if shuffle:
            indices = np.arange(len(merged_images))
            np.random.shuffle(indices)
            merged_images = [merged_images[i] for i in indices]
            merged_labels = merged_labels[indices]

        return merged_images, merged_labels

    @staticmethod
    def split_dataset(
        dataset: tuple[list[np.ndarray], np.ndarray], ratio: float = 0.8, shuffle: bool = True, seed: int | None = None
    ) -> tuple[tuple[list[np.ndarray], np.ndarray], tuple[list[np.ndarray], np.ndarray]]:
        """
        将数据集按比例分割为两部分

        Args:
            dataset: 要分割的数据集 (图像列表, 标签数组)
            ratio: 第一部分的比例 (0.0 ~ 1.0)
            shuffle: 是否在分割前打乱数据顺序 (默认为True)
            seed: 随机种子 (可选)

        Returns:
            (part1, part2) - 分割后的两个数据集，格式与输入相同

        Raises:
            ValueError: 如果ratio不在0~1之间
        """
        # 验证输入比例
        if not (0.0 < ratio < 1.0):
            raise ValueError(f"Invalid ratio: {ratio}. Must be between 0.0 and 1.0")

        # 验证数据集格式
        DatasetMerger._validate_dataset(dataset, "input")

        images, labels = dataset
        n_total = len(images)
        n_part1 = int(n_total * ratio)

        # 创建索引列表
        indices = list(range(n_total))

        # 如果需要打乱顺序
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(indices)

        # 分割索引
        indices_part1 = indices[:n_part1]
        indices_part2 = indices[n_part1:]

        # 分割图像和标签
        images_part1 = [images[i] for i in indices_part1]
        labels_part1 = labels[indices_part1]

        images_part2 = [images[i] for i in indices_part2]
        labels_part2 = labels[indices_part2]

        return (images_part1, labels_part1), (images_part2, labels_part2)

    @staticmethod
    def _shift_labels(labels_b: np.ndarray, max_label_a: int) -> np.ndarray:
        """将B数据集标签偏移到新范围 (max_label_a+1 开始)"""
        return labels_b + max_label_a + 1

    @staticmethod
    def _convert_to_one_class(labels_b: np.ndarray, max_label_a: int) -> np.ndarray:
        """将B数据集所有标签转换为单一新类 (值为 max_label_a+1)"""
        return np.full_like(labels_b, max_label_a + 1)

    @staticmethod
    def _validate_dataset(dataset: tuple[list[np.ndarray], np.ndarray], dataset_name: str) -> None:
        """验证数据集格式是否正确"""
        images, labels = dataset

        if not isinstance(images, list) or not all(isinstance(img, np.ndarray) for img in images):
            raise TypeError(f"Dataset {dataset_name} images must be a list of numpy arrays")

        if not isinstance(labels, np.ndarray) or labels.ndim != 1:
            raise TypeError(f"Dataset {dataset_name} labels must be a 1D numpy array")

        if len(images) != len(labels):
            raise ValueError(
                f"Dataset {dataset_name}: number of images ({len(images)}) " f"does not match number of labels ({len(labels)})"
            )


# ==================== 主测试函数 ====================


def main():
    """主测试函数"""
    print("=" * 50)
    print("Starting dataset processing test")
    print("=" * 50)

    # 1. 加载三个数据集
    mnist_dataset = load_mnist(max_samples=20)[:2]
    notmnist_dataset = load_notmnist(max_samples=20)[:2]
    cifar10_dataset = load_cifar10(max_samples=10)[:2]

    # 可视化原始数据集
    visualize_dataset(mnist_dataset, "Original MNIST Dataset")
    visualize_dataset(notmnist_dataset, "Original notMNIST Dataset")
    visualize_dataset(cifar10_dataset, "Original CIFAR-10 Dataset")

    # 2. 将MNIST数据集20%单独划分出来
    print("\nSplitting MNIST dataset...")
    mnist_train, mnist_val = DatasetMerger.split_dataset(mnist_dataset, ratio=0.8, shuffle=True, seed=42)

    print(f"  - MNIST training set size: {len(mnist_train[0])} images")
    print(f"  - MNIST validation set size: {len(mnist_val[0])} images")

    # 3. 将MNIST和notMNIST按照shift方式合并
    print("\nMerging MNIST and notMNIST with shift mode...")
    merged_shift, labels_shift = DatasetMerger.merge_datasets(mnist_train, notmnist_dataset, shift_mode=True)

    # 检查合并结果
    unique_labels = np.unique(labels_shift)
    print(f"  - Merged dataset size: {len(merged_shift)} images")
    print(f"  - Unique labels in merged dataset: {unique_labels}")
    print(f"  - Label range: {np.min(labels_shift)}-{np.max(labels_shift)}")

    # 4. 将CIFAR-10归为单一类
    print("\nMerging with CIFAR-10 as one class...")
    # 注意：这里使用上一步合并后的数据集作为基础
    final_dataset, final_labels = DatasetMerger.merge_datasets(
        (merged_shift, labels_shift), cifar10_dataset, shift_mode=False, shuffle=True
    )

    # 检查最终结果
    unique_labels_final = np.unique(final_labels)
    print(f"  - Final dataset size: {len(final_dataset)} images")
    print(f"  - Unique labels in final dataset: {unique_labels_final}")

    # 可视化最终数据集
    visualize_dataset((final_dataset, final_labels), "Final Merged Dataset", num_samples=8)

    # 5. 验证验证集是否独立
    print("\nValidating MNIST validation set...")
    val_images, val_labels = mnist_val
    print(f"  - Validation set labels: {np.unique(val_labels)}")
    print(f"  - Validation set size: {len(val_images)} images")

    # 检查验证集标签是否在0-9范围内（未被修改）
    if np.all((val_labels >= 0) & (val_labels <= 9)):
        print("  - Validation set labels are intact (0-9)")
    else:
        print("  - ERROR: Validation set labels have been modified!")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
