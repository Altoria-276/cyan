import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from .visualize import visualize_dataset


# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parent


class ParquetImageLoader:
    def __init__(
        self,
        image_field: str = "image",
        bytes_key: str = "bytes",
        label_field: str = "label",
        target_size: tuple[int, int] | None = None,
        normalize: bool = True,
        image_mode: str = "RGB",
        dtype=np.float32,
    ):
        """
        初始化 Parquet 图像加载器

        :param image_field: DataFrame 中包含图像字典的列名
        :param bytes_key: 图像字典中字节数据的键名
        :param label_field: 标签列名
        :param target_size: 目标图像尺寸 (height, width)，None 表示保持原始尺寸
        :param normalize: 是否归一化到 [0, 1]
        :param image_mode: PIL 图像模式 (如 'L', 'RGB', 'RGBA')
        :param dtype: 输出数组的数据类型
        """
        self.image_field = image_field
        self.bytes_key = bytes_key
        self.label_field = label_field
        self.target_size = target_size
        self.normalize = normalize
        self.image_mode = image_mode
        self.dtype = dtype

    def load_parquet(self, file_path: str | Path) -> pd.DataFrame:
        """
        加载 Parquet 文件
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        logger.info(f"Loading parquet file: {path}")
        return pd.read_parquet(path)

    def _extract_bytes(self, row) -> bytes:
        """
        从行数据中提取图像字节
        """
        img_dict = row.get(self.image_field)
        if not isinstance(img_dict, dict):
            raise ValueError(f"Field '{self.image_field}' is not a dictionary")

        img_bytes = img_dict.get(self.bytes_key)
        if not isinstance(img_bytes, bytes):
            raise ValueError(f"Key '{self.bytes_key}' not found or not bytes")

        return img_bytes

    def bytes_to_np(self, img_bytes: bytes) -> np.ndarray:
        """
        将字节数据转换为 NumPy 数组

        :return: 图像数组 (H x W x C)
        """
        try:
            # 从字节数据创建 PIL 图像
            img = Image.open(BytesIO(img_bytes))

            # 转换为指定模式
            if img.mode != self.image_mode:
                img = img.convert(self.image_mode)

            # 调整尺寸（如果需要）
            if self.target_size:
                img = img.resize(self.target_size)

            # 转换为 NumPy 数组 (H x W x C)
            np_img = np.array(img, dtype=self.dtype)

            # 归一化
            if self.normalize and np_img.dtype != np.uint8:
                np_img = np_img.astype(self.dtype)
                np_img /= 255.0

            return np_img

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            # 创建占位符图像
            h, w = self.target_size or (32, 32)
            c = 1 if self.image_mode == "L" else 3
            return np.zeros((h, w, c), dtype=self.dtype)

    def process_dataframe(self, df: pd.DataFrame, max_samples: int | None = None) -> tuple[list[np.ndarray], np.ndarray]:
        """
        处理 DataFrame 并返回图像数组和标签数组

        :param max_samples: 最大处理样本数（None 表示处理所有）
        :return: (图像数组列表, 标签数组)
        """
        images = []
        labels = []

        # 限制处理样本数
        subset = df if max_samples is None else df.iloc[:max_samples]
        total = len(subset)

        logger.info(f"Processing {total} images...")

        for idx, (_, row) in enumerate(subset.iterrows()):
            try:
                # 提取字节数据
                img_bytes = self._extract_bytes(row)

                # 转换为数组
                np_img = self.bytes_to_np(img_bytes)

                # 收集结果
                images.append(np_img)
                labels.append(row[self.label_field])

                # 进度日志
                if (idx + 1) % 1000 == 0 or (idx + 1) == total:
                    logger.info(f"Processed {idx + 1}/{total} images")

            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                continue

        # 转换为数组
        label_array = np.array(labels)

        logger.info(f"Successfully processed {len(images)} images")
        return images, label_array

    def create_dataset(self, images: list[np.ndarray], labels: np.ndarray) -> list[tuple[np.ndarray, int]]:
        """创建数据集 (图像数组, 标签) 元组列表"""
        return list(zip(images, labels))


def load_mnist(normalize: bool = True, max_samples: int | None = None) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
    """
    加载 MNIST 数据集
    :param normalize: 是否归一化
    :param max: 最大加载数量 (None 表示加载全部)
    :return: (训练图像, 训练标签, 测试图像, 测试标签)
    """
    train_path = ROOT / "datasets" / "mnist" / "train-00000-of-00001.parquet"
    test_path = ROOT / "datasets" / "mnist" / "test-00000-of-00001.parquet"

    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"MNIST dataset file not found: {path}")

    loader = ParquetImageLoader(
        image_field="image",
        bytes_key="bytes",
        label_field="label",
        target_size=(28, 28),
        normalize=normalize,
        image_mode="L",
        dtype=np.float32,
    )

    train_images, train_labels = loader.process_dataframe(loader.load_parquet(train_path), max_samples=max_samples)
    test_images, test_labels = loader.process_dataframe(loader.load_parquet(test_path), max_samples=max_samples)
    return train_images, train_labels, test_images, test_labels


def load_notmnist(
    normalize: bool = True, max_samples: int | None = None
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
    """
    加载 notMNIST 数据集
    :param normalize: 是否归一化
    :param max: 最大加载数量 (None 表示加载全部)
    :return: (训练图像, 训练标签, 测试图像, 测试标签)
    """
    train_path = ROOT / "datasets" / "notMNIST" / "train-00000-of-00001-66ec62000fa1b4e3.parquet"
    test_path = ROOT / "datasets" / "notMNIST" / "test-00000-of-00001-1136ef192260594c.parquet"

    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"notMNIST dataset file not found: {path}")

    loader = ParquetImageLoader(
        image_field="image",
        bytes_key="bytes",
        label_field="label",
        target_size=(28, 28),
        normalize=normalize,
        image_mode="L",
        dtype=np.float32,
    )

    train_images, train_labels = loader.process_dataframe(loader.load_parquet(train_path), max_samples=max_samples)
    test_images, test_labels = loader.process_dataframe(loader.load_parquet(test_path), max_samples=max_samples)
    return train_images, train_labels, test_images, test_labels


def load_cifar10(
    normalize: bool = True, max_samples: int | None = None
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
    """
    加载 CIFAR-10 数据集
    :param normalize: 是否归一化
    :param max: 最大加载数量 (None 表示加载全部)
    :return: (训练图像, 训练标签, 测试图像, 测试标签)
    """
    train_path = ROOT / "datasets" / "cifar10" / "train-00000-of-00001.parquet"
    test_path = ROOT / "datasets" / "cifar10" / "test-00000-of-00001.parquet"

    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"CIFAR-10 dataset file not found: {path}")

    loader = ParquetImageLoader(
        image_field="img",
        bytes_key="bytes",
        label_field="label",
        target_size=(32, 32),
        normalize=normalize,
        image_mode="RGB",
        dtype=np.float32,
    )

    train_images, train_labels = loader.process_dataframe(loader.load_parquet(train_path), max_samples=max_samples)
    test_images, test_labels = loader.process_dataframe(loader.load_parquet(test_path), max_samples=max_samples)
    return train_images, train_labels, test_images, test_labels


def main():
    """主测试函数"""
    num_samples = 10

    # 使用现代化路径处理
    datasets_dir = ROOT / "datasets"

    if not datasets_dir.exists():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        return

    logger.info(f"Using datasets directory: {datasets_dir}")

    try:
        # 1. 测试 MNIST
        logger.info("\n===== Testing MNIST =====")
        mnist_file = datasets_dir / "mnist/train-00000-of-00001.parquet"
        if mnist_file.exists():
            mnist_images, mnist_labels = load_mnist()[:2]
            logger.info(f"Loaded {len(mnist_images)} MNIST images")
            logger.info(f"Image shape: {mnist_images[0].shape}")
            logger.info(f"Label dtype: {mnist_labels.dtype}")
            logger.info(f"Unique labels: {np.unique(mnist_labels)}")

            # 显示样本
            visualize_dataset((mnist_images, mnist_labels), title="MNIST Samples", num_samples=num_samples)

        else:
            logger.warning(f"MNIST file not found: {mnist_file}")

        # 2. 测试 notMNIST
        logger.info("\n===== Testing notMNIST =====")
        notmnist_file = datasets_dir / "notMNIST/train-00000-of-00001-66ec62000fa1b4e3.parquet"
        if notmnist_file.exists():
            notmnist_images, notmnist_labels = load_notmnist()[:2]
            logger.info(f"Loaded {len(notmnist_images)} notMNIST images")
            logger.info(f"Unique labels: {np.unique(notmnist_labels)}")
            visualize_dataset((notmnist_images, notmnist_labels), title="notMNIST Samples", num_samples=num_samples)
        else:
            logger.warning(f"notMNIST file not found: {notmnist_file}")

        # 3. 测试 CIFAR-10
        logger.info("\n===== Testing CIFAR-10 =====")
        cifar_file = datasets_dir / "cifar10/train-00000-of-00001.parquet"
        if cifar_file.exists():
            cifar_images, cifar_labels = load_cifar10()[:2]
            logger.info(f"Loaded {len(cifar_images)} CIFAR-10 images")
            logger.info(f"Image shape: {cifar_images[0].shape}")
            logger.info(f"Unique labels: {np.unique(cifar_labels)}")

            # 显示样本
            visualize_dataset((cifar_images, cifar_labels), title="CIFAR-10 Samples", num_samples=num_samples)
        else:
            logger.warning(f"CIFAR-10 file not found: {cifar_file}")

    except Exception as e:
        logger.exception("Error in main function")


if __name__ == "__main__":
    main()
