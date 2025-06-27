import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image  # 添加PIL图像处理
from sklearn.model_selection import train_test_split


class CustomImageDataset(Dataset):
    def __init__(self, images: list[np.ndarray], labels: np.ndarray, transform=None):
        """
        自定义图像数据集类

        参数:
            images: 图像列表，每个元素为 (H, W) 或 (H, W, C) 格式的numpy数组
            labels: 对应标签数组，形状应为 (N,) 的整数数组
            transform: 可选的数据预处理变换操作(默认None)
        """
        self.images = images
        self.labels = labels
        self.transform = transform

        # 数据一致性验证
        if len(images) != len(labels):
            raise ValueError("图像数量必须与标签数量一致")

    def __len__(self):
        """返回数据集大小"""
        return len(self.labels)

    def __getitem__(self, idx):
        """获取单个样本"""
        image = self.images[idx].copy()  # 创建副本避免修改原始数据
        label = self.labels[idx]

        # 将numpy数组转换为PIL图像
        if image.ndim == 2:  # 灰度图像
            pil_image = Image.fromarray(image, mode="L")
        elif image.ndim == 3:  # 彩色图像
            pil_image = Image.fromarray(image, mode="RGB")
        else:
            raise ValueError(f"不支持的图像维度: {image.ndim}")

        # 应用变换（在PIL图像上）
        if self.transform:
            pil_image = self.transform(pil_image)
            image_tensor = pil_image  # 变换后已经是张量
        else:
            # 如果没有变换，手动转换为张量
            image_tensor = torch.from_numpy(image).float()
            if image_tensor.ndim == 2:  # 灰度图
                image_tensor = image_tensor.unsqueeze(0)  # 添加通道维度
            elif image_tensor.ndim == 3:  # 彩色图
                image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW

        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


# 在文件顶部添加常量定义
NORMALIZATION_MEAN_STD = {
    "mnist": ((0.1307,), (0.3081,)),
    "notmnist": ((0.1307,), (0.3081,)),
    "cifar10": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def convert_gray_to_rgb(tensor):
    """
    将单通道灰度张量转换为三通道RGB张量
    通过复制灰度通道到所有三个RGB通道
    """
    return tensor.repeat(3, 1, 1)  # 在通道维度重复3次


def create_dataloaders(
    images,
    labels,
    test_size: float = 0.2,
    batch_size: int = 32,
    random_state: int = 520,
    dataset_name: str | None = "mnist",
    input_size: int = 32,
    transform: transforms.Compose | None = None,
):
    """
    创建训练集和验证集的数据加载器

    参数:
        images: 输入图像列表
        labels: 对应标签数组
        test_size: 验证集划分比例(默认0.2)
        batch_size: 批量大小(默认32)
        random_state: 随机种子(默认520)
        dataset_name: 数据集名称 ["mnist", "notmnist", "cifar10"](默认"mnist")
        input_size: 输入图像尺寸(默认32)
        transform: 自定义数据预处理流程(默认None时自动创建)

    返回:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 全局定义灰度转RGB函数，避免Windows多进程序列化问题

    # 预处理配置
    if transform is None and dataset_name:
        # 获取归一化参数
        mean, std = NORMALIZATION_MEAN_STD[dataset_name]

        # 确定是否为灰度数据集
        convert_gray = dataset_name in ["mnist", "notmnist"]

        # 对于灰度数据集，扩展归一化参数
        if convert_gray and len(mean) == 1:
            mean = mean * 3  # 扩展为三通道
            std = std * 3  # 扩展为三通道

        # 创建变换步骤列表
        transform_steps = []

        # 1. 调整尺寸（在PIL图像上操作）
        transform_steps.append(transforms.Resize((input_size, input_size)))

        # 2. 转换为张量
        transform_steps.append(transforms.ToTensor())

        # 3. 灰度转RGB（仅限灰度数据集）
        if convert_gray:
            transform_steps.append(transforms.Lambda(convert_gray_to_rgb))

        # 4. 归一化
        transform_steps.append(transforms.Normalize(mean, std))

        # 组合所有变换
        transform = transforms.Compose(transform_steps)

    # 数据集划分
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # 创建数据集实例
    train_dataset = CustomImageDataset(train_images, train_labels, transform=transform)
    val_dataset = CustomImageDataset(val_images, val_labels, transform=transform)

    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if cuda_available else 0,
        pin_memory=cuda_available,
        persistent_workers=cuda_available,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if cuda_available else 0,
        pin_memory=cuda_available,
        persistent_workers=cuda_available,
    )

    return train_loader, val_loader
