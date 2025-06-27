from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseOODDetector(ABC):
    def __init__(self, model: nn.Module, threshold: tuple[float]):
        self.model = model
        self.model.eval()
        self.threshold: tuple[float] = threshold

    @abstractmethod
    def calculate(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 OOD 分数（分数越高表示 OOD 可能性越大）

        参数:
            x (torch.Tensor): 输入数据 [batch_size, ...]

        返回:
            torch.Tensor: OOD 分数 [batch_size]
        """
        pass

    def detect(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        OOD 检测函数（更新为多级分类输出）

        参数:
            x (torch.Tensor): 输入数据 [batch_size, ...]

        返回:
            Tuple[torch.Tensor, torch.Tensor]:
                - OOD 分类 [batch_size]
                - OOD 分数 [batch_size]
        """
        ood_scores = self.calculate(x)

        # 使用边界查找确定分类级别
        thresholds_tensor = torch.tensor(self.threshold, device=ood_scores.device)
        ood_flags = torch.bucketize(ood_scores, thresholds_tensor, right=True)

        return ood_flags, ood_scores

    def calibrate_threshold(self, inputs: torch.Tensor, percentile: float = 0.9) -> float:
        """
        在 ID 数据上校准阈值（输入改为Tensor格式）

        参数:
            inputs (torch.Tensor): 输入数据张量 [batch_size, ...]
            percentile: 0-1 (0.9 表示 FPR=10%)

        返回:
            校准后的阈值
        """
        with torch.no_grad():
            # 确保输入在正确的设备上
            inputs = inputs.to(next(self.model.parameters()).device)
            scores = self.calculate(inputs).cpu().numpy()

        return float(np.percentile(scores, percentile * 100))

    def set_threshold(self, threshold: tuple[float]) -> None:
        """
        直接设置阈值

        参数:
            threshold (float): 要设置的阈值
        """
        self.threshold = threshold


class MSPDetector(BaseOODDetector):
    def __init__(self, model: nn.Module, threshold: tuple[float] = (0.5,)):
        """
        MSP (Maximum Softmax Probability) OOD 检测器

        参数:
            model (nn.Module): 预训练好的分类模型
        """
        super().__init__(model, threshold)

    def calculate(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算基于最大 softmax 概率的 OOD 分数

        注意: 分数 = 1 - max(softmax)
              分数越高表示 OOD 可能性越大
        """
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            return 1.0 - max_probs  # 低置信度 = 高 OOD 分数


class ODINDetector(BaseOODDetector):
    def __init__(
        self,
        model: nn.Module,
        threshold: tuple[float] = (0.5,),
        temperature: float = 1000.0,
        noise_magnitude: float = 0.0014,
        epsilon: float = 0.0,
    ):
        """
        ODIN (Out-of-Distribution Detector) OOD 检测器

        参数:
            model (nn.Module): 预训练好的分类模型
            temperature (float): 温度缩放参数
            noise_magnitude (float): 输入扰动强度
            epsilon (float): 梯度符号扰动前的微小偏移
        """
        super().__init__(model, threshold)
        self.temperature = temperature
        self.noise_magnitude = noise_magnitude
        self.epsilon = epsilon

    def calculate(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 ODIN 的 OOD 分数（包含温度缩放和输入扰动）
        """
        # 启用输入梯度
        x.requires_grad = True

        # 前向传播
        logits = self.model(x)

        # 应用温度缩放
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)

        # 计算梯度
        loss = -torch.log(max_probs + self.epsilon).mean()
        loss.backward()

        # 获取梯度并应用扰动
        if x.grad is None:
            raise ValueError("Gradient is None. Please check your model and inputs.")
        gradient = torch.sign(x.grad.data)
        perturbed_x = x.data - self.noise_magnitude * gradient

        # 禁用梯度计算
        with torch.no_grad():
            perturbed_logits = self.model(perturbed_x)
            perturbed_scaled_logits = perturbed_logits / self.temperature
            perturbed_probs = F.softmax(perturbed_scaled_logits, dim=1)
            max_perturbed_probs, _ = torch.max(perturbed_probs, dim=1)

        # 清理梯度
        x.grad = None
        x.requires_grad = False

        return 1.0 - max_perturbed_probs  # 低置信度 = 高 OOD 分数


class EnergyDetector(BaseOODDetector):
    def __init__(self, model: nn.Module, threshold: tuple[float] = (0.0,), temperature: float = 1.0):
        """
        基于能量分数的 OOD 检测器

        参数:
            model (nn.Module): 预训练好的分类模型
            temperature (float): 温度缩放参数
        """
        super().__init__(model, threshold)
        self.temperature = temperature

    def calculate(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算能量分数（分数越高表示 OOD 可能性越大）
        添加归一化到0-1范围

        公式: E(x) = -T * log(sum(exp(f_i(x)/T))
        """
        with torch.no_grad():
            logits = self.model(x)
            scaled_logits = logits / self.temperature
            return -self.temperature * torch.logsumexp(scaled_logits, dim=1)


if __name__ == "__main__":
    # 示例用法
    import torchvision.models as models

    # 1. 加载预训练模型
    model = models.resnet18()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # 2. 创建检测器实例
    msp_detector = MSPDetector(model)
    odin_detector = ODINDetector(model, temperature=1000.0, noise_magnitude=0.0014)
    energy_detector = EnergyDetector(model, temperature=1.0)

    # 3. 准备测试数据 (ID 和 OOD)
    id_sample = torch.randn(1, 3, 224, 224).to(device)  # ImageNet 类数据
    ood_sample = torch.rand(1, 3, 224, 224).to(device)  # 随机噪声

    # 4. 计算 OOD 分数
    for name, detector in [("MSP", msp_detector), ("ODIN", odin_detector), ("Energy", energy_detector)]:
        id_score = detector.calculate(id_sample).item()
        ood_score = detector.calculate(ood_sample).item()
        print(f"{name} - ID score: {id_score:.4f}, OOD score: {ood_score:.4f}, Diff: {ood_score - id_score:.4f}")

    # 5. 多级阈值检测示例
    # 设置双阈值 (0.3, 0.7) 对应三个 OOD Flags
    for detector in [msp_detector, odin_detector, energy_detector]:
        detector.set_threshold((0.3, 0.7))

        # 执行检测（不再需要传入threshold参数）
        id_classes, id_scores = detector.detect(id_sample)
        ood_classes, ood_scores = detector.detect(ood_sample)

        print(f"\nDetector: {detector.__class__.__name__}")
        print(f"ID sample: Score={id_scores.item():.4f}, OOD Flags: {id_classes.item()}")
        print(f"OOD sample: Score={ood_scores.item():.4f}, OOD Flags: {ood_classes.item()}")
