import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    """
    YAML配置文件加载器

    功能:
    加载YAML配置文件

    使用示例:
        config = ConfigLoader.load("config.yaml")
        lr = config("learning_rate")
    """

    @staticmethod
    def load(file_path: str) -> "ConfigLoader":
        """
        从文件加载配置

        参数:
            file_path: YAML配置文件路径

        返回:
            ConfigLoader 实例
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")

        # 加载YAML文件
        with open(file_path, "r") as f:
            config_data = yaml.safe_load(f) or {}

        return ConfigLoader(config_data)

    def __init__(self, config_data: Dict[str, Any]):
        self._config = config_data

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        参数:
            key: 点分隔的配置路径 (如: "training.learning_rate")
            default: 找不到配置时的默认值

        返回:
            配置值或默认值
        """
        keys = key.split(".")
        current = self._config

        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default

    def get_int(self, key: str, default: int = 0) -> int:
        """获取整数配置值"""
        value = self.get(key, default)
        return int(value)

    def get_float(self, key: str, default: float = 0.0) -> float:
        """获取浮点数配置值"""
        value = self.get(key, default)
        return float(value)

    def get_bool(self, key: str, default: bool = False) -> bool:
        """获取布尔值配置值"""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).lower() in ["true", "1", "yes", "y"]

    def get_str(self, key: str, default: str = "") -> str:
        """获取字符串配置值"""
        value = self.get(key, default)
        return str(value)

    def get_list(self, key: str, default: list = None) -> list:
        """获取列表配置值"""
        if default is None:
            default = []
        value = self.get(key, default)
        return list(value) if value is not None else []

    def get_dict(self, key: str, default: dict = None) -> dict:
        """获取字典配置值"""
        if default is None:
            default = {}
        value = self.get(key, default)
        return dict(value) if value is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        """返回完整的配置字典"""
        return self._config

    def __repr__(self) -> str:
        return f"ConfigLoader({self._config})"
