from typing import Dict, List, Tuple, Any, Callable, Optional
import torch
from collections import defaultdict


class ConfidenceAwareCache:
    def __init__(
        self,
        update_thresholds: Dict[str, int],
        update_handlers: Dict[str, Callable[[List[Any]], None]],
        eviction_policy: Optional[Callable] = None,
    ):
        """
        自适应置信度感知缓存池

        Args:
            update_thresholds: 各置信度级别的触发阈值
                (e.g. {'high': 50, 'medium': 200, 'low': 1000})
            update_handlers: 各置信度级别的更新回调函数
            eviction_policy: 自定义数据淘汰策略函数
        """
        self.cache = defaultdict(list)
        self.thresholds = update_thresholds
        self.handlers = update_handlers
        self.eviction_policy = eviction_policy

        # 状态监控
        self.update_counts = {level: 0 for level in update_thresholds.keys()}

    def add_data(self, confidence_level: str, data: torch.Tensor, metadata: Optional[dict] = None):
        """添加数据到缓存池

        Args:
            confidence_level: 置信度分级 ('high'/'medium'/'low')
            data: 图像张量 [C, H, W]
            metadata: 附加元数据 (如原始路径、检测分数等)
        """
        # 封装数据单元
        data_unit = {"tensor": data.clone().detach(), "metadata": metadata or {}}

        self.cache[confidence_level].append(data_unit)

        # 触发条件检查
        if len(self.cache[confidence_level]) >= self.thresholds[confidence_level]:
            self._trigger_update(confidence_level)

        # 执行淘汰策略
        if self.eviction_policy:
            self.cache[confidence_level] = self.eviction_policy(self.cache[confidence_level], self.thresholds[confidence_level])

    def _trigger_update(self, level: str):
        """触发指定级别的更新操作"""
        if level not in self.handlers:
            raise KeyError(f"No handler registered for level '{level}'")

        # 获取当前批次数据并清空缓存
        batch_data = self.cache[level]
        self.cache[level] = []

        # 执行更新操作
        self.handlers[level](batch_data)
        self.update_counts[level] += 1

        print(f"🚀 Triggered {level}-confidence update (batch size: {len(batch_data)})")

    def manual_flush(self, level: str):
        """强制刷新指定级别缓存"""
        if self.cache[level]:
            self._trigger_update(level)

    def get_cache_status(self) -> dict:
        """返回当前缓存状态统计"""
        return {
            level: {"count": len(data), "update_count": self.update_counts[level], "threshold": self.thresholds[level]}
            for level, data in self.cache.items()
        }
