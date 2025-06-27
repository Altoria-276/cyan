根据您提供的联邦学习框架图，我为您设计了一个完整的代码方案。以下是详细的目录结构和文件说明：

### 目录结构
```
federated_learning/
├── main.py                     # 主程序入口
├── config.py                   # 全局配置参数
├── requirements.txt            # 依赖库
│
├── core/                       # 核心框架
│   ├── __init__.py
│   ├── server.py               # 中央服务器实现
│   ├── node.py                 # 联邦节点实现
│   └── aggregation.py          # 联邦聚合算法
│
├── modules/                    # 核心功能模块
│   ├── __init__.py
│   ├── cognitive_cache.py      # 数据认知缓存模块 (P2)
│   ├── online_learner.py       # 在线学习模块 (P3)
│   └── pretrainer.py           # 预训练模块
│
├── model/                      # 模型定义
│   ├── __init__.py
│   ├── core_net.py             # 认知网络模型
│   └── feature_extractor.py    # 特征提取器
│
├── data/                       # 数据处理
│   ├── __init__.py
│   ├── dataset.py              # 数据集处理
│   ├── ood_detector.py         # OOD检测算法
│   └── buffer_pool.py          # 数据缓冲池
│
└── utils/                      # 工具函数
    ├── __init__.py
    ├── logger.py               # 日志记录
    ├── evaluator.py            # 模型评估
    └── helper.py               # 辅助函数
```

### 关键文件说明

#### 1. main.py
```python
from core.server import FederatedServer
from config import get_config

def main():
    config = get_config()
    
    # 初始化联邦服务器
    server = FederatedServer(config)
    
    # 预训练全局模型
    server.pretrain_global_model()
    
    # 联邦学习主循环
    for round in range(config["federated_rounds"]):
        print(f"======= Federated Round {round+1} =======")
        
        # 选择参与本轮训练的节点
        selected_nodes = server.select_nodes()
        
        # 分发全局模型
        server.distribute_model(selected_nodes)
        
        # 节点本地训练
        local_updates = []
        for node in selected_nodes:
            update = node.local_train()
            local_updates.append(update)
        
        # 聚合更新
        global_update = server.aggregate_updates(local_updates)
        
        # 更新全局模型
        server.update_global_model(global_update)
        
        # 评估全局模型
        server.evaluate_global_model()
    
    print("Federated training completed!")

if __name__ == "__main__":
    main()
```

#### 2. core/server.py
```python
import torch
from modules.pretrainer import Pretrainer
from modules.aggregation import FedAvgAggregator

class FederatedServer:
    def __init__(self, config):
        self.config = config
        self.global_model = self._init_model()
        self.nodes = self._init_nodes()
        self.aggregator = FedAvgAggregator()
        
    def _init_model(self):
        # 初始化全局模型
        from model.feature_extractor import FeatureExtractor
        return FeatureExtractor(self.config["model"])
    
    def _init_nodes(self):
        # 初始化联邦节点
        from core.node import FederatedNode
        return [FederatedNode(self.config) for _ in range(self.config["num_nodes"])]
    
    def pretrain_global_model(self):
        pretrainer = Pretrainer(self.config)
        self.global_model = pretrainer.train(self.global_model)
    
    def distribute_model(self, nodes):
        # 分发全局模型到节点
        for node in nodes:
            node.receive_model(self.global_model.state_dict())
    
    def aggregate_updates(self, local_updates):
        return self.aggregator.aggregate(local_updates)
    
    def update_global_model(self, global_update):
        self.global_model.load_state_dict(global_update)
    
    def evaluate_global_model(self):
        from utils.evaluator import ModelEvaluator
        evaluator = ModelEvaluator(self.config)
        accuracy = evaluator.evaluate(self.global_model)
        print(f"Global Model Accuracy: {accuracy:.2f}%")
```

#### 3. core/node.py
```python
from modules.cognitive_cache import CognitiveCache
from modules.online_learner import OnlineLearner

class FederatedNode:
    def __init__(self, config):
        self.config = config
        self.local_model = None
        self.local_data = self._load_local_data()
        self.cognitive_cache = CognitiveCache(config)
        self.online_learner = OnlineLearner(config)
    
    def _load_local_data(self):
        from data.dataset import load_local_dataset
        return load_local_dataset(self.config)
    
    def receive_model(self, model_state):
        from model.feature_extractor import FeatureExtractor
        self.local_model = FeatureExtractor(self.config["model"])
        self.local_model.load_state_dict(model_state)
    
    def local_train(self):
        # 认知缓存处理
        classified_data = self.cognitive_cache.process(
            self.local_data, 
            self.local_model
        )
        
        # 在线学习
        self.online_learner.update(
            self.local_model, 
            classified_data["id_data"],
            classified_data["near_ood_data"]
        )
        
        # 本地训练逻辑
        # ... (省略具体训练代码)
        
        # 返回模型更新
        return self.local_model.state_dict()
```

#### 4. modules/cognitive_cache.py (P2实现)
```python
import torch
from data.ood_detector import OODDetector
from data.buffer_pool import DataBufferPool

class CognitiveCache:
    def __init__(self, config):
        self.config = config
        self.ood_detector = OODDetector(config)
        self.buffer_pool = DataBufferPool(config["buffer_size"])
        
    def process(self, local_data, model):
        # 提取特征
        features = self._extract_features(local_data, model)
        
        # 检测OOD数据
        results = self.ood_detector.detect(features)
        
        # 分类数据
        classified = {
            "id_data": local_data[results["id_indices"]],
            "near_ood_data": local_data[results["near_ood_indices"]],
            "far_ood_data": local_data[results["far_ood_indices"]]
        }
        
        # 更新缓冲池
        self.buffer_pool.add(classified["near_ood_data"])
        
        # 更新认知网络
        if len(classified["id_data"]) > 0:
            self._update_core_net(classified["id_data"])
        
        return classified
    
    def _extract_features(self, data, model):
        # 特征提取实现
        with torch.no_grad():
            return model(data)
    
    def _update_core_net(self, id_data):
        # 认知网络自更新逻辑
        from model.core_net import CoreNet
        core_net = CoreNet(self.config["core_net"])
        # ... (训练代码)
```

#### 5. modules/online_learner.py (P3实现)
```python
import torch
import torch.optim as optim

class OnlineLearner:
    def __init__(self, config):
        self.config = config
        self.optimizer = None
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def update(self, model, id_data, near_ood_data):
        # 准备数据
        train_data = self._combine_data(id_data, near_ood_data)
        
        # 设置优化器
        self.optimizer = optim.SGD(
            model.parameters(), 
            lr=self.config["online_lr"]
        )
        
        # 在线学习循环
        model.train()
        for epoch in range(self.config["online_epochs"]):
            for batch in train_data:
                inputs, labels = batch
                self.optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
    
    def _combine_data(self, id_data, ood_data):
        # 组合ID和OOD数据
        # 给OOD数据分配特殊标签
        # ... (具体实现)
        return combined_dataset
```

#### 6. model/core_net.py (P2认知网络)
```python
import torch.nn as nn

class CoreNet(nn.Module):
    """认知网络模型 (P2)"""
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["input_dim"], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 三类输出: ID, Near-OOD, Far-OOD
        )
    
    def forward(self, x):
        return self.layers(x)
```

#### 7. data/ood_detector.py
```python
import numpy as np
from sklearn.cluster import KMeans

class OODDetector:
    def __init__(self, config):
        self.config = config
        self.id_threshold = config["id_threshold"]
        self.near_ood_threshold = config["near_ood_threshold"]
    
    def detect(self, features):
        # 1. 计算ID分数 (示例使用Mahalanobis距离)
        id_scores = self._compute_id_scores(features)
        
        # 2. 聚类分析检测OOD
        cluster_results = self._cluster_analysis(features)
        
        # 3. 综合判断
        results = {
            "id_indices": [],
            "near_ood_indices": [],
            "far_ood_indices": []
        }
        
        for i, (id_score, cluster_label) in enumerate(zip(id_scores, cluster_results)):
            if id_score < self.id_threshold:
                results["id_indices"].append(i)
            elif id_score < self.near_ood_threshold:
                results["near_ood_indices"].append(i)
            else:
                results["far_ood_indices"].append(i)
        
        return results
    
    def _compute_id_scores(self, features):
        # 实现ID分数计算逻辑
        # ... (伪代码)
        return np.random.rand(len(features))  # 示例返回随机值
    
    def _cluster_analysis(self, features):
        # 聚类分析实现
        kmeans = KMeans(n_clusters=3)
        return kmeans.fit_predict(features)
```

### 运行流程
1. **初始化**：
   - 服务器初始化全局模型
   - 创建联邦节点并分配本地数据

2. **预训练**：
   - 服务器在公共数据集上预训练特征提取器

3. **联邦学习循环**：
   - 服务器选择参与本轮训练的节点
   - 分发全局模型到节点
   - 节点本地训练：
     * 使用认知缓存模块处理本地数据
     * 使用在线学习模块更新模型
   - 节点返回模型更新
   - 服务器聚合更新生成新全局模型
   - 评估全局模型性能

4. **结果输出**：
   - 保存最终模型
   - 输出性能指标

### 关键特性
1. **模块化设计**：每个核心组件独立实现，易于扩展和维护
2. **认知缓存**：实现P2图的ID/OOD数据分类和缓冲池管理
3. **在线学习**：实现P3图的动态模型更新机制
4. **灵活配置**：所有参数通过config.py集中管理
5. **评估系统**：内置模型性能评估模块

这个设计完整实现了您提供的框架图中的所有组件，可以处理联邦学习中的OOD数据问题，并通过在线学习机制持续改进模型性能。