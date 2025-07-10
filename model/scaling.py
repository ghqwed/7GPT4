import math
import torch
import numpy as np

class PredictableScaling:
    def __init__(self, config):
        self.config = config
        self.base_params = self._calculate_base_params()
        
    def _calculate_base_params(self):
        """计算基础模型的参数量"""
        L = self.config.model.n_layers
        d = self.config.model.dim
        h = self.config.model.n_heads
        V = self.config.model.vocab_size
        
        # Transformer主要参数
        attn_params = 4 * L * d * d  # QKV投影 + 输出投影
        ff_params = 2 * L * d * 4 * d  # 前馈网络
        embed_params = V * d + self.config.model.max_seq_len * d  # token和位置嵌入
        
        return attn_params + ff_params + embed_params
    
    def predict_performance(self, scale_factor):
        """
        预测模型缩放后的性能
        scale_factor: 缩放因子 (深度或宽度)
        """
        # 计算缩放后的参数量
        scaled_params = self.base_params * scale_factor**2
        
        # 预测计算量 (FLOPs/token)
        flops = 2 * scaled_params  # 近似估计
        
        # 预测内存占用
        memory = scaled_params * 4 / (1024**3)  # 假设32位浮点, 转换为GB
        
        return {
            'parameters': int(scaled_params),
            'flops_per_token': int(flops),
            'memory_gb': round(memory, 2)
        }
    
    def find_optimal_scale(self, target_flops, max_memory=10):
        """
        根据目标计算量和内存限制找到最优缩放因子
        max_memory: 最大内存限制(GB)
        """
        max_scale = math.sqrt(max_memory * (1024**3) / (4 * self.base_params))
        
        # 二分查找最优缩放因子
        low = 0.1
        high = max_scale
        for _ in range(20):
            mid = (low + high) / 2
            pred = self.predict_performance(mid)
            if pred['flops_per_token'] < target_flops:
                low = mid
            else:
                high = mid
        
        optimal_scale = (low + high) / 2
        return optimal_scale, self.predict_performance(optimal_scale)
