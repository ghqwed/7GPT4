import torch
import torch.nn as nn
from model.transformer import Transformer

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.transformer = base_model
        # 冻结基础模型参数
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        # 添加奖励头
        self.reward_head = nn.Linear(base_model.config.model.dim, 1)
        
    def forward(self, input_ids):
        # 获取Transformer隐藏状态
        hidden_states = self.transformer(input_ids, return_hidden=True)
        # 使用最后一层隐藏状态计算奖励
        last_hidden = hidden_states[:, -1, :]  # 取序列最后一个token
        rewards = self.reward_head(last_hidden)
        return rewards.squeeze(-1)

def load_pretrained(config):
    # 加载预训练模型
    model = Transformer(config)
    # 这里应该加载预训练权重，暂时跳过
    return RewardModel(model)
