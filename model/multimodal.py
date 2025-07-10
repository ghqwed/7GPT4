import torch
import torch.nn as nn
from model.transformer import Transformer

class MultimodalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 文本编码器
        self.text_encoder = Transformer(config)
        
        # 图像编码器 (简化版)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, config.model.dim)
        )
        
        # 多模态融合层
        self.fusion = nn.Linear(2 * config.model.dim, config.model.dim)
        
    def forward(self, text_input, image_input=None):
        # 文本编码
        text_features = self.text_encoder(text_input, return_hidden=True)
        
        if image_input is not None:
            # 图像编码
            image_features = self.image_encoder(image_input)
            # 特征融合
            combined = torch.cat([text_features[:, -1, :], image_features], dim=1)
            fused = self.fusion(combined)
            return fused
        return text_features
