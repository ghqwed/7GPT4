import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import yaml
from model.transformer import Transformer
from data.dataset import TextDataset, get_batch
import os

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确保数值类型正确
    for section in config.values():
        for key, value in section.items():
            if isinstance(value, str) and value.replace('.', '', 1).isdigit():
                section[key] = float(value) if '.' in value else int(value)
    
    return config

def main():
    # 加载配置
    config_dict = load_config('config/base.yaml')
    
    # 将字典转换为SimpleNamespace对象
    from types import SimpleNamespace
    config = SimpleNamespace()
    config.model = SimpleNamespace()
    for key, value in config_dict['model'].items():
        setattr(config.model, key, value)
    for key, value in config_dict['training'].items():
        setattr(config, key, value)
    
    # 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(config)
    model = model.to(device)
    
    # 准备数据
    dataset = TextDataset('data/train.txt', config)
    
    # 确保所有数值参数正确转换
    learning_rate = float(config.learning_rate) if isinstance(config.learning_rate, str) else config.learning_rate
    warmup_steps = int(config.warmup_steps) if isinstance(config.warmup_steps, str) else config.warmup_steps
    max_steps = int(config.max_steps) if isinstance(config.max_steps, str) else config.max_steps
    batch_size = int(config.batch_size) if isinstance(config.batch_size, str) else config.batch_size
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # 训练循环
    for step in range(1, max_steps + 1):
        # 学习率warmup
        if step < warmup_steps:
            lr = learning_rate * (step / warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # 获取批次数据
        x, y = get_batch(dataset, config.batch_size)
        x, y = x.to(device), y.to(device)
        
        # 前向传播
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 打印训练信息并保存checkpoint
        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # 每1000步保存一次模型
            if step % 1000 == 0:
                checkpoint = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }
                torch.save(checkpoint, f'checkpoints/checkpoint_step{step}.pt')
                print(f"Saved checkpoint at step {step}")
                
        # 确保checkpoints目录存在
        os.makedirs('checkpoints', exist_ok=True)

if __name__ == "__main__":
    main()
