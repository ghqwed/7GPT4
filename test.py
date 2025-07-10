import torch
from model.transformer import Transformer
from config import load_config

def test_transformer():
    config = load_config('config/base.yaml')
    model = Transformer(config)
    
    # 测试输入
    input_ids = torch.randint(0, config.model.vocab_size, (1, config.model.max_seq_len))
    
    # 前向传播测试
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print("Transformer forward pass test passed!")

def test_reward_model():
    from rlhf.reward_model import load_pretrained
    config = load_config('config/base.yaml')
    model = load_pretrained(config)
    
    # 测试输入
    input_ids = torch.randint(0, config.model.vocab_size, (1, config.model.max_seq_len))
    
    # 奖励计算测试
    rewards = model(input_ids)
    print(f"Reward shape: {rewards.shape}")
    print("Reward model test passed!")

if __name__ == "__main__":
    test_transformer()
    test_reward_model()
