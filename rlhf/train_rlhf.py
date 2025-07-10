import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.transformer import Transformer
from rlhf.reward_model import RewardModel
from data.dataset import TextDataset
from config import load_config
import numpy as np

class PPOTrainer:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 初始化策略模型
        self.policy_model = Transformer(config).to(self.device)
        
        # 初始化奖励模型
        self.reward_model = RewardModel.load_pretrained(config).to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.policy_model.parameters(), 
            lr=config['rlhf']['reward_model_lr']
        )
        
        # 经验缓冲区
        self.buffer = []
        
    def collect_experience(self, dataset):
        # 从数据集中采样生成响应
        loader = DataLoader(dataset, batch_size=self.config['training']['batch_size'])
        
        for batch in loader:
            input_ids = batch[0].to(self.device)
            
            # 使用策略模型生成响应
            with torch.no_grad():
                logits = self.policy_model(input_ids)
                probs = torch.softmax(logits, dim=-1)
                actions = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
                actions = actions.view(probs.size(0), -1)
                
            # 计算奖励
            with torch.no_grad():
                rewards = self.reward_model(torch.cat([input_ids, actions], dim=1))
                
            # 存储经验
            self.buffer.append({
                'input_ids': input_ids,
                'actions': actions,
                'rewards': rewards
            })
    
    def update_policy(self):
        # PPO更新
        for epoch in range(self.config['rlhf']['ppo_epochs']):
            for experience in self.buffer:
                # 计算旧策略概率
                with torch.no_grad():
                    old_logits = self.policy_model(experience['input_ids'])
                    old_probs = torch.softmax(old_logits, dim=-1)
                    old_action_probs = old_probs.gather(-1, experience['actions'].unsqueeze(-1)).squeeze(-1)
                
                # 计算新策略概率
                new_logits = self.policy_model(experience['input_ids'])
                new_probs = torch.softmax(new_logits, dim=-1)
                new_action_probs = new_probs.gather(-1, experience['actions'].unsqueeze(-1)).squeeze(-1)
                
                # 计算比率和PPO损失
                ratios = new_action_probs / old_action_probs
                advantages = experience['rewards'] - experience['rewards'].mean()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.config['rlhf']['clip_epsilon'], 
                                    1+self.config['rlhf']['clip_epsilon']) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 更新模型
                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()
        
        # 清空缓冲区
        self.buffer = []

def main():
    config = load_config('config/base.yaml')
    dataset = TextDataset('data/train.txt', config)
    trainer = PPOTrainer(config)
    
    for _ in range(config['training']['max_steps']):
        trainer.collect_experience(dataset)
        trainer.update_policy()

if __name__ == "__main__":
    main()
