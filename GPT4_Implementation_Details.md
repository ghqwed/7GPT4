# GPT-4 实现详解

## 项目概述
本项目实现了GPT-4技术报告中的核心算法，包括：
1. Transformer架构
2. 预训练(下一个token预测)
3. RLHF对齐微调
4. 多模态扩展(文本+图像)
5. 可预测缩放算法

```
项目结构:
.
├── config/                # 配置文件
│   └── base.yaml          # 基础模型配置
├── data/                  # 数据相关
│   ├── dataset.py         # 数据加载和处理
│   └── train.txt          # 示例训练数据
├── model/                 # 模型实现
│   ├── transformer.py     # Transformer核心架构
│   ├── multimodal.py      # 多模态扩展
│   └── scaling.py         # 可预测缩放算法
├── rlhf/                  # RLHF实现
│   ├── reward_model.py    # 奖励模型
│   └── train_rlhf.py      # RLHF训练
├── train.py               # 预训练脚本
├── test.py                # 测试脚本
└── config.py              # 配置加载工具
```

## Transformer架构详解 (带详细代码注释)

### 核心组件
1. **多头注意力机制**:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.model.n_heads  # 注意力头数量
        self.dim = config.model.dim          # 模型维度
        self.head_dim = self.dim // self.n_heads  # 每个头的维度
        
        # QKV投影矩阵 (将输入同时投影到Q,K,V空间)
        self.qkv = nn.Linear(self.dim, 3 * self.dim)  
        # 输出投影矩阵
        self.proj = nn.Linear(self.dim, self.dim)  
        
    def forward(self, x):
        B, T, C = x.shape  # 批量大小,序列长度,特征维度
        # 将输入分割为Q,K,V三个部分
        qkv = self.qkv(x).split(self.dim, dim=2)  
        # 重塑为多头形式 [B, n_heads, T, head_dim]
        q, k, v = [y.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) 
                  for y in qkv]
        
        # 计算注意力分数 (QK^T/sqrt(d))
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # 计算注意力权重 (softmax归一化)
        attn = F.softmax(attn, dim=-1)
        # 加权求和 (注意力权重×V)
        out = attn @ v
        # 合并多头 [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C) 
        # 输出投影
        return self.proj(out)
```

2. **前馈网络**:
```python
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 两层全连接网络，中间层扩大4倍
        self.net = nn.Sequential(
            nn.Linear(config.model.dim, 4 * config.model.dim),  # 扩展层
            nn.GELU(),  # 激活函数
            nn.Linear(4 * config.model.dim, config.model.dim)  # 压缩回原尺寸
        )
        
    def forward(self, x):
        return self.net(x)  # 顺序执行定义好的网络
```

3. **Transformer块**:
```python 
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 第一个层归一化(注意力前)
        self.ln1 = nn.LayerNorm(config.model.dim)  
        # 多头注意力层
        self.attn = MultiHeadAttention(config)  
        # 第二个层归一化(前馈网络前)
        self.ln2 = nn.LayerNorm(config.model.dim)  
        # 前馈网络
        self.ff = FeedForward(config)  
        
    def forward(self, x):
        # 残差连接1: 输入 + 注意力输出
        x = x + self.attn(self.ln1(x))  
        # 残差连接2: 输入 + 前馈网络输出
        x = x + self.ff(self.ln2(x))  
        return x
```

### 完整架构
```python
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token嵌入层 (将token ID转换为向量)
        self.token_embed = nn.Embedding(config.model.vocab_size, config.model.dim)
        # 位置嵌入层 (为每个位置生成向量)
        self.pos_embed = nn.Embedding(config.model.max_seq_len, config.model.dim)
        
        # 堆叠多个Transformer块
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.model.n_layers)])
        # 最终层归一化
        self.ln_f = nn.LayerNorm(config.model.dim)
        # 输出头 (预测下一个token的概率分布)
        self.head = nn.Linear(config.model.dim, config.model.vocab_size, bias=False)
        
    def forward(self, idx, return_hidden=False):
        B, T = idx.shape  # 批量大小, 序列长度
        # Token嵌入
        tok_emb = self.token_embed(idx)
        # 位置编码 (0到T-1的位置ID)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_embed(pos)
        
        # 合并token和位置嵌入
        x = tok_emb + pos_emb
        # 通过所有Transformer块
        x = self.blocks(x)
        # 最终层归一化
        x = self.ln_f(x)
        
        if return_hidden:  # 如果只需要隐藏状态
            return x
        # 计算输出logits (未归一化的概率)
        logits = self.head(x)
        return logits
```

## 多模态处理流程 (带详细代码注释)

```python
class MultimodalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 文本编码器 (标准Transformer)
        self.text_encoder = Transformer(config)
        
        # 图像编码器 (简化CNN结构)
        self.image_encoder = nn.Sequential(
            # 第一卷积层: 3通道输入,64通道输出,3x3卷积核,步长2
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # 激活函数
            # 第二卷积层: 64通道输入,128通道输出,3x3卷积核,步长2 
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # 激活函数
            # 全局平均池化 (将任意尺寸特征图变为1x1)
            nn.AdaptiveAvgPool2d(1),
            # 展平特征
            nn.Flatten(),
            # 线性投影到模型维度
            nn.Linear(128, config.model.dim)
        )
        
        # 特征融合层 (将文本和图像特征合并)
        self.fusion = nn.Linear(2 * config.model.dim, config.model.dim)
        
    def forward(self, text_input, image_input=None):
        # 文本编码 [batch_size, seq_len] -> [batch_size, seq_len, dim]
        text_features = self.text_encoder(text_input, return_hidden=True)
        
        if image_input is not None:
            # 图像编码 [batch_size, 3, H, W] -> [batch_size, dim]
            image_features = self.image_encoder(image_input)
            # 特征融合: 取文本最后一个token + 图像特征
            combined = torch.cat([
                text_features[:, -1, :],  # 取序列最后一个token的特征
                image_features
            ], dim=1)
            # 通过融合层降维
            fused = self.fusion(combined)
            return fused
        return text_features  # 如果没有图像输入，只返回文本特征
```

## RLHF训练过程 (带详细代码注释)

### 1. 奖励模型
```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # 使用预训练的Transformer作为基础模型
        self.transformer = base_model
        # 冻结基础模型参数 (RLHF阶段不更新)
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        # 添加奖励预测头 (单输出值)
        self.reward_head = nn.Linear(base_model.config.model.dim, 1)
        
    def forward(self, input_ids):
        # 获取Transformer的隐藏状态
        hidden_states = self.transformer(input_ids, return_hidden=True)
        # 取序列最后一个token的隐藏状态作为特征
        last_hidden = hidden_states[:, -1, :]  
        # 计算奖励分数
        rewards = self.reward_head(last_hidden)
        return rewards.squeeze(-1)  # 去除多余的维度
```

### 2. PPO训练流程
```python
class PPOTrainer:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 策略模型 (需要训练)
        self.policy_model = Transformer(config).to(self.device)
        # 奖励模型 (固定参数)
        self.reward_model = RewardModel.load_pretrained(config).to(self.device)
        
        # 优化器 (只优化策略模型)
        self.optimizer = optim.AdamW(
            self.policy_model.parameters(), 
            lr=config['rlhf']['reward_model_lr']
        )
        
        # 经验缓冲区
        self.buffer = []
    
    def collect_experience(self, dataset):
        loader = DataLoader(dataset, batch_size=self.config['training']['batch_size'])
        
        for batch in loader:
            input_ids = batch[0].to(self.device)
            
            # 策略模型生成响应
            with torch.no_grad():
                logits = self.policy_model(input_ids)
                probs = torch.softmax(logits, dim=-1)
                # 根据概率分布采样token
                actions = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
                actions = actions.view(probs.size(0), -1)
                
            # 奖励模型评分
            with torch.no_grad():
                rewards = self.reward_model(torch.cat([input_ids, actions], dim=1))
                
            # 存储经验 (状态-动作-奖励)
            self.buffer.append({
                'input_ids': input_ids,
                'actions': actions,
                'rewards': rewards
            })
    
    def update_policy(self):
        # PPO多轮更新
        for epoch in range(self.config['rlhf']['ppo_epochs']):
            for experience in self.buffer:
                # 计算旧策略的概率
                with torch.no_grad():
                    old_logits = self.policy_model(experience['input_ids'])
                    old_probs = torch.softmax(old_logits, dim=-1)
                    old_action_probs = old_probs.gather(-1, 
                        experience['actions'].unsqueeze(-1)).squeeze(-1)
                
                # 计算新策略的概率
                new_logits = self.policy_model(experience['input_ids'])
                new_probs = torch.softmax(new_logits, dim=-1)
                new_action_probs = new_probs.gather(-1, 
                    experience['actions'].unsqueeze(-1)).squeeze(-1)
                
                # 计算概率比 (新策略概率/旧策略概率)
                ratios = new_action_probs / old_action_probs
                # 计算优势 (奖励减去基线)
                advantages = experience['rewards'] - experience['rewards'].mean()
                
                # PPO损失计算
                surr1 = ratios * advantages  # 原始概率比
                surr2 = torch.clamp(ratios,  # 裁剪后的概率比
                    1-self.config['rlhf']['clip_epsilon'], 
                    1+self.config['rlhf']['clip_epsilon']) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()  # 取最小值
                
                # 梯度更新
                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()
        
        self.buffer = []  # 清空经验缓冲区
```

## 可预测缩放算法 (带详细代码注释)

```python
class PredictableScaling:
    def __init__(self, config):
        self.config = config
        # 计算基础模型的参数量
        self.base_params = self._calculate_base_params()
        
    def _calculate_base_params(self):
        """计算基础模型的参数量"""
        L = self.config.model.n_layers  # Transformer层数
        d = self.config.model.dim       # 模型维度
        h = self.config.model.n_heads   # 注意力头数
        V = self.config.model.vocab_size  # 词表大小
        
        # 注意力层参数 (QKV投影 + 输出投影)
        attn_params = 4 * L * d * d  
        # 前馈网络参数 (两层全连接)
        ff_params = 2 * L * d * 4 * d  
        # 嵌入层参数 (token嵌入 + 位置嵌入)
        embed_params = V * d + self.config.model.max_seq_len * d  
        
        return attn_params + ff_params + embed_params
    
    def predict_performance(self, scale_factor):
        """
        预测缩放后模型的性能指标
        scale_factor: 缩放因子 (调整模型宽度或深度)
        """
        # 缩放后的参数量 (与scale_factor的平方成正比)
        scaled_params = self.base_params * scale_factor**2
        
        # 预测计算量 (FLOPs/token)
        flops = 2 * scaled_params  # 每个参数约2次浮点运算
        
        # 预测内存占用 (GB)
        memory = scaled_params * 4 / (1024**3)  # 每个参数4字节(32位浮点)
        
        return {
            'parameters': int(scaled_params),
            'flops_per_token': int(flops),
            'memory_gb': round(memory, 2)
        }
    
    def find_optimal_scale(self, target_flops, max_memory=10):
        """
        寻找满足目标计算量的最优缩放因子
        target_flops: 目标计算量 (FLOPs/token)
        max_memory: 最大内存限制(GB)
        """
        # 计算基于内存限制的最大可能缩放因子
        max_scale = math.sqrt(max_memory * (1024**3) / (4 * self.base_params))
        
        # 二分查找寻找最优缩放因子
        low = 0.1  # 最小缩放因子
        high = max_scale  # 最大缩放因子
        for _ in range(20):  # 20次迭代足够精确
            mid = (low + high) / 2
            pred = self.predict_performance(mid)
            if pred['flops_per_token'] < target_flops:
                low = mid  # 如果计算量不足，增大缩放因子
            else:
                high = mid  # 如果计算量超限，减小缩放因子
        
        # 返回最终缩放因子和预测性能
        optimal_scale = (low + high) / 2
        return optimal_scale, self.predict_performance(optimal_scale)
```

### 算法原理
1. **参数计算**:
   - 注意力参数: 4 × 层数 × 维度² (Q,K,V投影 + 输出投影)
   - 前馈参数: 2 × 层数 × 维度 × (4×维度) (两层全连接)
   - 嵌入参数: 词表大小 × 维度 + 最大序列长度 × 维度

2. **性能预测**:
   - 计算量 ≈ 2 × 参数量 (每个参数约2次浮点运算)
   - 内存 ≈ 参数量 × 4字节 (32位浮点数)

3. **缩放因子搜索**:
   - 使用二分查找在内存限制内找到满足目标计算量的最优缩放因子
   - 缩放因子与模型维度或深度成正比

## 使用示例

### 基础使用
```python
from model.transformer import Transformer
model = Transformer(config)
output = model(input_ids)
```

### 多模态使用
```python
from model.multimodal import MultimodalTransformer
model = MultimodalTransformer(config)
output = model(text_input, image_input)
```

### 可预测缩放
```python
from model.scaling import PredictableScaling
scaler = PredictableScaling(config)
scale, metrics = scaler.find_optimal_scale(target_flops=1e9)
```

### RLHF训练
```bash
python rlhf/train_rlhf.py
```

## 配置参数
```yaml
model:
  dim: 768           # 模型维度
  n_layers: 12       # Transformer层数
  n_heads: 12        # 注意力头数
  vocab_size: 50257  # 词表大小
  max_seq_len: 128   # 最大序列长度

training:
  batch_size: 32     # 训练批量大小
  learning_rate: 6e-5 # 学习率
