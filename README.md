# GPT-4 简化实现

本项目实现了GPT-4技术报告中的核心算法，包括：

1. Transformer架构
2. 预训练（下一个token预测）
3. RLHF对齐微调
4. 多模态扩展（文本+图像）

## 文件结构

```
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

## 使用说明

1. 安装依赖：
```bash
pip install torch transformers tokenizers numpy pyyaml
```

2. 运行预训练：
```bash
python train.py
```

3. 运行RLHF微调：
```bash
python rlhf/train_rlhf.py
```

4. 运行测试：
```bash
python test.py
```

## 新增功能示例

### 多模态使用
```python
from model.multimodal import MultimodalTransformer
model = MultimodalTransformer(config)
# text_input: [batch_size, seq_len]
# image_input: [batch_size, 3, 224, 224] 
output = model(text_input, image_input)
```

### 可预测缩放
```python
from model.scaling import PredictableScaling
scaler = PredictableScaling(config)
scale_factor, metrics = scaler.find_optimal_scale(target_flops=1e9)
print(f"Optimal scale: {scale_factor:.2f}, Params: {metrics['parameters']}")
```

## 注意事项

1. 本项目是GPT-4的简化实现，仅供学习研究使用
2. 完整训练需要大量计算资源和数据
3. 多模态部分需要准备图像数据
4. 缩放算法基于理论计算，实际性能可能有所差异
