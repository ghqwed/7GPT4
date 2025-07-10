import yaml
from types import SimpleNamespace

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 将字典转换为对象属性访问方式
    config = SimpleNamespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            setattr(config, key, SimpleNamespace(**value))
        else:
            setattr(config, key, value)
    return config
