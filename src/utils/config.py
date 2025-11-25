import yaml


def load_config(path):
    with open(path, 'r', encoding='utf-8-sig') as f:  # chú ý 'utf-8-sig' để bỏ BOM
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Failed to load config from {path}")
    return config