import toml
from pathlib import Path

_config = None

def get_config() -> dict:
    global _config
    if _config is None:
        config_path = Path(__file__).parent / "config.toml"
        if not config_path.exists():
            raise FileNotFoundError(
                "config.toml not found. Copy config.toml.example and fill in your values."
            )
        _config = toml.load(config_path)
    return _config