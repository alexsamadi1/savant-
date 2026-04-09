import toml
import os
from pathlib import Path

_config = None

def get_config() -> dict:
    global _config
    if _config is None:
        base_path = Path(__file__).parent / "config.toml"
        if not base_path.exists():
            raise FileNotFoundError("config.toml not found.")
        _config = toml.load(base_path)

        tenant = os.environ.get("TENANT_PREFIX", "").strip("/")
        if tenant:
            tenant_path = Path(__file__).parent / "clients" / f"{tenant}.toml"
            if tenant_path.exists():
                override = toml.load(tenant_path)
                for section, values in override.items():
                    if isinstance(values, dict) and section in _config:
                        _config[section].update(values)
                    else:
                        _config[section] = values
                print(f"[CONFIG] Loaded tenant config: config_{tenant}.toml")
            else:
                print(f"[CONFIG] No tenant config found for: {tenant} — using defaults")

        REQUIRED_TENANT_KEYS = {
            "assistant": ["welcome_message", "sample_questions",
                          "fallback_message", "chat_placeholder"],
            "brand": ["app_name", "company_name"],
            "contact": ["contact_email"]
        }

        if tenant:
            for section, keys in REQUIRED_TENANT_KEYS.items():
                for key in keys:
                    if key not in _config.get(section, {}):
                        print(f"[CONFIG] WARNING: {tenant} config missing "
                              f"{section}.{key} — using default")

    return _config

def reset_config():
    global _config
    _config = None
    print("[CONFIG] Cache cleared — will reload on next get_config() call")
