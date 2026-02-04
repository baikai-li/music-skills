"""
Configuration management module for music-skills CLI.
Supports configuration via config file and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for music-skills CLI."""

    DEFAULT_CONFIG_NAME = "config.yaml"
    DEFAULT_CONFIG_PATHS = [
        Path.cwd() / DEFAULT_CONFIG_NAME,
        Path.home() / ".music-skills" / DEFAULT_CONFIG_NAME,
        Path(__file__).parent.parent.parent / DEFAULT_CONFIG_NAME,
    ]

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration manager.

        Args:
            config_path: Optional path to custom config file.
        """
        self._config: Dict[str, Any] = {}
        self._config_path = config_path
        self.load()

    def load(self) -> None:
        """Load configuration from file."""
        config_path = self._find_config_file()
        if config_path and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = {}

    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file.

        Returns:
            Path to configuration file or None.
        """
        if self._config_path:
            return self._config_path

        for path in self.DEFAULT_CONFIG_PATHS:
            if path.exists():
                return path

        return None

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key (dot notation supported).
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        # Check environment variable first
        env_key = f"MUSIC_SKILLS_{key.upper().replace('.', '_')}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value

        # Navigate nested config
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default

        return value

    def get_path(self, key: str, default: Path = Path.cwd()) -> Path:
        """Get configuration value as path.

        Args:
            key: Configuration key.
            default: Default path if not found.

        Returns:
            Path object.
        """
        value = self.get(key, str(default))
        return Path(value)

    @property
    def audio_source_dir(self) -> Path:
        """Get audio source directory."""
        return self.get_path("paths.audio_source", Path("/Users/mindstorm/Projects/Virtual-IP-DB/audios/韩立"))

    @property
    def models_dir(self) -> Path:
        """Get models directory."""
        return self.get_path("paths.models", Path("assets/models"))

    @property
    def output_dir(self) -> Path:
        """Get output directory."""
        return self.get_path("paths.output", Path("assets/output"))

    @property
    def separated_dir(self) -> Path:
        """Get separated audio directory."""
        return self.get_path("paths.separated", Path("assets/separated"))

    @property
    def sample_rate(self) -> int:
        """Get default sample rate."""
        return int(self.get("audio.sample_rate", 48000))

    @property
    def rvc_config(self) -> Dict[str, Any]:
        """Get RVC configuration."""
        return self.get("rvc", {
            "f0method": "harvest",
            "pitch_shift": 0,
            "index_rate": 0.5,
            "filter_radius": 3,
            "resample_sr": 0,
            "rms_mix_rate": 0.5,
            "protect": 0.33,
        })

    @property
    def uvr_config(self) -> Dict[str, Any]:
        """Get UVR configuration."""
        return self.get("uvr", {
            "model_name": "HP2",
            "device": "cuda",
            "segment_size": 512,
            "batch_size": 1,
        })

    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to file.

        Args:
            path: Path to save configuration.
        """
        save_path = path or self._config_path or Path.cwd() / self.DEFAULT_CONFIG_NAME
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """Get global configuration instance.

    Args:
        config_path: Optional path to config file.

    Returns:
        Configuration instance.
    """
    global _config
    if _config is None or config_path is not None:
        _config = Config(config_path)
    return _config


def reset_config() -> None:
    """Reset global configuration."""
    global _config
    _config = None
