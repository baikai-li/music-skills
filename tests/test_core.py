"""Tests for music-skills CLI."""

import pytest
from pathlib import Path
import tempfile

from music_skills.config import Config, get_config
from music_skills.preprocessing.audio import AudioPreprocessor


class TestConfig:
    """Tests for configuration module."""

    def test_config_load(self):
        """Test configuration loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("paths:\n  audio_source: /test/path\n")
            f.flush()
            config = Config(Path(f.name))
            assert str(config.audio_source_dir) == "/test/path"

    def test_config_get_nested(self):
        """Test getting nested configuration values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("rvc:\n  f0method: harvest\n")
            f.flush()
            config = Config(Path(f.name))
            assert config.get("rvc.f0method") == "harvest"

    def test_config_default(self):
        """Test configuration default values."""
        config = Config()
        assert config.sample_rate == 48000


class TestAudioPreprocessor:
    """Tests for audio preprocessor."""

    def test_preprocessor_init(self):
        """Test preprocessor initialization."""
        preprocessor = AudioPreprocessor(sample_rate=44100)
        assert preprocessor.sample_rate == 44100

    def test_get_audio_info(self, tmp_path):
        """Test getting audio file info."""
        # Create a simple test WAV file
        import numpy as np
        import soundfile as sf
        test_file = tmp_path / "test.wav"
        sf.write(str(test_file), np.random.randn(48000), 48000)

        preprocessor = AudioPreprocessor()
        info = preprocessor.get_audio_info(test_file)

        assert "duration" in info
        assert "sample_rate" in info
        assert info["sample_rate"] == 48000


class TestAudioMerger:
    """Tests for audio merger."""

    def test_volume_ratios(self):
        """Test default volume ratios."""
        from music_skills.merging.audio_merger import AudioMerger
        merger = AudioMerger()
        assert merger.volume_ratios["lead"] == 1.2
        assert merger.volume_ratios["harmony"] == 1.0
        assert merger.volume_ratios["accompaniment"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__])
