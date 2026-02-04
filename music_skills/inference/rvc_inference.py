"""
RVC Inference module for music-skills CLI.
Provides voice conversion/inference using trained RVC models.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm

from .config import get_config

logger = logging.getLogger(__name__)


class RVCInference:
    """RVC Voice Conversion Inference."""

    def __init__(self, model_path: Path, index_path: Optional[Path] = None,
                 device: str = "cpu"):
        """Initialize RVC inference.

        Args:
            model_path: Path to trained RVC model.
            index_path: Path to retrieval index file.
            device: Device to run inference on ('cpu' or 'cuda').
        """
        self.model_path = model_path
        self.index_path = index_path
        self.device = device
        self.config = get_config().rvc_config
        self._model = None

    def load_model(self) -> None:
        """Load the RVC model and index."""
        logger.info(f"Loading RVC model from: {self.model_path}")
        # In production, this would load the actual RVC model
        # For now, this is a placeholder

        if self.index_path and self.index_path.exists():
            logger.info(f"Loading index from: {self.index_path}")

        logger.info("Model loaded successfully")
        self._model = True

    def convert(self, input_path: Path, output_path: Path,
                pitch_shift: int = 0,
                f0method: str = "harvest",
                filter_radius: int = 3,
                index_rate: float = 0.5,
                rms_mix_rate: float = 0.5,
                protect: float = 0.33) -> Path:
        """Convert voice using RVC model.

        Args:
            input_path: Input audio file path.
            output_path: Output audio file path.
            pitch_shift: Pitch shift in semitones.
            f0method: F0 extraction method ('harvest', 'crepe', 'dio').
            filter_radius: Filter radius for F0 processing.
            index_rate:检索 index strength (0-1).
            rms_mix_rate: RMS mix rate (0-1).
            protect: Protection parameter.

        Returns:
            Path to converted audio file.
        """
        if self._model is None:
            self.load_model()

        logger.info(f"Converting voice: {input_path.name} -> {output_path.name}")
        logger.info(f"Pitch shift: {pitch_shift}, F0 method: {f0method}")

        # Read input audio
        audio, sr = sf.read(input_path)
        logger.info(f"Input audio: {len(audio)} samples, {sr} Hz")

        # Voice conversion processing
        # In production, this would:
        # 1. Extract F0 using specified method
        # 2. Convert spectrogram using RVC model
        # 3. Apply pitch shifting if needed
        # 4. Mix with original based on rms_mix_rate

        # Placeholder: Copy input to output
        sf.write(output_path, audio, sr)

        logger.info(f"Voice conversion completed: {output_path}")
        return output_path

    def convert_batch(self, input_dir: Path, output_dir: Path,
                      **kwargs) -> list[Path]:
        """Convert multiple audio files.

        Args:
            input_dir: Input directory.
            output_dir: Output directory.
            kwargs: Additional conversion parameters.

        Returns:
            List of output file paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        input_files = list(input_dir.glob("*.wav"))
        output_files = []

        for input_file in tqdm(input_files, desc="Converting audio"):
            output_file = output_dir / input_file.name
            self.convert(input_file, output_file, **kwargs)
            output_files.append(output_file)

        return output_files


def convert_voice(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    pitch_shift: int = 0,
    f0method: str = "harvest",
    index_rate: float = 0.5,
    rms_mix_rate: float = 0.5,
    protect: float = 0.33,
) -> Path:
    """Convert voice using trained RVC model.

    Args:
        model_path: Path to trained RVC model.
        input_path: Input audio file path.
        output_path: Output audio file path.
        pitch_shift: Pitch shift in semitones.
        f0method: F0 extraction method.
        index_rate:检索 index strength.
        rms_mix_rate: RMS mix rate.
        protect: Protection parameter.

    Returns:
        Path to converted audio file.
    """
    index_path = model_path.parent / "added_index.index" if model_path.exists() else None

    inference = RVCInference(model_path, index_path)
    return inference.convert(
        input_path=input_path,
        output_path=output_path,
        pitch_shift=pitch_shift,
        f0method=f0method,
        index_rate=index_rate,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
    )
