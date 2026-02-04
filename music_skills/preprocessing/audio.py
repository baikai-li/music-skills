"""
Audio preprocessing module for music-skills CLI.
Handles format conversion, normalization, and audio preparation.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing handler."""

    SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

    def __init__(self, sample_rate: int = 48000, normalize: bool = True):
        """Initialize audio preprocessor.

        Args:
            sample_rate: Target sample rate.
            normalize: Whether to normalize audio.
        """
        self.sample_rate = sample_rate
        self.normalize = normalize

    def convert_format(self, input_path: Path, output_path: Optional[Path] = None) -> Path:
        """Convert audio to WAV format.

        Args:
            input_path: Input audio file path.
            output_path: Output file path (optional).

        Returns:
            Path to converted file.
        """
        if output_path is None:
            output_path = input_path.with_suffix(".wav")

        input_format = input_path.suffix.lower()
        if input_format == ".mp3":
            audio = AudioSegment.from_mp3(input_path)
        elif input_format == ".flac":
            audio = AudioSegment.from_file(input_path, format="flac")
        elif input_format == ".ogg":
            audio = AudioSegment.from_ogg(input_path)
        elif input_format == ".m4a":
            audio = AudioSegment.from_file(input_path, format="m4a")
        elif input_format == ".aac":
            audio = AudioSegment.from_file(input_path, format="aac")
        else:
            audio = AudioSegment.from_wav(input_path)

        # Convert to target sample rate and mono
        audio = audio.set_frame_rate(self.sample_rate)
        audio = audio.set_channels(1)

        # Export as WAV
        audio.export(output_path, format="wav")
        logger.info(f"Converted {input_path.name} -> {output_path.name}")
        return output_path

    def normalize_audio(self, input_path: Path, output_path: Optional[Path] = None,
                        target_dbfs: float = -3.0) -> Path:
        """Normalize audio to target loudness.

        Args:
            input_path: Input audio file path.
            output_path: Output file path (optional).
            target_dbfs: Target loudness in dBFS.

        Returns:
            Path to normalized file.
        """
        if output_path is None:
            output_path = input_path.with_suffix(".normalized.wav")

        audio = AudioSegment.from_wav(input_path)
        current_dbfs = audio.dBFS

        if current_dbfs < target_dbfs:
            gain = target_dbfs - current_dbfs
            audio = audio.apply_gain(gain)
        elif current_dbfs > target_dbfs:
            # Soft limiter to prevent clipping
            audio = audio.apply_gain(-(current_dbfs - target_dbfs))

        audio.export(output_path, format="wav")
        logger.info(f"Normalized {input_path.name} (dBFS: {current_dbfs:.1f} -> {target_dbfs:.1f})")
        return output_path

    def trim_silence(self, input_path: Path, output_path: Optional[Path] = None,
                     threshold_db: float = -50, min_duration: float = 0.5) -> Path:
        """Remove silence from beginning and end of audio.

        Args:
            input_path: Input audio file path.
            output_path: Output file path (optional).
            threshold_db: Silence threshold in dB.
            min_duration: Minimum duration to keep in seconds.

        Returns:
            Path to trimmed file.
        """
        if output_path is None:
            output_path = input_path.with_suffix(".trimmed.wav")

        audio = AudioSegment.from_wav(input_path)

        # Trim silence from beginning
        start_trim = self._detect_leading_silence(audio, threshold_db)
        # Trim silence from end
        end_trim = len(audio) - self._detect_leading_silence(audio.reverse(), threshold_db)

        # Ensure minimum duration
        if len(audio) - start_trim - (len(audio) - end_trim) < min_duration * 1000:
            start_trim = 0
            end_trim = len(audio)

        trimmed_audio = audio[start_trim:end_trim]
        trimmed_audio.export(output_path, format="wav")
        logger.info(f"Trimmed silence from {input_path.name} ({start_trim}ms start, {len(audio) - end_trim}ms end)")
        return output_path

    def _detect_leading_silence(self, audio: AudioSegment, threshold_db: float) -> int:
        """Detect leading silence in audio.

        Args:
            audio: Audio segment.
            threshold_db: Silence threshold in dB.

        Returns:
            Length of leading silence in milliseconds.
        """
        chunk_size = 10  # ms
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if chunk.dBFS > threshold_db:
                return i
        return len(audio)

    def split_into_chunks(self, input_path: Path, output_dir: Path,
                          chunk_duration: float = 10.0,
                          overlap: float = 0.5) -> list[Path]:
        """Split audio into chunks.

        Args:
            input_path: Input audio file path.
            output_dir: Output directory for chunks.
            chunk_duration: Duration of each chunk in seconds.
            overlap: Overlap between chunks in seconds.

        Returns:
            List of paths to chunk files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        audio = AudioSegment.from_wav(input_path)
        chunk_ms = int(chunk_duration * 1000)
        overlap_ms = int(overlap * 1000)
        step_ms = chunk_ms - overlap_ms

        chunks = []
        for i in range(0, len(audio), step_ms):
            chunk = audio[i:i + chunk_ms]
            if len(chunk) > 1000:  # Minimum 1 second
                output_path = output_dir / f"{input_path.stem}_chunk_{i // step_ms:04d}.wav"
                chunk.export(output_path, format="wav")
                chunks.append(output_path)

        logger.info(f"Split {input_path.name} into {len(chunks)} chunks")
        return chunks

    def merge_audio_files(self, input_files: list[Path], output_path: Path) -> Path:
        """Merge multiple audio files into one.

        Args:
            input_files: List of audio file paths.
            output_path: Output file path.

        Returns:
            Path to merged file.
        """
        if not input_files:
            raise ValueError("No input files provided")

        combined = AudioSegment.from_wav(input_files[0])
        for file_path in input_files[1:]:
            audio = AudioSegment.from_wav(file_path)
            combined += audio

        combined.export(output_path, format="wav")
        logger.info(f"Merged {len(input_files)} files into {output_path.name}")
        return output_path

    def get_audio_info(self, input_path: Path) -> dict:
        """Get audio file information.

        Args:
            input_path: Audio file path.

        Returns:
            Dictionary with audio information.
        """
        audio = AudioSegment.from_wav(input_path)
        data, sr = sf.read(input_path)

        return {
            "duration": len(audio) / 1000.0,
            "sample_rate": sr,
            "channels": audio.channels,
            "dBFS": audio.dBFS,
            "max_amplitude": float(np.max(np.abs(data))),
            "file_size": input_path.stat().st_size,
        }


def preprocess_source_files(source_dir: Path, output_dir: Path,
                            sample_rate: int = 48000) -> list[Path]:
    """Preprocess all source audio files in directory.

    Args:
        source_dir: Source audio directory.
        output_dir: Output directory.
        sample_rate: Target sample rate.

    Returns:
        List of processed file paths.
    """
    preprocessor = AudioPreprocessor(sample_rate=sample_rate)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_files = []
    audio_files = [
        f for f in source_dir.iterdir()
        if f.suffix.lower() in preprocessor.SUPPORTED_FORMATS
    ]

    for audio_file in tqdm(audio_files, desc="Preprocessing audio files"):
        try:
            # Convert to WAV
            wav_path = output_dir / f"{audio_file.stem}.wav"
            processed = preprocessor.convert_format(audio_file, wav_path)

            # Normalize
            normalized_path = output_dir / f"{audio_file.stem}.normalized.wav"
            processed = preprocessor.normalize_audio(processed, normalized_path)

            processed_files.append(processed)
        except Exception as e:
            logger.error(f"Failed to process {audio_file.name}: {e}")

    return processed_files
