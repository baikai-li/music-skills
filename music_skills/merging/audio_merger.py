"""
Audio Merging module for music-skills CLI.
Handles merging of converted vocals with accompaniment using ffmpeg.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from pydub import AudioSegment
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AudioMerger:
    """Audio merging handler using ffmpeg."""

    def __init__(self):
        """Initialize audio merger."""
        self.volume_ratios = {
            "lead": 1.2,
            "harmony": 1.0,
            "accompaniment": 0.8,
        }

    def adjust_volume(self, audio_path: Path, output_path: Path,
                      volume_db: float = 0.0) -> Path:
        """Adjust audio volume.

        Args:
            audio_path: Input audio file path.
            output_path: Output audio file path.
            volume_db: Volume adjustment in dB.

        Returns:
            Path to adjusted audio file.
        """
        audio = AudioSegment.from_wav(audio_path)
        adjusted = audio + volume_db
        adjusted.export(output_path, format="wav")
        logger.info(f"Adjusted volume: {volume_db} dB -> {output_path.name}")
        return output_path

    def apply_volume_ratio(self, audio_path: Path, output_path: Path,
                           ratio: float) -> Path:
        """Apply volume ratio to audio.

        Args:
            audio_path: Input audio file path.
            output_path: Output audio file path.
            ratio: Volume ratio.

        Returns:
            Path to adjusted audio file.
        """
        # Convert ratio to dB: ratio = 10^(dB/20)
        volume_db = 20 * (ratio ** 0.5 - 1)
        return self.adjust_volume(audio_path, output_path, volume_db)

    def merge_tracks(self, track_paths: list[Path], output_path: Path,
                     volumes: Optional[list] = None) -> Path:
        """Merge multiple audio tracks into one.

        Args:
            track_paths: List of audio file paths.
            output_path: Output file path.
            volumes: Optional list of volume adjustments for each track.

        Returns:
            Path to merged file.
        """
        if not track_paths:
            raise ValueError("No tracks to merge")

        if volumes is None:
            volumes = [0.0] * len(track_paths)

        # Load and mix all tracks
        combined = None
        for track_path, volume in zip(track_paths, volumes):
            audio = AudioSegment.from_wav(track_path)
            if volume != 0:
                audio = audio + volume
            if combined is None:
                combined = audio
            else:
                combined = combined.overlay(audio)

        if combined is not None:
            combined.export(output_path, format="wav")
            logger.info(f"Merged {len(track_paths)} tracks into: {output_path.name}")

        return output_path

    def merge_with_ffmpeg(self, track_paths: list[Path], output_path: Path,
                          filter_complex: Optional[str] = None) -> Path:
        """Merge audio tracks using ffmpeg.

        Args:
            track_paths: List of audio file paths.
            output_path: Output file path.
            filter_complex: FFmpeg filter complex string.

        Returns:
            Path to merged file.
        """
        cmd = ["ffmpeg", "-y"]

        # Add input files
        for track_path in track_paths:
            cmd.extend(["-i", str(track_path)])

        # Build filter complex for mixing
        if filter_complex is None:
            # Simple mix
            num_tracks = len(track_paths)
            inputs_part = "".join(f"[{i}:0]" for i in range(num_tracks))
            filter_str = f"{inputs_part}amix=inputs={num_tracks}:duration=longest"
            cmd.extend(["-filter_complex", filter_str])

        cmd.extend([
            "-ac", "2",  # Stereo output
            "-ar", "48000",  # Sample rate
            str(output_path),
        ])

        logger.info(f"Running ffmpeg merge: {' '.join(cmd[:4])}...")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        logger.info(f"FFmpeg merge completed: {output_path.name}")
        return output_path

    def merge_song_cover(self, lead_path: Path, harmony_path: Path,
                         accompaniment_path: Path, output_path: Path,
                         lead_ratio: float = 1.2,
                         harmony_ratio: float = 1.0,
                         accompaniment_ratio: float = 0.8) -> Path:
        """Merge converted lead vocals, harmony, and accompaniment.

        Args:
            lead_path: Path to converted lead vocals.
            harmony_path: Path to harmony vocals.
            accompaniment_path: Path to accompaniment.
            output_path: Output file path.
            lead_ratio: Volume ratio for lead vocals.
            harmony_ratio: Volume ratio for harmony.
            accompaniment_ratio: Volume ratio for accompaniment.

        Returns:
            Path to final merged song.
        """
        logger.info("Merging song cover components:")
        logger.info(f"  Lead ratio: {lead_ratio}")
        logger.info(f"  Harmony ratio: {harmony_ratio}")
        logger.info(f"  Accompaniment ratio: {accompaniment_ratio}")

        # Create temporary adjusted files
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Adjust volumes
            lead_adjusted = tmpdir / "lead_adjusted.wav"
            harmony_adjusted = tmpdir / "harmony_adjusted.wav"
            acc_adjusted = tmpdir / "acc_adjusted.wav"

            self.apply_volume_ratio(lead_path, lead_adjusted, lead_ratio)
            self.apply_volume_ratio(harmony_path, harmony_adjusted, harmony_ratio)
            self.apply_volume_ratio(accompaniment_path, acc_adjusted, accompaniment_ratio)

            # Merge using ffmpeg
            result = self.merge_with_ffmpeg(
                [lead_adjusted, harmony_adjusted, acc_adjusted],
                output_path,
            )

        logger.info(f"Song cover created: {output_path}")
        return result


def merge_cover(lead_path: Path, harmony_path: Path,
                accompaniment_path: Path, output_path: Path,
                lead_ratio: float = 1.2,
                harmony_ratio: float = 1.0,
                accompaniment_ratio: float = 0.8) -> Path:
    """Merge converted vocals and accompaniment into final cover.

    Args:
        lead_path: Path to converted lead vocals.
        harmony_path: Path to harmony vocals.
        accompaniment_path: Path to accompaniment.
        output_path: Output file path.
        lead_ratio: Volume ratio for lead vocals.
        harmony_ratio: Volume ratio for harmony.
        accompaniment_ratio: Volume ratio for accompaniment.

    Returns:
        Path to final merged cover.
    """
    merger = AudioMerger()
    return merger.merge_song_cover(
        lead_path=lead_path,
        harmony_path=harmony_path,
        accompaniment_path=accompaniment_path,
        output_path=output_path,
        lead_ratio=lead_ratio,
        harmony_ratio=harmony_ratio,
        accompaniment_ratio=accompaniment_ratio,
    )
