"""
CLI interface for music-skills using Click.
"""

import logging
from pathlib import Path
from typing import Optional

import click

from .config import get_config, Config
from .preprocessing.audio import AudioPreprocessor, preprocess_source_files
from .training.rvc_trainer import RVCTrainer, train_voice_model
from .inference.rvc_inference import RVCInference, convert_voice
from .separation.uvr_separator import UVRSeparator, separate_audio
from .merging.audio_merger import AudioMerger, merge_cover
from .song_downloader import SongDownloader, search_songs, download_song

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--config", "-c", type=Path, help="Path to config file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx: click.Context, config: Optional[Path], verbose: bool) -> None:
    """Music Skills CLI - Voice conversion and song covering tool."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    cfg = get_config(config) if config else get_config()
    ctx.obj = cfg


@main.group()
def preprocess():
    """Audio preprocessing commands."""
    pass


@preprocess.command("audio")
@click.argument("source_dir", type=Path)
@click.argument("output_dir", type=Path)
@click.option("--sample-rate", default=48000, help="Target sample rate")
@click.pass_obj
def preprocess_audio(obj: Config, source_dir: Path, output_dir: Path,
                      sample_rate: int) -> None:
    """Preprocess audio source files for training.

    SOURCE_DIR: Directory containing source audio files
    OUTPUT_DIR: Directory for processed output
    """
    click.echo(f"Preprocessing audio from: {source_dir}")
    processed = preprocess_source_files(source_dir, output_dir, sample_rate)
    click.echo(f"Processed {len(processed)} files")


@preprocess.command("info")
@click.argument("audio_file", type=Path)
@click.pass_obj
def audio_info(obj: Config, audio_file: Path) -> None:
    """Get information about an audio file."""
    preprocessor = AudioPreprocessor(sample_rate=obj.sample_rate)
    info = preprocessor.get_audio_info(audio_file)

    click.echo(f"Audio Information: {audio_file.name}")
    for key, value in info.items():
        click.echo(f"  {key}: {value}")


@main.group()
def train():
    """Voice model training commands."""
    pass


@train.command("model")
@click.argument("audio_dir", type=Path)
@click.argument("model_name", type=str)
@click.option("--epochs", default=100, help="Number of training epochs")
@click.option("--batch-size", default=4, help="Batch size")
@click.option("--sample-rate", default=48000, help="Sample rate")
@click.pass_obj
def train_model(obj: Config, audio_dir: Path, model_name: str,
                epochs: int, batch_size: int, sample_rate: int) -> None:
    """Train a new RVC voice model.

    AUDIO_DIR: Directory containing training audio files
    MODEL_NAME: Name for the trained model
    """
    click.echo(f"Training RVC model: {model_name}")
    click.echo(f"Source: {audio_dir}")
    click.echo(f"Epochs: {epochs}, Batch size: {batch_size}")

    model_path = train_voice_model(
        audio_dir=audio_dir,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        sample_rate=sample_rate,
    )

    click.echo(f"Model trained: {model_path}")


@main.group()
def separate():
    """Audio separation commands (UVR)."""
    pass


@separate.command("full")
@click.argument("input_file", type=Path)
@click.option("--output-dir", "-o", type=Path, help="Output directory")
@click.option("--model", default="HP2", help="UVR model name")
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]))
@click.pass_obj
def separate_full(obj: Config, input_file: Path, output_dir: Optional[Path],
                  model: str, device: str) -> None:
    """Run full separation pipeline (vocals+accompaniment, lead+harmony).

    INPUT_FILE: Input audio file
    """
    click.echo(f"Separating: {input_file.name}")
    click.echo(f"Model: {model}, Device: {device}")

    result = separate_audio(
        input_path=input_file,
        output_dir=output_dir,
        model_name=model,
        device=device,
    )

    click.echo("Separation complete:")
    for key, path in result.items():
        click.echo(f"  {key}: {path.name}")


@separate.command("vocals")
@click.argument("input_file", type=Path)
@click.option("--output-dir", "-o", type=Path, help="Output directory")
@click.option("--model", default="HP2", help="UVR model name")
@click.pass_obj
def separate_vocals(obj: Config, input_file: Path, output_dir: Optional[Path],
                    model: str) -> None:
    """Separate vocals from accompaniment.

    INPUT_FILE: Input audio file
    """
    separator = UVRSeparator()
    vocals, accompaniment = separator.separate_vocals_accompaniment(
        input_file, output_dir, model
    )

    click.echo(f"Vocals: {vocals.name}")
    click.echo(f"Accompaniment: {accompaniment.name}")


@main.group()
def cover():
    """Song cover generation commands."""
    pass


@cover.command("generate")
@click.argument("song_file", type=Path)
@click.argument("model_path", type=Path)
@click.argument("output_file", type=Path)
@click.option("--pitch-shift", default=0, help="Pitch shift in semitones")
@click.option("--f0method", default="harvest", type=click.Choice(["harvest", "crepe", "dio"]))
@click.option("--lead-ratio", default=1.2, help="Lead vocals volume ratio")
@click.option("--harmony-ratio", default=1.0, help="Harmony volume ratio")
@click.option("--accompaniment-ratio", default=0.8, help="Accompaniment volume ratio")
@click.pass_obj
def generate_cover(obj: Config, song_file: Path, model_path: Path,
                   output_file: Path, pitch_shift: int, f0method: str,
                   lead_ratio: float, harmony_ratio: float,
                   accompaniment_ratio: float) -> None:
    """Generate a song cover with converted voice.

    SONG_FILE: Original song audio file
    MODEL_PATH: Path to trained RVC model
    OUTPUT_FILE: Output file path for the cover
    """
    click.echo(f"Generating cover: {song_file.name}")
    click.echo(f"Model: {model_path.name}")

    # Step 1: Separate song into vocals and accompaniment
    click.echo("Step 1: Separating vocals and accompaniment...")
    separator = UVRSeparator()
    temp_dir = output_file.parent / "temp_separation"
    vocals, accompaniment = separator.separate_vocals_accompaniment(
        song_file, temp_dir, device="cpu"
    )

    # Step 2: Separate lead and harmony
    click.echo("Step 2: Separating lead and harmony...")
    lead, harmony = separator.separate_harmony(vocals, temp_dir)

    # Step 3: Convert lead vocals
    click.echo("Step 3: Converting lead vocals with RVC model...")
    converted_lead = temp_dir / "converted_lead.wav"
    convert_voice(
        model_path=model_path,
        input_path=lead,
        output_path=converted_lead,
        pitch_shift=pitch_shift,
        f0method=f0method,
    )

    # Step 4: Merge final cover
    click.echo("Step 4: Merging final cover...")
    merge_cover(
        lead_path=converted_lead,
        harmony_path=harmony,
        accompaniment_path=accompaniment,
        output_path=output_file,
        lead_ratio=lead_ratio,
        harmony_ratio=harmony_ratio,
        accompaniment_ratio=accompaniment_ratio,
    )

    click.echo(f"Cover generated: {output_file}")


@cover.command("search")
@click.argument("query", type=str)
@click.option("--limit", default=10, help="Maximum results")
def search_cover(query: str, limit: int) -> None:
    """Search for songs to cover."""
    results = search_songs(query, limit)

    click.echo(f"Search results for '{query}':")
    for i, result in enumerate(results, 1):
        click.echo(f"{i}. {result['title']}")
        click.echo(f"   Duration: {result.get('duration', 'N/A')}s")
        click.echo(f"   URL: {result['url']}")


@cover.command("download")
@click.argument("url", type=str)
@click.option("--output", "-o", type=Path, help="Output file path")
def download_cover(url: str, output: Optional[Path]) -> None:
    """Download a song for covering."""
    path = download_song(url, output)
    click.echo(f"Downloaded: {path}")


@main.command()
def version():
    """Show version information."""
    from importlib.metadata import version
    try:
        ver = version("music-skills")
    except ImportError:
        ver = "0.1.0"
    click.echo(f"Music Skills CLI v{ver}")


@main.command()
@click.option("--output", "-o", type=Path, help="Output file path")
def config_show(output: Optional[Path]) -> None:
    """Show current configuration."""
    cfg = get_config()
    import yaml
    config_dict = {
        "audio_source_dir": str(cfg.audio_source_dir),
        "models_dir": str(cfg.models_dir),
        "output_dir": str(cfg.output_dir),
        "separated_dir": str(cfg.separated_dir),
        "sample_rate": cfg.sample_rate,
        "rvc_config": cfg.rvc_config,
        "uvr_config": cfg.uvr_config,
    }
    yaml_output = yaml.dump(config_dict, default_flow_style=False)

    if output:
        output.write_text(yaml_output)
        click.echo(f"Config saved to: {output}")
    else:
        click.echo(yaml_output)


if __name__ == "__main__":
    main()
