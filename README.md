# Music Skills CLI

Python CLI tool for personalized voice training and song covering using RVC (Retrieval-based Voice Conversion) and UVR (Ultimate Vocal Remover).

## Features

- **Train personalized voice models** using RVC architecture
- **Generate song covers** with converted voice
- **Audio separation** using UVR for vocals, accompaniment, and harmony
- **Audio preprocessing** with format conversion and normalization
- **Song search and download** from YouTube

## Installation

```bash
# Clone the repository
git clone https://github.com/baikai-li/music-skills.git
cd music-skills

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### Preprocessing Audio Sources

Prepare your voice source files for training:

```bash
music-skills preprocess audio /path/to/source/audio /path/to/output
```

Get audio file information:

```bash
music-skills preprocess info audio_file.wav
```

### Training a Voice Model

Train a new RVC voice model:

```bash
music-skills train model /path/to/audio_files model_name --epochs 100 --batch-size 4
```

### Audio Separation

Separate vocals from accompaniment:

```bash
music-skills separate vocals input.wav --output /path/to/output
```

Run full separation pipeline (vocals+accompaniment, then lead+harmony):

```bash
music-skills separate full input.wav --output /path/to/output --model HP2 --device cuda
```

### Generating Song Covers

Generate a song cover with converted voice:

```bash
music-skills cover generate song.wav model.pth output_cover.wav \
    --pitch-shift 0 \
    --f0method harvest \
    --lead-ratio 1.2 \
    --harmony-ratio 1.0 \
    --accompaniment-ratio 0.8
```

### Searching and Downloading Songs

Search for songs to cover:

```bash
music-skills cover search "song title"
```

Download a song:

```bash
music-skills cover download "https://youtube.com/watch?v=..."
```

### Configuration

Show current configuration:

```bash
music-skills config
```

## SOP Workflow

1. **Collect voice source files** - Place audio files in the source directory
2. **Preprocess audio** - Convert formats and normalize
3. **Train RVC model** - Train personalized voice model
4. **Search/download songs** - Find songs to cover
5. **Separate audio** - Extract vocals, lead, and harmony
6. **Convert voice** - Apply RVC model to lead vocals
7. **Merge final cover** - Mix converted vocals with harmony and accompaniment

## Configuration

Edit `config.yaml` to customize:

- Audio source directories
- RVC model parameters
- UVR separation settings
- Volume mixing ratios
- GitHub repository settings

Environment variables can also be used:
- `MUSIC_SKILLS_PATHS_AUDIO_SOURCE`
- `MUSIC_SKILLS_PATHS_MODELS`
- etc.

## Project Structure

```
music-skills/
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
├── pyproject.toml       # Project metadata
├── README.md            # This file
├── music_skills/        # Main package
│   ├── __init__.py
│   ├── cli/             # CLI interface
│   │   ├── main.py
│   │   └── song_downloader.py
│   ├── config/          # Configuration management
│   ├── preprocessing/   # Audio preprocessing
│   ├── training/        # RVC model training
│   ├── inference/       # RVC inference
│   ├── separation/     # UVR-based separation
│   └── merging/        # Audio merging
├── assets/
│   ├── models/         # Trained models
│   ├── output/        # Generated covers
│   └── separated/     # Separated audio files
├── docs/               # Documentation
└── tests/             # Unit tests
```

## Dependencies

- Python 3.10+
- Click - CLI framework
- PyYAML - Configuration
- Pydub - Audio processing
- Librosa - Audio analysis
- SoundFile - Audio I/O
- yt-dlp - YouTube downloader
- FFmpeg - Audio merging (required)

## References

- [RVC (Retrieval-based Voice Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [UVR (Ultimate Vocal Remover)](https://github.com/Anjok07/ultimatevocalremovergui)

## License

MIT License

## Author

Baikai Li - [https://github.com/baikai-li](https://github.com/baikai-li)
