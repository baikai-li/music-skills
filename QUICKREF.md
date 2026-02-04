# Music Skills CLI - Quick Reference

## Installation

```bash
# Install dependencies and package
./setup.sh

# Verify installation
music-skills --help
```

## Preprocessing

```bash
# Preprocess source audio files
music-skills preprocess audio /Users/mindstorm/Projects/Virtual-IP-DB/audios/韩立 ./processed

# Get audio file info
music-skills preprocess info audio_file.wav
```

## Training

```bash
# Train a new voice model
music-skills train model ./processed hanli_model --epochs 100 --batch-size 4
```

## Separation

```bash
# Full separation pipeline
music-skills separate full song.wav --output ./separated --model HP2 --device cuda

# Separate vocals only
music-skills separate vocals song.wav --output ./separated --model HP2
```

## Cover Generation

```bash
# Generate a song cover
music-skills cover generate song.wav model.pth output_cover.wav \
    --pitch-shift 0 \
    --f0method harvest \
    --lead-ratio 1.2 \
    --harmony-ratio 1.0 \
    --accompaniment-ratio 0.8
```

## Song Search & Download

```bash
# Search for songs
music-skills cover search "song title"

# Download a song
music-skills cover download "https://youtube.com/watch?v=..."
```

## Configuration

```bash
# Show current config
music-skills config
# Or use config.yaml to customize paths and parameters
```

## GitHub Integration

```bash
# Initialize GitHub repository (requires token)
./init_github.sh YOUR_GITHUB_TOKEN

# Commit and push changes
git add -A
git commit -m "Your commit message"
git push
```

## SOP Workflow

```
1. Collect source audio → preprocess audio
2. Train RVC model → train model
3. Find song to cover → cover search/download
4. Separate vocals/accompaniment → separate full
5. Separate lead/harmony → separate full
6. Convert lead vocals → cover generate
7. Merge final cover → cover generate
```

## Directory Structure

```
music-skills/
├── config.yaml           # Configuration
├── requirements.txt      # Dependencies
├── pyproject.toml       # Project metadata
├── README.md            # Documentation
├── setup.sh             # Installation script
├── init_github.sh       # GitHub init script
├── music_skills/        # Main package
│   ├── cli/             # CLI commands
│   ├── config/          # Config management
│   ├── preprocessing/  # Audio preprocessing
│   ├── training/        # RVC training
│   ├── inference/       # RVC inference
│   ├── separation/      # UVR separation
│   └── merging/         # Audio merging
└── assets/              # Models and outputs
```
