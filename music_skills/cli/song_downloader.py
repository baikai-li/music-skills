"""
Song downloader module for music-skills CLI.
Handles searching and downloading cover songs using yt-dlp.
"""

import logging
from pathlib import Path
from typing import Optional

import yt_dlp
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SongDownloader:
    """Song downloader using yt-dlp."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize song downloader.

        Args:
            output_dir: Default output directory for downloads.
        """
        self.output_dir = output_dir or Path("assets/downloads")

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search for songs.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of search results.
        """
        logger.info(f"Searching for: {query}")

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "playlist_items": f"1-{limit}",
        }

        results = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
            if "entries" in info:
                for entry in info["entries"]:
                    results.append({
                        "title": entry.get("title"),
                        "url": entry.get("webpage_url"),
                        "duration": entry.get("duration"),
                        "uploader": entry.get("uploader"),
                        "thumbnail": entry.get("thumbnail"),
                    })

        return results

    def download(self, url: str, output_path: Optional[Path] = None,
                 format: str = "bestaudio/best") -> Path:
        """Download a song.

        Args:
            url: YouTube URL.
            output_path: Output file path.
            format: Audio format to download.

        Returns:
            Path to downloaded file.
        """
        if output_path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / "download"

        logger.info(f"Downloading: {url}")

        ydl_opts = {
            "format": format,
            "outtmpl": str(output_path),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }],
            "quiet": False,
            "no_warnings": False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Rename to .wav if needed
        wav_path = Path(str(output_path) + ".wav")
        if not wav_path.exists():
            potential = list(self.output_dir.glob("*.wav"))
            if potential:
                wav_path = potential[0]

        logger.info(f"Download completed: {wav_path}")
        return wav_path

    def download_playlist(self, url: str, output_dir: Optional[Path] = None,
                          limit: int = 10) -> list[Path]:
        """Download multiple songs from playlist.

        Args:
            url: Playlist URL.
            output_dir: Output directory.
            limit: Maximum number of songs.

        Returns:
            List of downloaded file paths.
        """
        output_dir = output_dir or self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading playlist: {url}")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }],
            "playlist_items": f"1-{limit}",
            "quiet": True,
        }

        downloaded = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            # Get downloaded files
            downloaded = list(output_dir.glob("*.wav"))

        logger.info(f"Downloaded {len(downloaded)} songs from playlist")
        return downloaded


def search_songs(query: str, limit: int = 10) -> list[dict]:
    """Search for songs.

    Args:
        query: Search query.
        limit: Maximum results.

    Returns:
        List of search results.
    """
    downloader = SongDownloader()
    return downloader.search(query, limit)


def download_song(url: str, output_path: Optional[Path] = None) -> Path:
    """Download a song.

    Args:
        url: YouTube URL.
        output_path: Output file path.

    Returns:
        Path to downloaded file.
    """
    downloader = SongDownloader()
    return downloader.download(url, output_path)
