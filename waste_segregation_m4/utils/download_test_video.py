#!/usr/bin/env python3
"""
Utility script to download a sample conveyor belt video for testing.
Downloads from a reliable source or provides instructions.
"""

import os
import sys
import subprocess
from pathlib import Path

# Sample video URLs (public domain or free to use)
SAMPLE_VIDEOS = {
    "conveyor_belt_1": {
        "url": "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4",
        "description": "Sample video (replace with actual conveyor belt video)",
        "note": "This is a placeholder - you'll need to find an actual conveyor belt video"
    }
}

def download_video(url: str, output_path: str) -> bool:
    """Download video using wget or curl."""
    try:
        # Try wget first
        result = subprocess.run(
            ["wget", "-O", output_path, url],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    
    try:
        # Try curl as fallback
        result = subprocess.run(
            ["curl", "-L", "-o", output_path, url],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    
    return False

def main():
    """Main function to download test video."""
    print("=" * 60)
    print("Conveyor Belt Test Video Downloader")
    print("=" * 60)
    print()
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    videos_dir = project_root / "test_videos"
    videos_dir.mkdir(exist_ok=True)
    
    print("📹 Sample Conveyor Belt Video Sources:")
    print()
    print("1. **Pexels** (Free stock videos):")
    print("   https://www.pexels.com/search/conveyor%20belt/")
    print("   - Search for 'conveyor belt'")
    print("   - Download free videos")
    print()
    print("2. **Pixabay** (Free stock videos):")
    print("   https://pixabay.com/videos/search/conveyor%20belt/")
    print("   - Search for 'conveyor belt'")
    print("   - Download free videos")
    print()
    print("3. **YouTube** (Use yt-dlp to download):")
    print("   Search for: 'conveyor belt sorting', 'industrial conveyor', 'waste sorting'")
    print("   Install: pip install yt-dlp")
    print("   Download: yt-dlp -f 'best[ext=mp4]' <youtube_url>")
    print()
    print("4. **Sample Video Direct Links:**")
    print("   - Look for videos showing items moving on a conveyor belt")
    print("   - Ensure they're free to use or in public domain")
    print()
    
    print("=" * 60)
    print("Recommended: Use Pexels or Pixabay for free, high-quality videos")
    print("=" * 60)
    print()
    
    # Check if yt-dlp is available
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        print("✅ yt-dlp is installed. You can download YouTube videos.")
        print()
        print("Example command:")
        print("  yt-dlp -f 'best[ext=mp4]' -o test_videos/conveyor_test.mp4 <youtube_url>")
        print()
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ℹ️  yt-dlp not installed. Install with: pip install yt-dlp")
        print()
    
    print(f"📁 Videos will be saved to: {videos_dir}")
    print()
    print("After downloading, use the video in Streamlit:")
    print("  1. Go to sidebar → Select 'Upload Conveyor Video' mode")
    print("  2. Click 'Upload Video File'")
    print("  3. Select your downloaded video")
    print("  4. The system will process it automatically!")
    print()

if __name__ == "__main__":
    main()
