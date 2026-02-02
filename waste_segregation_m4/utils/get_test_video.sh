#!/bin/bash
# Quick script to download a sample conveyor belt video for testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VIDEOS_DIR="$PROJECT_ROOT/test_videos"
mkdir -p "$VIDEOS_DIR"

echo "📹 Downloading Sample Conveyor Belt Video..."
echo ""

# Check if yt-dlp is installed
if command -v yt-dlp &> /dev/null; then
    echo "✅ yt-dlp found. Downloading sample video from YouTube..."
    echo ""
    
    # Sample YouTube video URL (replace with actual conveyor belt video)
    # This is a placeholder - you'll need to find a real conveyor belt video
    echo "⚠️  Please provide a YouTube URL for a conveyor belt video."
    echo "   Example search terms: 'conveyor belt sorting', 'industrial conveyor'"
    echo ""
    read -p "Enter YouTube URL (or press Enter to skip): " YOUTUBE_URL
    
    if [ -n "$YOUTUBE_URL" ]; then
        echo "Downloading..."
        yt-dlp -f 'best[ext=mp4]' -o "$VIDEOS_DIR/conveyor_test.mp4" "$YOUTUBE_URL"
        echo ""
        echo "✅ Video downloaded to: $VIDEOS_DIR/conveyor_test.mp4"
    else
        echo "Skipped download."
    fi
else
    echo "ℹ️  yt-dlp not installed."
    echo "   Install with: pip install yt-dlp"
    echo ""
    echo "📋 Alternative: Download manually from:"
    echo "   1. Pexels: https://www.pexels.com/search/conveyor%20belt/"
    echo "   2. Pixabay: https://pixabay.com/videos/search/conveyor%20belt/"
    echo ""
    echo "   Save the video to: $VIDEOS_DIR/"
fi

echo ""
echo "📁 Test videos directory: $VIDEOS_DIR"
echo ""
echo "To use in Streamlit:"
echo "  1. Select 'Upload Conveyor Video' mode in sidebar"
echo "  2. Upload the video file"
echo "  3. System will process automatically!"
