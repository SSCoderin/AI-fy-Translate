# Video Translator: Hindi to Marathi

This project automatically translates Hindi videos to Marathi by:

1. Extracting frames from the video
2. Using OCR to detect Hindi text in frames
3. Translating Hindi text to Marathi using OpenAI GPT-4
4. Overlaying translated Marathi text on frames
5. Converting audio to Marathi using Google Text-to-Speech
6. Creating a final video with synchronized audio

## Prerequisites

### System Requirements

- Python 3.8+
- FFmpeg (for video/audio processing)
- Font files for Devanagari script

### API Keys Required

1. **OpenAI API Key**: For text translation and audio transcription
2. **Google Cloud Service Account**: For Text-to-Speech (Marathi voice)

## Installation

### 1. Install System Dependencies

#### On Windows:

```bash
# Install FFmpeg using chocolatey (or download from https://ffmpeg.org)
choco install ffmpeg
```

#### On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

#### On macOS:

```bash
brew install ffmpeg
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Font Files

The Noto Sans Devanagari fonts are already included in the `noto-sans-devanagari/` directory.

## Configuration

### 1. Set up Google Cloud Credentials

1. Create a Google Cloud project
2. Enable Text-to-Speech API
3. Create a service account and download the JSON key
4. Save the JSON file in your project directory

### 2. Get OpenAI API Key

1. Sign up at [OpenAI](https://platform.openai.com)
2. Generate an API key from the dashboard

### 3. Create .env Configuration File

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
# Edit .env with your actual API keys and paths
```

Your `.env` file should look like:

```env
OPENAI_API_KEY=sk-proj-your-actual-openai-api-key
GOOGLE_CREDENTIALS_PATH=sampark-ai-bc13b9af3b55.json
```

### 4. Alternative: Create Configuration File

You can also use the Python config approach:

```bash
cp config_example.py config.py
# Edit config.py with your actual API keys and paths
```

## Usage

### 🚀 FastAPI Server (Recommended for Production)

**Step 1: Start the API Server**

```bash
python api.py
```

**Step 2: Use the API**

```bash
# Start translation
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "input_video.mp4"}'

# Check status  
curl "http://localhost:8000/status/YOUR_JOB_ID"

# Download result
curl -o "translated.mp4" "http://localhost:8000/download/YOUR_JOB_ID"
```

**Python Client:**

```python
from api_client_example import VideoTranslationClient

client = VideoTranslationClient()
output_video = client.translate_and_wait("input_video.mp4")
print(f"Translated video: {output_video}")
```

**Interactive Documentation:** http://localhost:8000/docs

### Command Line Usage

#### Basic Usage:

```bash
python video_translator.py "path/to/your/video.mp4" \
  --openai-key "your_openai_api_key" \
  --google-creds "path/to/google-credentials.json"
```

#### Advanced Usage:

```bash
python video_translator.py "path/to/your/video.mp4" \
  --openai-key "your_openai_api_key" \
  --google-creds "path/to/google-credentials.json" \
  --font-path "custom/font/path.ttf" \
  --no-cleanup  # Keep temporary files for debugging
```

### Python Function Usage (Recommended)

**Step 1: Create .env file**

```bash
# Copy the example file
cp .env.example .env
# Edit .env with your actual credentials
```

**Step 2: Use the function**

```python
from video_translator import translate_video

# Simple usage - credentials loaded from .env
output_video = translate_video("input_video.mp4")
print(f"Translated video: {output_video}")

# Advanced usage with custom parameters
output_video = translate_video(
    video_path="my_video.mp4",
    font_path="custom-font.ttf",
    cleanup=False,  # Keep temp files
    env_file="custom.env"
)
```

### Python Class Usage (Alternative)

```python
from video_translator import VideoTranslator

# Initialize translator
translator = VideoTranslator(
    openai_api_key="your_openai_api_key",
    google_credentials_path="path/to/google-credentials.json",
    font_path="noto-sans-devanagari/NotoSansDevanagari-Regular.ttf"
)

# Process video
output_video_path = translator.process_video("input_video.mp4")
print(f"Translated video saved to: {output_video_path}")
```

## Project Structure

```
video-translation/
├── api.py                     # FastAPI server (recommended)
├── api_client_example.py      # Python API client
├── api_usage.md              # Detailed API documentation
├── video_translator.py        # Main translation script
├── translate_video()          # Main function (in video_translator.py)
├── .env.example              # Environment variables template
├── config_example.py         # Alternative configuration
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── noto-sans-devanagari/     # Font files directory
├── frames/                   # Temporary frames (auto-generated)
├── translated_frames/        # Processed frames (auto-generated)
└── output_*.mp4             # Final translated videos
```

## How It Works

1. **Frame Extraction**: Uses FFmpeg to extract frames from input video at 30 FPS
2. **OCR Processing**: EasyOCR detects Hindi text in each frame and tracks consistent text regions
3. **Text Translation**: OpenAI GPT-4 translates Hindi text to Marathi
4. **Frame Processing**: Overlays Marathi text on frames, replacing Hindi text with white background
5. **Audio Processing**: Transcribes original Hindi audio and translates to Marathi
6. **TTS Generation**: Google Cloud TTS generates Marathi audio
7. **Video Assembly**: FFmpeg combines processed frames with Marathi audio

## Features

- **Smart Text Tracking**: Tracks text regions across frames to maintain consistency
- **Font Auto-sizing**: Automatically adjusts font size to fit bounding boxes
- **Audio Synchronization**: Matches video duration with generated audio
- **Cleanup Management**: Automatically removes temporary files (optional)
- **Error Handling**: Comprehensive error handling with descriptive messages

## Troubleshooting

### Common Issues

1. **FFmpeg not found**

   - Ensure FFmpeg is installed and in your system PATH
   - Test with: `ffmpeg -version`
2. **Google Cloud Authentication Error**

   - Check if your service account JSON file is valid
   - Ensure Text-to-Speech API is enabled in your Google Cloud project
3. **Font rendering issues**

   - Verify Devanagari font files exist in the specified path
   - Try using different font files from the `noto-sans-devanagari/` directory
4. **OpenAI API Errors**

   - Verify your API key is active and has sufficient credits
   - Check rate limits if processing multiple videos

### Debug Mode

Use the `--no-cleanup` flag to keep temporary files for debugging:

```bash
python video_translator.py input.mp4 --openai-key "key" --google-creds "creds.json" --no-cleanup
```

This will preserve:

- `frames/video_id/` - Original extracted frames
- `translated_frames/video_id/` - Processed frames with Marathi text
- `audio_video_id.wav` - Original audio
- `marathi_audio_video_id.mp3` - Translated Marathi audio

## Performance Tips

- **Processing Time**: Expect 5-10 minutes per minute of video (depending on text density)
- **API Costs**: OpenAI costs depend on text amount; Google TTS charges per character
- **Memory Usage**: Large videos may require 4-8GB RAM for processing
- **Disk Space**: Temporary files can use 2-3x the original video size

## License

This project is for educational and research purposes. Please ensure you have proper licenses for:

- Font files used
- Videos being processed
- API services utilized

## Contributing

Feel free to submit issues and pull requests to improve the translation quality and performance.
