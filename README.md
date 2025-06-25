# Video Transcribe and Analyze

This project transcribes video content using Hugging Face's Transformers library (with Whisper model) and analyzes the transcript for politically incorrect or sensitive information using various LLMs.

## Features

- Transcribe video files using Hugging Face's Transformers (Whisper model)
- Analyze transcripts for sensitive content
- Write analysis results to a text file
- Easy-to-use command-line interface

## Prerequisites

1. Install `uv` (if not already installed):

   ```bash
   pip install uv
   ```

2. Ensure you have FFmpeg installed and added to your system PATH.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/video-transcribe-analyze.git
   cd video-transcribe-analyze
   ```

2. Create and activate a virtual environment using `uv`:

   ```bash
   uv venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. Install the required packages:

   ```bash
   uv sync
   ```

4. Set up your API keys:
   - Create a copy of the `.env.example` file and name it `.env`:
     ```bash
     cp .env.example .env
     ```
   - Open the `.env` file and replace `your_api_key_here` with your actual API keys

## Usage

Run the script with a video file as an argument:

```bash
uv run src/main.py path/to/your/video.mp3
```

The script will create a text file in the current directory with the transcript and analysis results. The filename will be `[original_video_name]_analysis.txt`.

## Development

To add new dependencies:

```bash
uv add package-name
```

To add development dependencies:

```bash
uv add --dev package-name
```

To update dependencies:

```bash
uv sync --upgrade
```

## Troubleshooting

If you encounter issues with package installation or compatibility:

1. Ensure you're using a recent version of Python (3.10 or later is required).
2. Make sure `uv` is properly installed and in your PATH.
3. Ensure you have FFmpeg installed and added to your system PATH.
4. If you're using a GPU and encounter CUDA-related issues, ensure that your CUDA toolkit version is compatible with the installed PyTorch version.

## Credit

Project idea from [Divide-By-0](https://github.com/Divide-By-0/) and [MIT SOUL](http://soul.mit.edu/)
