# Video Transcribe and Analyze

This project transcribes video content using Hugging Face's Transformers library (with Whisper model) and analyzes the transcript for politically incorrect or sensitive information using Anthropic's Claude 3.5 Sonnet.

## Features

- Transcribe video files using Hugging Face's Transformers (Whisper model)
- Analyze transcripts for sensitive content using Claude 3.5 Sonnet
- Write analysis results to a text file
- Easy-to-use command-line interface

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/video-transcribe-analyze.git
   cd video-transcribe-analyze
   ```

2. Create and activate a virtual environment:

   - For Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```

   - For macOS and Linux:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

   If you encounter any issues, try upgrading pip first:
   ```
   pip install --upgrade pip
   ```
   Then retry the installation.

4. Set up your Anthropic API key:
   - Create a copy of the `.env.example` file and name it `.env`:
     - For Windows:
       ```
       copy .env.example .env
       ```
     - For macOS and Linux:
       ```
       cp .env.example .env
       ```
   - Open the `.env` file and replace `your_api_key_here` with your actual Anthropic API key:
     ```
     ANTHROPIC_API_KEY=your_actual_api_key
     ```

## Usage

Run the script with a video file as an argument:

- For Windows:
  ```
  python src\video_transcribe_analyze\main.py path\to\your\video.mp4
  ```

- For macOS and Linux:
  ```
  python src/video_transcribe_analyze/main.py /path/to/your/video.mp4
  ```

The script will create a text file in the current directory with the transcript and analysis results. The filename will be `[original_video_name]_analysis.txt`.

## Troubleshooting

If you encounter issues with package installation or compatibility:

1. Ensure you're using a recent version of Python (3.8 or later is recommended).
2. Try upgrading pip before installing the requirements:
   ```
   pip install --upgrade pip
   ```
3. Ensure you have FFmpeg installed and added to your system PATH.

4. If you're using a GPU and encounter CUDA-related issues, ensure that your CUDA toolkit version is compatible with the installed PyTorch version.

## Credit
Project idea from [Divide-By-0](https://github.com/Divide-By-0/)
