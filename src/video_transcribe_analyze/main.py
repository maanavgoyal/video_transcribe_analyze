import os
import argparse
import whisper
import torch
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def transcribe_video(video_path):
    model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
    result = model.transcribe(video_path)
    return result["text"]

def analyze_content(transcript):
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""This is a transcript of a video:

    {transcript}

    You are an assistant who is very good at analyzing content for swear words, personal information, specific incidents, sensitive information.

    Go through the transcript and flag out swear words, specific incidents, personal information,  
    
    statements that show certain places or people in bad light.

    Also, return the exact sentences which are to be flagged along with the reason
"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=8192,
        # system="You are an assistant who is very good at analyzing content for personal information, politically incorrect content, sensitive information, or swear words.",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return message.content[0].text

def write_analysis_to_file(video_path, transcript, analysis):
    # Create a filename based on the input video name
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{base_name}_analysis.txt"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"Analysis for video: {video_path}\n\n")
        f.write("\n\nContent Analysis:\n")
        f.write(analysis)
    
    return output_filename

def main(video_path):
    if not ANTHROPIC_API_KEY:
        raise ValueError("Anthropic API key not found. Please set it in your .env file.")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The video file '{video_path}' does not exist.")
    
    transcript = transcribe_video(video_path)
    analysis = analyze_content(transcript)
    output_file = write_analysis_to_file(video_path, transcript, analysis)

def cli():
    parser = argparse.ArgumentParser(description="Transcribe and analyze a video.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()
    main(args.video_path)

if __name__ == "__main__":
    cli()