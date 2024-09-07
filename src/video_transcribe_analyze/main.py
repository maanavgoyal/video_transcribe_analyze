import os
import argparse
import whisper
import torch
from anthropic import Anthropic
from dotenv import load_dotenv
import google.generativeai as genai
import openai

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def transcribe_video(video_path):
    # Create a filename for the transcript
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    transcript_filename = f"{base_name}_transcript.txt"
    
    # Check if the transcript file already exists
    if os.path.exists(transcript_filename):
        with open(transcript_filename, 'r', encoding='utf-8') as f:
            return f.read()
    
    # If not, transcribe the video
    model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
    result = model.transcribe(video_path)
    transcript = result["text"]
    
    # Save the transcript
    with open(transcript_filename, 'w', encoding='utf-8') as f:
        f.write(transcript)
    
    return transcript

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def analyze_content(transcript, llm_choice):
    example_flags = read_file(r"C:\Users\hp\video_transcribe_analyze\src\prompts\human_flags_1017.txt")
    claude_flags = read_file(r"C:\Users\hp\video_transcribe_analyze\src\prompts\claude_flags_1017.txt")
    example_transcript = read_file(r"C:\Users\hp\video_transcribe_analyze\src\prompts\transcript_1017.txt")

    prompt1 = f"""

    Go through the entire transcript and point out 
    1. any profanity (words like God, shit, fuck, etc) 
    2. personal information related to the professor,
    3. specific incidents, 
    4. controversial statements that show certain places or people in bad light, 
    
    and flag out everything you can find.
"""
    prompt2 = f"""
    Here is an example transcript:
    {example_transcript}
    and here are just of the flags found by a human:
    {example_flags}
    """

    prompt3 = f"""
    Here are some additional flags found by me which match the criteria of the prompt:
    {claude_flags}
    """

    prompt4 = f"""
    Now go through this entire transcript and get similar flags.

    Be very sensitive, flag anything if you think that meets the criteria above.

    For the transcript below, return the specific sentences in quotes, and the reason you flagged them:

    The main transcript:

    {transcript}.

    """

    prompt = f"{prompt1}\n\n{prompt2}\n\n{prompt3}\n\n{prompt4}"

    if llm_choice == "anthropic":
        return analyze_with_anthropic(prompt)
    elif llm_choice == "gemini":
        return analyze_with_gemini(prompt)
    elif llm_choice == "openai":
        return analyze_with_openai(prompt)
    else:
        raise ValueError("Invalid LLM choice. Choose 'anthropic', 'gemini', or 'openai'.")

def analyze_with_anthropic(prompt):
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=8192,
        system="You are an extremely sensitive assistant who is very good at finding swear words, personal information, specific incidents, sensitive information in text.",
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

def analyze_with_gemini(prompt):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

def analyze_with_openai(prompt):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an extremely sensitive assistant who is very good at finding swear words, personal information, specific incidents, sensitive information in text."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def write_analysis_to_file(video_path, transcript, analysis):
    # Create a filename based on the input video name
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{base_name}_analysis.txt"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"Analysis for video: {video_path}\n\n")
        f.write("\n\nContent Analysis:\n")
        f.write(analysis)
    
    return output_filename

def main(video_path, llm_choice):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The video file '{video_path}' does not exist.")
    
    # transcript = transcribe_video(video_path)
    transcript = read_file(r"C:\Users\hp\video_transcribe_analyze\1003_transcript.txt")
    analysis = analyze_content(transcript, llm_choice)
    output_file = write_analysis_to_file(video_path, transcript, analysis)

def cli():
    parser = argparse.ArgumentParser(description="Transcribe and analyze a video.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--llm", type=str, choices=["anthropic", "gemini", "openai"], default="anthropic", help="Choose the LLM to use for analysis")
    args = parser.parse_args()
    main(args.video_path, args.llm)

if __name__ == "__main__":
    cli()