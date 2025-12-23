import whisper
import json
import sys
import os
import subprocess
import time
from pathlib import Path
import argparse

def ms_to_srt_time(milliseconds):
    """Convert milliseconds to SRT format (HH:MM:SS,mmm)"""
    seconds = milliseconds / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(milliseconds % 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def ms_to_vtt_time(milliseconds):
    """Convert milliseconds to VTT format (HH:MM.mmm)"""
    seconds = milliseconds / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(milliseconds % 1000)
    return f"{hours:02d}:{minutes:02d}.{millis:03d}" if hours > 0 else f"{minutes:02d}:{secs:02d}.{millis:03d}"

def save_txt(result, filename):
    """Save plain text transcription"""
    with open(filename, 'w') as f:
        f.write(result["text"])
    print(f"Saved {filename}")

def save_tsv(result, filename):
    """Save Tab-Separated Values with timestamps in milliseconds"""
    with open(filename, 'w') as f:
        f.write("start\tend\ttext\n")
        for segment in result["segments"]:
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            text = segment["text"].strip()
            f.write(f"{start_ms}\t{end_ms}\t{text}\n")
    print(f"Saved {filename}")

def save_srt(result, filename):
    """Save SubRip subtitle format"""
    with open(filename, 'w') as f:
        for i, segment in enumerate(result["segments"], 1):
            start_time = ms_to_srt_time(int(segment["start"] * 1000))
            end_time = ms_to_srt_time(int(segment["end"] * 1000))
            text = segment["text"].strip()
            f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
    print(f"Saved {filename}")

def save_vtt(result, filename):
    """Save WebVTT subtitle format"""
    with open(filename, 'w') as f:
        f.write("WEBVTT\n\n")
        for segment in result["segments"]:
            start_time = ms_to_vtt_time(int(segment["start"] * 1000))
            end_time = ms_to_vtt_time(int(segment["end"] * 1000))
            text = segment["text"].strip()
            f.write(f"{start_time} --> {end_time}\n{text}\n\n")
    print(f"Saved {filename}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe MP4 files using Whisper")
    parser.add_argument("input", help="Path to the input MP4 file")
    parser.add_argument("--model", default="large-v3", help="Whisper model to use (default: large-v3)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File '{input_path}' not found!")
        sys.exit(1)

    # Create output directory: input_transcribed
    output_dir = input_path.parent / f"{input_path.stem}_transcribed"
    output_dir.mkdir(exist_ok=True)

    # Base filename for output files inside the directory
    base_name = input_path.stem

    # Previous code to extract audio to disk (commented out)
    # audio_wav = output_dir / f"{base_name}.wav"
    # print(f"Extracting audio to {audio_wav}...")
    # try:
    #     subprocess.run(
    #         [
    #             "ffmpeg",
    #             "-y",
    #             "-i", str(input_path),
    #             "-vn",
    #             "-ac", "1",
    #             "-ar", "16000",
    #             "-c:a", "pcm_s16le",
    #             str(audio_wav),
    #         ],
    #         check=True,
    #         capture_output=True
    #     )
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during audio extraction: {e.stderr.decode()}")
    #     sys.exit(1)

    print(f"Loading model {args.model}...")
    model = whisper.load_model(args.model, device="cuda")
    
    print(f"Transcribing {input_path}...")
    start_time = time.time()
    # result = model.transcribe(str(audio_wav)) # Previous way using extracted wav
    result = model.transcribe(str(input_path))
    end_time = time.time()

    execution_time = end_time - start_time

    # Generate all output formats in the output directory
    save_txt(result, output_dir / f"{base_name}.txt")
    save_tsv(result, output_dir / f"{base_name}.tsv")
    save_srt(result, output_dir / f"{base_name}.srt")
    save_vtt(result, output_dir / f"{base_name}.vtt")

    print(f"\nTranscription complete! Files saved in {output_dir}")
    print(f"Time taken: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()


