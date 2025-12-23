	
import whisper
import json
import sys
import os
import time

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

# Load model and transcribe
model = whisper.load_model("large-v3", device="cuda")

# Get input file from command line argument or use default
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = "audio.mp3"

# Check if file exists
if not os.path.exists(input_file):
    print(f"Error: File '{input_file}' not found!")
    sys.exit(1)

# Get base filename without extension
base_filename = os.path.splitext(input_file)[0]

print(f"Transcribing {input_file}...")
start_time = time.time()
result = model.transcribe(input_file)
end_time = time.time()

execution_time = end_time - start_time

# Generate all output formats
save_txt(result, f"{base_filename}.txt")
save_tsv(result, f"{base_filename}.tsv")
save_srt(result, f"{base_filename}.srt")
save_vtt(result, f"{base_filename}.vtt")

print(f"\nTranscription complete!")
print(f"Time taken: {execution_time:.2f} seconds")
# print(result["text"])

