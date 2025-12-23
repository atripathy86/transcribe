import os
import sys
import time
import argparse
from pathlib import Path
from faster_whisper import WhisperModel
from tqdm import tqdm

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

def save_txt(segments, filename):
    """Save plain text transcription"""
    with open(filename, 'w') as f:
        for segment in segments:
            f.write(segment.text.strip() + " ")
    print(f"Saved {filename}")

def save_tsv(segments, filename):
    """Save Tab-Separated Values with timestamps in milliseconds"""
    with open(filename, 'w') as f:
        f.write("start\tend\ttext\n")
        for segment in segments:
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            text = segment.text.strip()
            f.write(f"{start_ms}\t{end_ms}\t{text}\n")
    print(f"Saved {filename}")

def save_srt(segments, filename):
    """Save SubRip subtitle format"""
    with open(filename, 'w') as f:
        for i, segment in enumerate(segments, 1):
            start_time = ms_to_srt_time(int(segment.start * 1000))
            end_time = ms_to_srt_time(int(segment.end * 1000))
            text = segment.text.strip()
            f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
    print(f"Saved {filename}")

def save_vtt(segments, filename):
    """Save WebVTT subtitle format"""
    with open(filename, 'w') as f:
        f.write("WEBVTT\n\n")
        for segment in segments:
            start_time = ms_to_vtt_time(int(segment.start * 1000))
            end_time = ms_to_vtt_time(int(segment.end * 1000))
            text = segment.text.strip()
            f.write(f"{start_time} --> {end_time}\n{text}\n\n")
    print(f"Saved {filename}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe MP4 files using faster-whisper")
    parser.add_argument("input", help="Path to the input MP4 file")
    parser.add_argument("--model", default="large-v3", help="Whisper model to use (default: large-v3)")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--compute_type", default="float16", help="Compute type (float16, int8_float16, int8, etc.)")
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

    print(f"Loading model {args.model} on {args.device} ({args.compute_type})...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    
    print(f"Transcribing {input_path}...")
    start_time = time.time()
    
    segments, info = model.transcribe(str(input_path), beam_size=5)
    
    # Use tqdm for progress bar based on audio duration
    pbar = tqdm(total=info.duration, unit="s", desc="Transcription Progress")
    
    all_segments = []
    for segment in segments:
        all_segments.append(segment)
        pbar.update(segment.end - pbar.n)
    pbar.close()
    
    end_time = time.time()
    execution_time = end_time - start_time

    # Generate all output formats in the output directory
    save_txt(all_segments, output_dir / f"{base_name}.txt")
    save_tsv(all_segments, output_dir / f"{base_name}.tsv")
    save_srt(all_segments, output_dir / f"{base_name}.srt")
    save_vtt(all_segments, output_dir / f"{base_name}.vtt")

    print(f"\nTranscription complete! Files saved in {output_dir}")
    print(f"Audio duration: {info.duration:.2f} seconds")
    print(f"Transcription time: {execution_time:.2f} seconds")
    print(f"Speedup: {info.duration / execution_time:.2f}x")

if __name__ == "__main__":
    main()
