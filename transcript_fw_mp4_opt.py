import os
import sys
import time
import argparse
import logging
from pathlib import Path
from faster_whisper import WhisperModel
from tqdm import tqdm
import pynvml

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

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

def check_gpu_and_pause(temp_threshold=55, load_threshold=70, pause_time=60):
    """Check GPU status and pause if thresholds are exceeded"""
    logger = logging.getLogger(__name__)
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assuming GPU 0
        
        while True:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            if temp > temp_threshold and util.gpu > load_threshold:
                logger.warning(f"[GPU ALERT] Temp: {temp}°C, Load: {util.gpu}%. Pausing for {pause_time}s to cool down...")
                time.sleep(pause_time)
            else:
                break
    except pynvml.NVMLError as e:
        # If NVML fails, we just continue to avoid breaking the transcription
        pass
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Transcribe MP4 files using faster-whisper with GPU thermal protection")
    parser.add_argument("input", help="Path to the input MP4 file")
    parser.add_argument("--model", default="large-v3", help="Whisper model to use (default: large-v3)")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--compute_type", default="float16", help="Compute type (float16, int8_float16, int8, etc.)")
    parser.add_argument("--temp_limit", type=int, default=55, help="Temperature threshold to pause (default: 55)")
    parser.add_argument("--load_limit", type=int, default=70, help="Utilization threshold to pause (default: 70)")
    parser.add_argument("--pause_time", type=int, default=90, help="Seconds to pause when thresholds exceeded (default: 90)")
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
    log_file = output_dir / f"{base_name}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            TqdmLoggingHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading model {args.model} on {args.device} ({args.compute_type})...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    
    logger.info(f"Transcribing {input_path}...")
    start_time = time.time()
    
    segments, info = model.transcribe(str(input_path), beam_size=5)
    
    logger.info(f"Audio duration: {info.duration:.2f} seconds ({info.duration/60:.2f} minutes)")

    # Use tqdm for progress bar based on audio duration
    # dynamic_ncols=True ensures the bar refreshes on the same line by adapting to terminal width
    pbar = tqdm(
        total=round(info.duration, 2), 
        unit="s", 
        desc="Transcription Progress", 
        bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
        dynamic_ncols=True,
        leave=True
    )
    
    # Open files for incremental writing
    txt_path = output_dir / f"{base_name}.txt"
    tsv_path = output_dir / f"{base_name}.tsv"
    srt_path = output_dir / f"{base_name}.srt"
    vtt_path = output_dir / f"{base_name}.vtt"

    with open(txt_path, 'w', encoding='utf-8') as txt_f, \
         open(tsv_path, 'w', encoding='utf-8') as tsv_f, \
         open(srt_path, 'w', encoding='utf-8') as srt_f, \
         open(vtt_path, 'w', encoding='utf-8') as vtt_f:
        
        tsv_f.write("start\tend\ttext\n")
        vtt_f.write("WEBVTT\n\n")
        
        for i, segment in enumerate(segments, 1):
            # Thermal protection check before processing each segment
            if args.device == "cuda":
                check_gpu_and_pause(
                    temp_threshold=args.temp_limit, 
                    load_threshold=args.load_limit,
                    pause_time=args.pause_time
                )
            
            text = segment.text.strip()
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)

            # Write to TXT
            txt_f.write(text + " ")
            txt_f.flush()

            # Write to TSV
            tsv_f.write(f"{start_ms}\t{end_ms}\t{text}\n")
            tsv_f.flush()

            # Write to SRT
            start_srt = ms_to_srt_time(start_ms)
            end_srt = ms_to_srt_time(end_ms)
            srt_f.write(f"{i}\n{start_srt} --> {end_srt}\n{text}\n\n")
            srt_f.flush()

            # Write to VTT
            start_vtt = ms_to_vtt_time(start_ms)
            end_vtt = ms_to_vtt_time(end_ms)
            vtt_f.write(f"{start_vtt} --> {end_vtt}\n{text}\n\n")
            vtt_f.flush()

            # Log segment to file and terminal (DEBUG level)
            logger.debug(f"[{ms_to_srt_time(start_ms)} -> {ms_to_srt_time(end_ms)}] {text}")

            # Update progress and efficiency
            # Efficiency is calculated as (processed audio duration) / (actual time elapsed)
            current_elapsed = time.time() - start_time
            efficiency = segment.end / current_elapsed if current_elapsed > 0 else 0
            pbar.set_postfix(eff=f"{efficiency:.2f}x")
            pbar.update(round(segment.end, 2) - pbar.n)
    
    pbar.close()
    
    end_time = time.time()
    elapsed = end_time - start_time
    speedup = info.duration / elapsed
    
    logger.info(f"\nTranscription completed in {elapsed:.2f} seconds.")
    logger.info(f"Audio duration: {info.duration:.2f} seconds.")
    logger.info(f"Speedup: {speedup:.2f}x")
    logger.info(f"Outputs saved in: {output_dir}")

if __name__ == "__main__":
    main()
