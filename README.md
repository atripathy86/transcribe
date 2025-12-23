# Audio & Video Transcription Tool

A collection of Python scripts that use OpenAI's Whisper and Faster-Whisper models to transcribe MP3 audio and MP4 video files into multiple subtitle and text formats.

## Overview

This tool transcribes media files and generates outputs in 4 different formats. It includes support for standard Whisper and the high-performance Faster-Whisper implementation.

## Performance & Timing

All scripts include performance metrics:
- **Transcription Time**: Total time taken to process the file.
- **Speedup**: (For Faster-Whisper) Ratio of audio duration to processing time.

## Progress Updates

- **`transcribe.py` & `transcribe_mp4.py`**: Standard Whisper library. No progress bar, but reports total time at the end.
- **`transcript_fw_mp4.py`**: Uses `faster-whisper` and `tqdm` for a real-time progress bar.

---

## DGX-Spark Installation (aarch64 + CUDA 13)

### The Issue
The official `ctranslate2` wheels (the engine behind `faster-whisper`) on PyPI for **aarch64** are built for CPU only. On NVIDIA DGX-Spark systems, which use the `aarch64` architecture and **CUDA 13**, installing via standard `pip` results in a `ValueError: This CTranslate2 package was not compiled with CUDA support`.

To enable GPU acceleration, we must use specialized binaries and link the Python wrapper against them.

### Clean Installation Steps

#### 1. System Dependencies
Install the required libraries for audio processing and linear algebra:
```bash
sudo apt-get update && sudo apt-get install -y libopenblas-dev ffmpeg
```

#### 2. CTranslate2 C++ Binaries
Download and install the pre-compiled CUDA 13 binaries for aarch64:
```bash
# Download specialized binaries
curl -L -o ctranslate2-dgxspark-aarch64-cuda13.tar.gz https://github.com/assix/ctranslate2-aarch64-cuda13-binaries/releases/download/v4.6.0-cuda13-aarch64/ctranslate2-dgxspark-aarch64-cuda13.tar.gz

# Extract to /opt
sudo tar -xzvf ctranslate2-dgxspark-aarch64-cuda13.tar.gz -C /opt

# Configure system linker
echo "/opt/ctranslate2/lib" | sudo tee /etc/ld.so.conf.d/ctranslate2.conf
sudo ldconfig
```

#### 3. Python Environment Setup
Use **Python 3.10.14** (matching the binary build environment):
```bash
# Create environment
uv venv --python 3.10.14
source .venv/bin/activate

# Install NVIDIA libraries from their index
uv pip install nvidia-cublas nvidia-cudnn-cu13 --extra-index-url https://pypi.nvidia.com

# Install other dependencies
uv pip install faster-whisper tqdm openai-whisper torch
```

#### 4. Link Python Wrapper to Binaries
You must reinstall the `ctranslate2` Python package from source, pointing it to the `/opt` installation. Use the version that matches the binaries (v4.6.0):
```bash
git clone --recursive https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2
git checkout v4.6.0
cd python
CTRANSLATE2_ROOT=/opt/ctranslate2 uv pip install .
cd ../..
rm -rf CTranslate2
```

#### 5. Verification
```python
python -c "import ctranslate2; print(f'CUDA device count: {ctranslate2.get_cuda_device_count()}')"
# Should output: CUDA device count: 1
```

---

## Usage

Ensure your virtual environment is activated before running the scripts:
```bash
source .venv/bin/activate
```

### Standard Whisper (MP3)
```bash
python transcribe.py path/to/audio.mp3
```

### Standard Whisper (MP4)
```bash
python transcribe_mp4.py path/to/video.mp4
```

### Faster-Whisper (MP4 with Progress Bar)
```bash
python transcript_fw_mp4.py path/to/video.mp4
```

## Output Formats
1. **`.txt`**: Plain text transcript.
2. **`.srt`**: SubRip subtitles (for video players).
3. **`.vtt`**: WebVTT subtitles (for web players).
4. **`.tsv`**: Tab-separated values with millisecond timestamps.

## Output Organization
- **MP3**: Files saved in the same directory as input.
- **MP4**: Files saved in a dedicated `<filename>_transcribed/` folder.

## Model Options
Available: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3` (default).
