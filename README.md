# Audio & Video Transcription Tool

A collection of Python scripts that use OpenAI's Whisper and Faster-Whisper models to transcribe MP3 audio and MP4 video files into multiple subtitle and text formats.

## Overview

This tool transcribes media files and generates outputs in 4 different formats. It includes support for standard Whisper and the high-performance Faster-Whisper implementation.

## Key Features

- **High Performance**: Uses `faster-whisper` for significantly faster transcription compared to standard OpenAI Whisper.
- **Thermal Protection**: Monitors GPU temperature and utilization. Automatically pauses transcription if thresholds are exceeded (>55°C or >70% load) to protect hardware.
- **Incremental Writing**: Writes segments to output files (`.txt`, `.srt`, `.vtt`, `.tsv`) immediately as they are generated, preventing data loss.
- **Batch Processing**: Support for processing multiple files in sequence with configurable delays.
- **Comprehensive Logging**: Detailed execution logs for both individual transcriptions and batch processes.
- **Real-time Progress**: Interactive progress bars with efficiency metrics (Audio Duration / Processing Time).

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
uv pip install faster-whisper tqdm openai-whisper torch nvidia-ml-py
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
```bash
# Add NVIDIA libraries to the linker path (Required for DGX-Spark)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cu13/lib

python -c "import ctranslate2; print(f'CUDA device count: {ctranslate2.get_cuda_device_count()}')"
# Should output: CUDA device count: 1
```

---

## Usage

Ensure your virtual environment is activated before running the scripts:
```bash
source .venv/bin/activate
```

**Note for DGX-Spark:** You must export the library paths for the NVIDIA dependencies so `ctranslate2` can find them. You can run this command manually, or append it to your `bin/activate` script to make it automatic:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cu13/lib
```

### Optimized Transcription (Recommended)
This version includes thermal protection, incremental writing, and detailed logging.
```bash
python transcript_fw_mp4_opt.py path/to/video.mp4
```

### Batch Transcription
Runs the optimized script on all `.mp4` files in `../recordings-downloads/` that haven't been processed yet, with a 120s delay between files.
```bash
python batch_transcribe.py
```

### Other Scripts
- **Standard Whisper (MP3)**: `python transcribe.py path/to/audio.mp3`
- **Standard Whisper (MP4)**: `python transcribe_mp4.py path/to/video.mp4`
- **Basic Faster-Whisper**: `python transcript_fw_mp4.py path/to/video.mp4`

---

## Output Formats & Organization

### Output Formats
1. **`.txt`**: Plain text transcript.
2. **`.srt`**: SubRip subtitles (for video players).
3. **`.vtt`**: WebVTT subtitles (for web players).
4. **`.tsv`**: Tab-separated values with millisecond timestamps.
5. **`.log`**: Detailed execution logs (Optimized and Batch versions).

### Organization
- **MP3**: Files saved in the same directory as input.
- **MP4**: Files saved in a dedicated `<filename>_transcribed/` folder.

### Model Options
Available: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3` (default).

---

## Progress Updates & Interpretation

### Progress Bar Metrics
When running `transcript_fw_mp4_opt.py`, you will see a real-time progress bar:
`Transcription Progress:  45%|████▌     | 797.86/1772.37 [02:15<02:45, 5.91s/s, eff=5.91x]`

- **Percentage & Bar**: Visual representation of the audio duration processed.
- **`797.86/1772.37`**: Seconds of audio processed vs. total audio duration.
- **`[02:15<02:45]`**: Elapsed time vs. estimated remaining time.
- **`5.91s/s`**: Processing rate (audio seconds processed per real-world second).
- **`eff=5.91x`**: **Efficiency Metric**. This represents the ratio of `Audio Duration / Processing Time`. 
    - An efficiency of **1.0x** means transcription is happening in real-time.
    - An efficiency of **5.0x** means 1 hour of audio is transcribed in 12 minutes.
    - Higher values indicate better performance.

### Performance Metrics
At the end of each run, the script reports:
- **Transcription Time**: Total time taken to process the file.
- **Speedup**: Ratio of audio duration to processing time.

---

## GPU Monitoring (NVIDIA DGX-Spark)

To monitor GPU temperature, load, and memory usage on the DGX-Spark, we use the official NVIDIA Management Library (NVML) Python bindings.

### Monitoring Script
A monitoring script `gpu_monitor.py` is included to check the status of the GPUs:
```bash
python gpu_monitor.py
```

### Why `nvidia-ml-py`?
- **Official Support**: Official Python binding for NVML.
- **Direct Access**: Communicates directly with the driver for high reliability.
- **Comprehensive**: Access to temperature, utilization, and memory metrics.
