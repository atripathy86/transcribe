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

### Faster-Whisper with Thermal Protection & Logging (Optimized)
This version monitors GPU temperature and utilization. If the GPU gets too hot (>55°C) while under heavy load (>70%), it automatically pauses transcription to allow the hardware to cool down.

**Key Features:**
- **Thermal Protection**: Pauses transcription if GPU thresholds are exceeded.
- **Incremental Writing**: Writes segments to output files immediately as they are generated, preventing data loss if the process is interrupted.
- **Comprehensive Logging**: Saves all transcription logs, including timestamps and segments, to a `.log` file in the output directory.
- **Real-time Feedback**: Logs segments to the terminal while maintaining a clean progress bar.

```bash
python transcript_fw_mp4_opt.py path/to/video.mp4
```

## Output Formats
1. **`.txt`**: Plain text transcript.
2. **`.srt`**: SubRip subtitles (for video players).
3. **`.vtt`**: WebVTT subtitles (for web players).
4. **`.tsv`**: Tab-separated values with millisecond timestamps.
5. **`.log`**: Detailed transcription logs (Optimized version only).

## Output Organization
- **MP3**: Files saved in the same directory as input.
- **MP4**: Files saved in a dedicated `<filename>_transcribed/` folder.

## Model Options
Available: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3` (default).

---

## GPU Monitoring (NVIDIA DGX-Spark)

To monitor GPU temperature, load, and memory usage on the DGX-Spark, we use the official NVIDIA Management Library (NVML) Python bindings.

### Why `nvidia-ml-py`?
- **Official Support**: It is the official Python binding for NVML, ensuring compatibility with NVIDIA drivers.
- **Direct Access**: Unlike parsing `nvidia-smi` output, it communicates directly with the driver for lower overhead and higher reliability.
- **Comprehensive**: Provides access to all hardware metrics including temperature, utilization, and memory.

### Installation
Install the package in your virtual environment:
```bash
uv pip install nvidia-ml-py
```

### Monitoring Script
A monitoring script `gpu_monitor.py` is included to check the status of the GPUs:

```python
import pynvml

def print_gpu_stats():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        print(f"Found {device_count} GPU(s)")
        print("-" * 60)
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get Device Name
            try:
                name = pynvml.nvmlDeviceGetName(handle)
            except pynvml.NVMLError:
                name = "Unknown"
            
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                temp_str = f"{temp}°C"
            except pynvml.NVMLError:
                temp_str = "N/A"
            
            # Memory usage
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = mem_info.used / 1024**2  # MB
                mem_total = mem_info.total / 1024**2
                mem_str = f"{mem_used:.1f} MB / {mem_total:.1f} MB"
            except pynvml.NVMLError:
                mem_str = "Not Supported"
            
            # Utilization (Load)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_load = f"{util.gpu}%"
            except pynvml.NVMLError:
                gpu_load = "N/A"
            
            print(f"GPU {i}: {name}")
            print(f"  Temperature: {temp_str}")
            print(f"  Memory:      {mem_str}")
            print(f"  GPU Load:    {gpu_load}")
            print("-" * 60)
            
    except pynvml.NVMLError as error:
        print(f"Driver Error: {error}")
    finally:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    print_gpu_stats()
```

### Usage
```bash
python gpu_monitor.py
```
