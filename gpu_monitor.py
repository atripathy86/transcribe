import pynvml
import time

def print_gpu_stats():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        print(f"Found {device_count} GPU(s)")
        print("-" * 60)
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                name = pynvml.nvmlDeviceGetName(handle)
            except pynvml.NVMLError as e:
                name = f"Unknown ({e})"
            
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                temp_str = f"{temp}°C"
            except pynvml.NVMLError as e:
                temp_str = f"N/A ({e})"
            
            # Memory usage
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = mem_info.used / 1024**2  # Convert to MB
                mem_total = mem_info.total / 1024**2
                mem_str = f"{mem_used:.1f} MB / {mem_total:.1f} MB ({mem_used/mem_total*100:.1f}%)"
            except pynvml.NVMLError as e:
                mem_str = f"N/A ({e})"
            
            # Utilization (Load)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_load = f"{util.gpu}%"
                mem_load = f"{util.memory}%"
            except pynvml.NVMLError as e:
                gpu_load = f"N/A ({e})"
                mem_load = f"N/A ({e})"
            
            print(f"GPU {i}: {name}")
            print(f"  Temperature: {temp_str}")
            print(f"  Memory:      {mem_str}")
            print(f"  GPU Load:    {gpu_load}")
            print(f"  Memory Load: {mem_load}")
            print("-" * 60)
            
    except pynvml.NVMLError as error:
        print(f"Error: {error}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

if __name__ == "__main__":
    print_gpu_stats()
