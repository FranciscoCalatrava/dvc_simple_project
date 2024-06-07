# gpu_utils.py
import pynvml
import os

def get_free_gpu(min_memory_required):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    free_memory = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if mem_info.free > min_memory_required:
            free_memory.append((i, mem_info.free))

    # Sort GPUs by free memory in descending order
    free_memory.sort(key=lambda x: x[1], reverse=True)
    
    # Return the GPU index with sufficient memory
    return free_memory[0][0] if free_memory else None

def set_gpu_device(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
