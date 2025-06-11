import torch
import torch
from torch.nn import Module

import gc
import logging
import subprocess
import shutil


def get_gpu_memory_info_smi():
    """
    Retrieve GPU memory information using nvidia-smi.

    Returns:
        list of dict: A list containing memory info for each GPU,
                      or an empty list if nvidia-smi is unavailable.
    """
    # Check if nvidia-smi is available
    if not shutil.which("nvidia-smi"):
        print(
            "nvidia-smi not found. Ensure NVIDIA drivers are installed and accessible."
        )
        return []

    try:
        # Run nvidia-smi command to fetch memory info
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        # Parse the output
        memory_info = []
        for line in result.stdout.strip().split("\n"):
            total, used, free = map(int, line.split(","))
            memory_info.append({"total": total, "used": used, "free": free})
        return memory_info

    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e.stderr}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def gpu_memory_info_smi(prefix=""):
    # Example usage
    gpu_info = get_gpu_memory_info_smi()
    if gpu_info:
        for idx, info in enumerate(gpu_info):
            logging.info(
                f"[gpu_memory_info_smi]{prefix} GPU {idx}: Total Memory: {info['total']} MiB, Used: {info['used']} MiB, Free: {info['free']} MiB"
            )
    else:
        logging.info("[gpu_memory_info_smi] No GPU information available.")


def gpu_memory_info(device_id=0):
    """Display free and used GPU memory."""
    # Ensure the device is initialized
    if not torch.cuda.is_available():
        logging.info("No CUDA device avaiable!")
        return

    torch.cuda.init()

    # Retrieve memory stats for the specified device
    free_mem = torch.cuda.memory_reserved(
        device=device_id
    ) - torch.cuda.memory_allocated(device=device_id)
    used_mem = torch.cuda.memory_allocated(device=device_id)
    total_mem = torch.cuda.get_device_properties(device_id).total_memory

    logging.info(f"[gpu_memory_info] Device ID: {device_id}")
    logging.info(f"[gpu_memory_info] Total Memory: {total_mem / 1024**2:.2f} MB")
    logging.info(f"[gpu_memory_info] Used Memory: {used_mem / 1024**2:.2f} MB")
    logging.info(f"[gpu_memory_info] Free Memory: {free_mem / 1024**2:.2f} MB")


def check_all_objects_with_names(scope=globals()):
    logging.info("Inspecting Tensors in Scope:")
    for name, obj in scope.items():
        if torch.is_tensor(obj):
            print(f"Name: {name}, Size: {obj.size()}, Device: {obj.device}")

    logging.info("\nInspecting Modules:")
    for obj in gc.get_objects():
        if isinstance(obj, Module):
            print(f"Module: {type(obj).__name__}")
            for name, param in obj.named_parameters(recurse=True):
                print(f"  Parameter: {name}, Device: {param.device} Size:{obj.size()}")
            for buffer_name, buffer in obj.named_buffers():
                print(
                    f"  Buffer: {buffer_name}, Device: {buffer.device} Size:{obj.size()}"
                )
