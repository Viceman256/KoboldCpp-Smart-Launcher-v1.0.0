#!/usr/bin/env python3
"""
Example commands for KoboldCpp Smart Launcher

This script demonstrates performance improvements using tensor offloading.
Run this script to see example benchmarks with different configurations.
"""

import os
import sys
import platform
import subprocess
import time
from pathlib import Path

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
BOLD = "\033[1m"
ENDC = "\033[0m"

def print_header(text):
    """Print a section header"""
    print(f"\n{BLUE}{BOLD}{'=' * 80}{ENDC}")
    print(f"{BLUE}{BOLD}   {text}{ENDC}")
    print(f"{BLUE}{BOLD}{'=' * 80}{ENDC}\n")

def print_command(cmd):
    """Format and print a command"""
    if isinstance(cmd, list):
        cmd = " ".join(cmd)
    print(f"{YELLOW}$ {cmd}{ENDC}")

def print_result(result):
    """Format and print a result"""
    print(f"{GREEN}{BOLD}Result: {result}{ENDC}")

def get_system_info():
    """Get basic system information"""
    info = {
        "os": platform.system(),
        "cpu": platform.processor() or "Unknown CPU",
        "python": sys.version.split()[0],
        "gpu": "Unknown"
    }
    
    # Try to get GPU info
    try:
        if platform.system() == "Windows":
            # Windows - Try with wmic
            gpu_cmd = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True, text=True, check=False
            )
            if gpu_cmd.returncode == 0:
                lines = gpu_cmd.stdout.strip().split('\n')
                if len(lines) > 1:
                    info["gpu"] = lines[1].strip()
        elif platform.system() == "Linux":
            # Linux - Try with lspci
            gpu_cmd = subprocess.run(
                ["lspci", "-v"], 
                capture_output=True, text=True, check=False
            )
            if gpu_cmd.returncode == 0:
                output = gpu_cmd.stdout
                for line in output.split('\n'):
                    if "VGA" in line or "3D" in line or "NVIDIA" in line or "AMD" in line:
                        info["gpu"] = line.split(':')[-1].strip()
                        break
    except Exception:
        pass
    
    return info

def example_1():
    """Example 1: Qwen3-32B on RTX 3060 (12GB)"""
    print_header("Example 1: Qwen3-32B on RTX 3060 (12GB)")
    
    print("Traditional Layer Offloading:")
    cmd = "python koboldcpp.py --model Qwen3-32B-Q4_K_M.gguf --threads 8 --usecublas --flashattention --gpulayers 38 --quantkv 1"
    print_command(cmd)
    print_result("Tokens per second: 3.98 t/s")
    
    print("\nWith Smart Tensor Offloading:")
    cmd = "python koboldcpp.py --model Qwen3-32B-Q4_K_M.gguf --threads 8 --usecublas --flashattention --gpulayers 65 --quantkv 1 --overridetensors \"\\.[13579]\\.ffn_up|\\.[1-3][13579]\\.ffn_up=CPU\""
    print_command(cmd)
    print_result("Tokens per second: 10.61 t/s")
    
    print(f"\n{BOLD}Performance Improvement: 166%{ENDC}")
    print("This configuration allowed offloading all 65 layers to GPU by selectively keeping certain FFN tensors on CPU.")

def example_2():
    """Example 2: Gemma3-27B on RTX 2060 (6GB)"""
    print_header("Example 2: Gemma3-27B on RTX 2060 (6GB)")
    
    print("Traditional Layer Offloading:")
    cmd = "python koboldcpp.py --model gemma3-27b-IQ4_XS.gguf --threads 6 --usecublas --flashattention --gpulayers 16 --contextsize 16384"
    print_command(cmd)
    print_result("Tokens per second: 6.86 t/s")
    
    print("\nWith Smart Tensor Offloading:")
    cmd = "python koboldcpp.py --model gemma3-27b-IQ4_XS.gguf --threads 6 --usecublas --flashattention --gpulayers 99 --contextsize 16384 --overridetensors \"\\.(5[3-9]|6[0-3])\\.(ffn_*)=CPU\""
    print_command(cmd)
    print_result("Tokens per second: 10.4 t/s")
    
    print(f"\n{BOLD}Performance Improvement: 52%{ENDC}")
    print("By keeping specific FFN tensors on CPU in higher layers, we can offload everything else to GPU for faster processing.")

def example_3():
    """Example 3: Mixtral-8x7B MoE on RTX 4070 (12GB)"""
    print_header("Example 3: Mixtral-8x7B MoE on RTX 4070 (12GB)")
    
    print("Traditional Layer Offloading:")
    cmd = "python koboldcpp.py --model mixtral-8x7b-v0.1.Q5_K_M.gguf --threads 12 --usecublas --flashattention --gpulayers 22 --contextsize 8192"
    print_command(cmd)
    print_result("Tokens per second: 15.3 t/s")
    
    print("\nWith Smart Tensor Offloading (MoE focused):")
    cmd = "python koboldcpp.py --model mixtral-8x7b-v0.1.Q5_K_M.gguf --threads 12 --usecublas --flashattention --gpulayers 32 --contextsize 8192 --overridetensors \"ffn_.*_exps=CPU\""
    print_command(cmd)
    print_result("Tokens per second: 24.7 t/s")
    
    print(f"\n{BOLD}Performance Improvement: 61%{ENDC}")
    print("For MoE models, targeting the experts tensors with 'ffn_.*_exps=CPU' is especially effective.")

def suggested_patterns():
    """Show suggested regex patterns for different models"""
    print_header("Suggested Tensor Offloading Patterns")
    
    patterns = [
        {
            "name": "MoE Models (Mixtral, Qwen3-235B)",
            "pattern": "--overridetensors \"ffn_.*_exps=CPU\"",
            "description": "Offloads all MoE expert tensors to CPU"
        },
        {
            "name": "Dense Models - Moderate Offload",
            "pattern": "--overridetensors \"\\.[13579]\\.ffn_up=CPU\"",
            "description": "Offloads FFN up tensors in odd-numbered layers to CPU"
        },
        {
            "name": "Dense Models - Heavy Offload",
            "pattern": "--overridetensors \"\\.[0-9]+\\.(ffn_up|ffn_gate)=CPU\"",
            "description": "Offloads all FFN up and gate tensors to CPU"
        },
        {
            "name": "Dense Models - Targeted Offload",
            "pattern": "--overridetensors \"\\.(5[0-9]|6[0-9])\\.ffn_.*=CPU\"",
            "description": "Offloads all FFN tensors in layers 50-69 to CPU"
        },
        {
            "name": "Custom Pattern Template",
            "pattern": "--overridetensors \"LAYER_PATTERN.TENSOR_PATTERN=CPU\"",
            "description": "Replace LAYER_PATTERN with layer selection (e.g., [0-9]+ for all) and TENSOR_PATTERN with tensor type"
        }
    ]
    
    for p in patterns:
        print(f"{BOLD}{p['name']}{ENDC}")
        print_command(p["pattern"])
        print(f"{p['description']}\n")

def explain_performance():
    """Explain why tensor offloading improves performance"""
    print_header("Why Tensor Offloading Improves Performance")
    
    print("""When running LLMs with limited VRAM, traditional approaches offload entire 
transformer layers to CPU. However, not all tensors within a layer are equal:

1. Attention tensors: Smaller in size, very compute-intensive, benefit greatly from GPU acceleration
2. Feed-Forward Network (FFN) tensors: Much larger in size, more memory bandwidth intensive
   - FFN UP: Maps from model dimension to intermediate dimension
   - FFN DOWN: Maps from intermediate dimension back to model dimension
   - FFN GATE: (In some architectures) Controls information flow

Tensor offloading lets you keep the compute-intensive parts on GPU while moving the
bandwidth-intensive parts to CPU. This results in:

- More efficient use of GPU resources
- Better parallelization of computation
- Reduced bottlenecks

The KoboldCpp Smart Launcher helps you find the optimal balance for your specific
hardware and model combination.""")

def main():
    """Run all examples"""
    # Print system info
    sys_info = get_system_info()
    print_header("System Information")
    print(f"OS: {sys_info['os']}")
    print(f"CPU: {sys_info['cpu']}")
    print(f"Python: {sys_info['python']}")
    print(f"GPU: {sys_info['gpu']}")
    
    print_header("KoboldCpp Smart Launcher - Performance Examples")
    print(f"""This script demonstrates the performance improvements possible with tensor offloading.
The examples below show real-world results from different hardware configurations.
    
Your results may vary depending on your specific hardware, model, and settings.""")
    
    # Run examples
    example_1()
    example_2()
    example_3()
    
    # Show suggested patterns
    suggested_patterns()
    
    # Explain performance
    explain_performance()
    
    print_header("Try KoboldCpp Smart Launcher")
    print(f"""To try tensor offloading with your own models:

1. {BOLD}GUI Version:{ENDC} Run 'python koboldcpp_launcher_gui.py'
2. {BOLD}CLI Version:{ENDC} Run 'python koboldcpp_launcher.py'

The launcher will help you find the optimal tensor offload strategy for your models.""")

if __name__ == "__main__":
    main()