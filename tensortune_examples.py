#!/usr/bin/env python3
"""
Example commands for TensorTune

This script demonstrates performance improvements possible using smart tensor offloading
with KoboldCpp. TensorTune helps you discover and apply such strategies.
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
    print(f"{YELLOW}  $ {cmd}{ENDC}")

def print_result(result):
    """Format and print a result"""
    print(f"{GREEN}{BOLD}  Result: {result}{ENDC}")

def get_system_info():
    """Get basic system information"""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu": platform.processor() or "Unknown CPU",
        "python": sys.version.split()[0],
        "gpu": "Unknown (GPU detection in this script is basic)"
    }

    # Basic GPU detection (can be expanded or use TensorTune's core for more accuracy if integrated)
    try:
        if platform.system() == "Windows":
            gpu_cmd = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True, text=True, check=False, timeout=5
            )
            if gpu_cmd.returncode == 0:
                lines = gpu_cmd.stdout.strip().split('\n')
                if len(lines) > 1 and lines[1].strip():
                    info["gpu"] = lines[1].strip()
        elif platform.system() == "Linux":
            if shutil.which("lspci"):
                gpu_cmd = subprocess.run(
                    ["lspci"], capture_output=True, text=True, check=False, timeout=5
                )
                if gpu_cmd.returncode == 0:
                    output = gpu_cmd.stdout
                    for line in output.split('\n'):
                        if "VGA compatible controller" in line or "3D controller" in line:
                            info["gpu"] = line.split(':')[-1].strip().split('(')[0].strip()
                            break
            elif shutil.which("rocm-smi"): # Basic check if ROCm is present
                 info["gpu"] = "AMD GPU (rocm-smi likely available)"
        elif platform.system() == "Darwin": # macOS
            try:
                # This is a very basic way; Metal API access would be better but complex for a simple script
                sp_gpu_cmd = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True, text=True, check=False, timeout=5
                )
                if sp_gpu_cmd.returncode == 0:
                    for line in sp_gpu_cmd.stdout.split('\n'):
                        if "Chipset Model:" in line:
                            info["gpu"] = line.split("Chipset Model:")[1].strip()
                            break
            except Exception:
                pass # Silently fail if system_profiler isn't helpful
    except Exception as e:
        print(f"{YELLOW}  Note: Basic GPU detection in examples script failed: {e}{ENDC}")
        pass # Silently fail if any part of GPU detection fails
    
    return info

def example_1():
    """Example 1: Qwen3-32B on RTX 3060 (12GB)"""
    print_header("Example 1: Qwen3-32B on RTX 3060 (12GB)")
    
    print("  Traditional Layer Offloading:")
    cmd = "python koboldcpp.py --model Qwen3-32B-Q4_K_M.gguf --threads 8 --usecublas --flashattention --gpulayers 38 --quantkv 1"
    print_command(cmd)
    print_result("Tokens per second: ~3.98 t/s")
    
    print("\n  With Smart Tensor Offloading (TensorTune can help find this):")
    cmd = "python koboldcpp.py --model Qwen3-32B-Q4_K_M.gguf --threads 8 --usecublas --flashattention --gpulayers 65 --quantkv 1 --overridetensors \"\\.[13579]\\.ffn_up|\\.[1-3][13579]\\.ffn_up=CPU\""
    print_command(cmd)
    print_result("Tokens per second: ~10.61 t/s")
    
    print(f"\n  {BOLD}Performance Improvement: ~166%{ENDC}")
    print("  This configuration allowed offloading all 65 layers to GPU by selectively keeping certain FFN tensors on CPU.")

def example_2():
    """Example 2: Gemma3-27B on RTX 2060 (6GB)"""
    print_header("Example 2: Gemma3-27B on RTX 2060 (6GB)")
    
    print("  Traditional Layer Offloading:")
    cmd = "python koboldcpp.py --model gemma3-27b-IQ4_XS.gguf --threads 6 --usecublas --flashattention --gpulayers 16 --contextsize 16384"
    print_command(cmd)
    print_result("Tokens per second: ~6.86 t/s")
    
    print("\n  With Smart Tensor Offloading (TensorTune can help find this):")
    cmd = "python koboldcpp.py --model gemma3-27b-IQ4_XS.gguf --threads 6 --usecublas --flashattention --gpulayers 99 --contextsize 16384 --overridetensors \"\\.(5[3-9]|6[0-3])\\.(ffn_*)=CPU\""
    print_command(cmd)
    print_result("Tokens per second: ~10.4 t/s")
    
    print(f"\n  {BOLD}Performance Improvement: ~52%{ENDC}")
    print("  By keeping specific FFN tensors on CPU in higher layers, everything else can be offloaded to GPU for faster processing.")

def example_3():
    """Example 3: Mixtral-8x7B MoE on RTX 4070 (12GB)"""
    print_header("Example 3: Mixtral-8x7B MoE on RTX 4070 (12GB)")
    
    print("  Traditional Layer Offloading:")
    cmd = "python koboldcpp.py --model mixtral-8x7b-v0.1.Q5_K_M.gguf --threads 12 --usecublas --flashattention --gpulayers 22 --contextsize 8192"
    print_command(cmd)
    print_result("Tokens per second: ~15.3 t/s")
    
    print("\n  With Smart Tensor Offloading (MoE focused, TensorTune can help find this):")
    cmd = "python koboldcpp.py --model mixtral-8x7b-v0.1.Q5_K_M.gguf --threads 12 --usecublas --flashattention --gpulayers 32 --contextsize 8192 --overridetensors \"ffn_.*_exps=CPU\""
    print_command(cmd)
    print_result("Tokens per second: ~24.7 t/s")
    
    print(f"\n  {BOLD}Performance Improvement: ~61%{ENDC}")
    print("  For MoE models, targeting the experts' tensors with 'ffn_.*_exps=CPU' is often effective.")

def suggested_patterns():
    """Show suggested regex patterns for different models"""
    print_header("Suggested Tensor Offloading Patterns (Examples for Manual Use)")
    
    patterns = [
        {
            "name": "MoE Models (e.g., Mixtral, some Qwen variants)",
            "pattern": "--overridetensors \"ffn_.*_exps\\.weight=CPU\"",
            "description": "Offloads all MoE expert FFN weights to CPU. Adjust pattern if your MoE model uses a different naming scheme for expert tensors."
        },
        {
            "name": "Dense Models - Moderate CPU Offload",
            "pattern": "--overridetensors \"blk\\.\\d*[13579]\\.ffn_up\\.weight=CPU\"",
            "description": "Offloads FFN up-projection weights in odd-numbered layers to CPU."
        },
        {
            "name": "Dense Models - Aggressive FFN CPU Offload",
            "pattern": "--overridetensors \"blk\\.\\d+\\.(ffn_up|ffn_gate)\\.weight=CPU\"",
            "description": "Offloads all FFN up-projection and gate weights to CPU."
        },
        {
            "name": "Dense Models - Targeted High-Layer FFN Offload",
            "pattern": "--overridetensors \"blk\\.(2[4-9]|[3-9]\\d+)\\.ffn_.*\\.weight=CPU\"",
            "description": "Example: Offloads all FFN weights in layers 24 and above to CPU."
        },
        {
            "name": "Custom Pattern Template",
            "pattern": "--overridetensors \"LAYER_REGEX\\.TENSOR_REGEX=CPU\"",
            "description": "Replace LAYER_REGEX (e.g., blk\\.\\d+) and TENSOR_REGEX (e.g., ffn_down\\.weight) with appropriate regex."
        }
    ]
    
    for p in patterns:
        print(f"  {BOLD}{p['name']}{ENDC}")
        print_command(f"  {p['pattern']}") # Indent the command slightly
        print(f"  {p['description']}\n")
    
    print(f"{YELLOW}  Note:{ENDC} The exact regex patterns can vary based on model architecture and naming conventions within the GGUF file. TensorTune's auto-tuning aims to discover effective patterns for you.")

def explain_performance():
    """Explain why tensor offloading improves performance"""
    print_header("Why Smart Tensor Offloading Improves Performance")
    
    print("""  When running Large Language Models (LLMs) with limited VRAM, the traditional approach
  is to offload entire transformer layers to the CPU once VRAM is full. However,
  not all parts (tensors) within a layer have the same impact on performance or VRAM usage:

  1.  ${BOLD}Attention Tensors:${ENDC} These are generally smaller in size but are very compute-intensive.
      They benefit significantly from GPU acceleration.
  2.  ${BOLD}Feed-Forward Network (FFN) Tensors:${ENDC} These are typically much larger and can be more
      memory bandwidth-intensive than compute-bound on some hardware.
      - FFN Up-Projection: Often expands the model's hidden dimension to an intermediate dimension.
      - FFN Down-Projection: Maps from the intermediate dimension back to the model dimension.
      - FFN Gate: (In architectures like GLU variants) Controls information flow within the FFN.

  ${BOLD}Smart Tensor Offloading${ENDC}, which TensorTune helps you manage, allows for a more granular
  approach. You can selectively keep the most compute-heavy and VRAM-light parts (like
  attention mechanisms) on the GPU, while offloading only the largest or most
  bandwidth-sensitive FFN tensors (or parts of them) to the CPU.

  This results in:
  - ${GREEN}More Layers on GPU:${ENDC} Often, more total layers can fit into VRAM because the bulkiest
    parts of some layers are on the CPU.
  - ${GREEN}Efficient VRAM Use:${ENDC} Prioritizes GPU VRAM for tensors that benefit most from its speed.
  - ${GREEN}Better Parallelization:${ENDC} CPU can work on offloaded FFN tensors while GPU handles attention.
  - ${GREEN}Reduced Bottlenecks:${ENDC} Can alleviate VRAM capacity or bandwidth limitations.

  TensorTune's auto-tuning feature explores different offload strategies (represented
  as "OT Levels" which translate to specific `--gpulayers` and `--overridetensors`
  patterns) to find a sweet spot for your specific hardware and model combination,
  aiming for the best possible performance within your VRAM constraints.""")

def main():
    """Run all examples"""
    # Print system info
    sys_info = get_system_info()
    print_header("System Information (Basic)")
    print(f"  OS: {sys_info['os']}")
    print(f"  Architecture: {sys_info['arch']}")
    print(f"  CPU: {sys_info['cpu']}")
    print(f"  Python: {sys_info['python']}")
    print(f"  GPU (basic detection): {sys_info['gpu']}")
    print(f"{YELLOW}  Note: For accurate GPU VRAM info, please use TensorTune itself.{ENDC}")
    
    print_header("TensorTune - Performance Examples with KoboldCpp")
    print(f"""  This script demonstrates the performance improvements possible with smart tensor
  offloading, a technique that TensorTune helps you automate and manage.
  The examples below show real-world results from different hardware setups using
  KoboldCpp directly with manually crafted --overridetensors flags.
    
  Your results may vary. TensorTune aims to find similar optimal configurations for you.
  All token-per-second (t/s) figures are approximate and for illustration.""")
    
    example_1()
    example_2()
    example_3()
    
    suggested_patterns()
    explain_performance()
    
    print_header("Try TensorTune v1.1.0")
    print(f"""  To use TensorTune for your own models:

  1. {BOLD}Ensure TensorTune is installed:{ENDC}
     If you haven't already, run: {YELLOW}python install.py{ENDC}

  2. {BOLD}Launch TensorTune:{ENDC}
     - {BOLD}GUI Version:{ENDC} {YELLOW}python tensortune_gui.py{ENDC} (or {GREEN}./launch_gui.sh{ENDC} on Linux/macOS if created by installer)
     - {BOLD}CLI Version:{ENDC} {YELLOW}python tensortune_cli.py{ENDC} (or {GREEN}./launch_cli.sh{ENDC} on Linux/macOS if created by installer)

  TensorTune's "Auto-Tune" feature will guide you through finding an optimal
  tensor offload strategy by adjusting GPU layers and --overridetensors patterns.""")

if __name__ == "__main__":
    # A simple check for shutil for lspci existence check
    try:
        import shutil
    except ImportError:
        print(f"{RED}Error: The 'shutil' module (standard library) could not be imported. This script might have issues.{ENDC}")
        # Potentially exit or degrade gracefully if shutil is critical for the script's core logic,
        # but for just `shutil.which` in a helper, a warning might suffice.
        # For this example script, we'll let it proceed with a warning.
    main()