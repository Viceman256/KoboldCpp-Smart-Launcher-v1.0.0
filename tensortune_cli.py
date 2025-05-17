#!/usr/bin/env python3
# TensorTune - VRAM Auto-Tuning Edition (CLI)
# Version 1.1.1 (CLI - TensorTune)

import json
import sys
import os
import subprocess
import re
import time
import threading
import signal # Keep for core's kill_process on non-Windows
import sqlite3
from datetime import datetime, timezone
import pathlib
import webbrowser
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import platform
from pathlib import Path
import shutil # For shutil.which

# Import the core script
import tensortune_core

# Handle dependency imports with helpful error messages
dependencies = {
    'rich': {'required': False, 'module': None, 'purpose': 'improved terminal UI'},
    'psutil': {'required': False, 'module': None, 'purpose': 'system resource monitoring (used by core)'},
    'pynvml': {'required': False, 'module': None, 'purpose': 'NVIDIA GPU VRAM detection (used by core)'},
    'tkinter': {'required': False, 'module': None, 'purpose': 'file open dialog (optional, less used in CLI)'}
}

for dep_name, dep_info in dependencies.items():
    try:
        if dep_name == 'tkinter': 
            pass 
        else:
            dependencies[dep_name]['module'] = __import__(dep_name)
    except ImportError:
        if dep_info['required']: 
            print(f"ERROR: Required dependency '{dep_name}' not found. Please install with: pip install {dep_name}")
            print(f"Purpose: {dep_info['purpose']}")
            sys.exit(1)

# Rich library setup (if available)
if dependencies['rich']['module']:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.live import Live
    from rich import print as rich_print
    console = Console()
    def print_title(text): console.print(Panel(text, style="bold blue", expand=False, title_align="left"))
    def print_success(text): console.print(f"[bold green]✓[/bold green] {text}")
    def print_error(text): console.print(f"[bold red]✗[/bold red] {text}")
    def print_warning(text): console.print(f"[bold yellow]![/bold yellow] {text}")
    def print_info(text): console.print(f"[cyan]ℹ[/cyan] {text}")
    def print_command(text): console.print(Syntax(text, "bash", theme="monokai", line_numbers=False, word_wrap=True))
    def prompt(text, choices=None, default=None, password=False):
        if choices: return Prompt.ask(text, choices=choices, default=default, password=password)
        return Prompt.ask(text, default=default, password=password)
    def confirm(text, default=True): return Confirm.ask(text, default=default)
else: # Fallback for no Rich
    def print_title(text): print(f"\n{'='*10} {text} {'='* (50-len(text) if len(text) < 50 else 0)}\n")
    def print_success(text): print(f"✓ {text}")
    def print_error(text): print(f"✗ {text}")
    def print_warning(text): print(f"! {text}")
    def print_info(text): print(f"ℹ {text}")
    def print_command(text): print(f"\n```bash\n{text}\n```\n")
    def prompt(text, choices=None, default=None, password=False): 
        prompt_text = text
        if choices: prompt_text += f" ({'/'.join(choices)})"
        if default is not None: prompt_text += f" [{default}]"
        prompt_text += ": "
        while True:
            response = input(prompt_text).strip()
            if not response and default is not None: return default
            if not choices or response in choices: return response
            print_error(f"Invalid choice. Please choose from: {', '.join(choices)}")
    def confirm(text, default=True):
        prompt_suffix = " [Y/n]" if default else " [y/N]"
        while True:
            response = input(f"{text}{prompt_suffix}: ").strip().lower()
            if not response: return default
            if response in ['y', 'yes']: return True
            if response in ['n', 'no']: return False
            print_error("Invalid input. Please answer 'yes' or 'no'.")

# --- Global Configuration Variables (populated from tensortune_core) ---
CONFIG: Dict[str, Any] = {}
KOBOLDCPP_EXECUTABLE = "" 
DB_FILE = "" 
DEFAULT_GGUF_DIR = "" 
AUTO_OPEN_WEBUI = True 
VRAM_SAFETY_BUFFER_MB = 768 
MIN_VRAM_FREE_AFTER_LOAD_MB = 512 
LOADING_TIMEOUT_SECONDS = 60 
VRAM_STABILIZATION_WAIT_S = 3.0
KOBOLD_SUCCESS_PATTERN = "" 
OOM_ERROR_KEYWORDS: List[str] = [] 
LAUNCHER_CLI_VERSION = "1.1.1 (CLI - TensorTune)"

# --- Runtime State Variables ---
last_gguf_directory = "" 
last_launched_process_info: Dict[str, Any] = {"pid": None, "process_obj": None, "command_list": []}
gguf_file_global = "" # For model selected in main menu
current_model_analysis_global: Dict[str, Any] = {} # For model selected in main menu

# Tuning-specific state
tuning_in_progress = False
current_tuning_attempt_level = 0
current_tuning_min_level = 0
current_tuning_max_level = 0
current_tuning_session_base_args: Dict[str, Any] = {} 
current_tuning_model_path_local = "" # Specific to current tuning session
current_tuning_model_analysis_local: Dict[str, Any] = {} # Specific to current tuning session
last_successful_monitored_run_details_cli: Optional[Dict[str, Any]] = None # For UI feedback in tuning

# KCPP Monitoring state (used by tuning)
kcpp_monitor_thread: Optional[threading.Thread] = None
kcpp_process_obj: Optional[subprocess.Popen] = None
kcpp_success_event = threading.Event()
kcpp_oom_event = threading.Event()
kcpp_output_lines_shared: List[str] = []
monitor_start_time: float = 0.0
user_requested_stop_monitoring_cli = False


def _update_cli_globals_from_config():
    global KOBOLDCPP_EXECUTABLE, DB_FILE, DEFAULT_GGUF_DIR, AUTO_OPEN_WEBUI
    global VRAM_SAFETY_BUFFER_MB, MIN_VRAM_FREE_AFTER_LOAD_MB, LOADING_TIMEOUT_SECONDS
    global VRAM_STABILIZATION_WAIT_S, KOBOLD_SUCCESS_PATTERN, OOM_ERROR_KEYWORDS
    global last_gguf_directory

    KOBOLDCPP_EXECUTABLE = CONFIG["koboldcpp_executable"]
    DB_FILE = CONFIG["db_file"]
    DEFAULT_GGUF_DIR = CONFIG.get("default_gguf_dir", "")
    AUTO_OPEN_WEBUI = CONFIG.get("auto_open_webui", True)
    VRAM_SAFETY_BUFFER_MB = CONFIG.get("vram_safety_buffer_mb", 768)
    MIN_VRAM_FREE_AFTER_LOAD_MB = CONFIG.get("min_vram_free_after_load_success_mb", 512)
    LOADING_TIMEOUT_SECONDS = CONFIG.get("loading_timeout_seconds", 60)
    VRAM_STABILIZATION_WAIT_S = CONFIG.get("vram_stabilization_wait_s", 3.0)
    KOBOLD_SUCCESS_PATTERN = CONFIG.get("kobold_success_pattern", r"Starting Kobold API on port (\d+)")
    OOM_ERROR_KEYWORDS = [k.lower() for k in CONFIG.get("oom_error_keywords", [])]
    last_gguf_directory = CONFIG.get("last_used_gguf_dir", "")


def _validate_and_update_kcpp_exe_path_in_config(config_dict: Dict[str, Any], input_path: str) -> bool:
    """Validates input_path and updates config_dict if valid and different. Returns True if path is valid."""
    resolved_path, message = tensortune_core.validate_and_resolve_koboldcpp_exe_path(input_path)
    if resolved_path:
        if config_dict["koboldcpp_executable"] != resolved_path:
            config_dict["koboldcpp_executable"] = resolved_path
            print_success(f"KoboldCpp executable path updated to: {resolved_path}")
        else:
            print_info(f"KoboldCpp executable path verified: {resolved_path}")
        return True
    else:
        print_error(f"KoboldCpp path validation failed: {message}")
        if input_path and input_path != config_dict["koboldcpp_executable"] : # Only print error if user provided a new, invalid path
             print_error(f"Keeping previous path: {config_dict['koboldcpp_executable']}")
        return False


def _update_gpu_backend_flags_in_config_cli(kcpp_caps_dict: Dict[str, Any]) -> bool:
    """Updates --usecublas/--usehipblas in CONFIG based on detected GPU and KCPP capabilities."""
    global CONFIG
    if "error" in kcpp_caps_dict:
        print_warning("Cannot auto-update GPU backend flags: KCPP capabilities detection failed.")
        return False 

    if "default_args" not in CONFIG: CONFIG["default_args"] = {}
    
    original_cublas = CONFIG["default_args"].get("--usecublas", False)
    original_hipblas = CONFIG["default_args"].get("--usehipblas", False)
    
    new_cublas = False
    new_hipblas = False

    # Get current GPU info based on config (selected_gpu_mode, etc.)
    _, _, _, current_gpu_info = tensortune_core.get_available_vram_mb(CONFIG)

    if current_gpu_info and current_gpu_info.get("success"):
        gpu_type = current_gpu_info.get("type", "").lower() # e.g., "nvidia", "amd"
        if "nvidia" in gpu_type and CONFIG.get("gpu_detection", {}).get("nvidia", True) and kcpp_caps_dict.get("cuda"):
            new_cublas = True
        elif "amd" in gpu_type and CONFIG.get("gpu_detection", {}).get("amd", True) and kcpp_caps_dict.get("rocm"):
            new_hipblas = True
    
    changed = False
    if new_cublas != original_cublas:
        CONFIG["default_args"]["--usecublas"] = new_cublas
        changed = True
    if new_hipblas != original_hipblas:
        CONFIG["default_args"]["--usehipblas"] = new_hipblas
        changed = True
    
    if changed:
        print_info(f"GPU backend flags auto-updated: CUBLAS={new_cublas}, HIPBLAS={new_hipblas}")
    return changed


def handle_first_run_prompts_cli(config_dict: Dict[str, Any]) -> bool:
    if config_dict.get('first_run_completed', False):
        return True

    print_title(" TensorTune Initial Setup ")
    if not config_dict.get('first_run_intro_shown', False):
        print_info("""
Welcome to the TensorTune CLI!
This tool helps you manage and launch KoboldCpp with auto-tuned settings.
We need to configure a few things for the first run.
        """)
        config_dict['first_run_intro_shown'] = True

    current_exe_in_config = config_dict.get("koboldcpp_executable", tensortune_core.DEFAULT_CONFIG_TEMPLATE["koboldcpp_executable"])
    user_exe_path_input = ""
    while True:
        user_exe_path_input = prompt(f"Enter path to KoboldCpp executable/script", default=current_exe_in_config).strip()
        if _validate_and_update_kcpp_exe_path_in_config(config_dict, user_exe_path_input):
            break
        # If validation fails, loop will continue asking for path.

    current_gguf_dir_in_config = config_dict.get("default_gguf_dir", os.getcwd())
    user_gguf_dir_path_input = prompt("Enter default GGUF model directory (or '.' for current, blank for none)",
                           default=current_gguf_dir_in_config if current_gguf_dir_in_config and os.path.isdir(current_gguf_dir_in_config) else os.getcwd()).strip()
    if user_gguf_dir_path_input == ".": user_gguf_dir_path_input = os.getcwd()
    
    if user_gguf_dir_path_input and not os.path.isdir(user_gguf_dir_path_input):
        print_warning(f"Directory '{user_gguf_dir_path_input}' invalid. No default GGUF directory will be set.")
        config_dict["default_gguf_dir"] = ""
    elif user_gguf_dir_path_input:
        config_dict["default_gguf_dir"] = os.path.abspath(user_gguf_dir_path_input)
    else: # Blank input
        config_dict["default_gguf_dir"] = ""
    
    # Prime last_used_gguf_dir with default if set
    if config_dict["default_gguf_dir"]:
        config_dict["last_used_gguf_dir"] = config_dict["default_gguf_dir"]

    db_file_location_absolute = config_dict["db_file"] # Already absolute from core
    print_info(f"History database will be stored at: {db_file_location_absolute}")

    default_port_from_core_template = tensortune_core.DEFAULT_CONFIG_TEMPLATE["default_args"]["--port"]
    current_port_in_config_dict = config_dict.get("default_args", {}).get("--port", default_port_from_core_template)
    while True:
        user_port_str_input = prompt(f"Default KoboldCpp port?", default=str(current_port_in_config_dict))
        try:
            user_port_int_val = int(user_port_str_input)
            if 1 <= user_port_int_val <= 65535:
                if "default_args" not in config_dict: config_dict["default_args"] = {}
                config_dict["default_args"]["--port"] = str(user_port_int_val)
                break
            else: print_error("Port must be between 1 and 65535.")
        except ValueError: print_error("Invalid port number.")

    config_dict["auto_open_webui"] = confirm("Auto-open Web UI in browser after successful launch?", default=config_dict.get("auto_open_webui", True))

    tkinter_available_check = False
    try: __import__('tkinter'); tkinter_available_check = True
    except ImportError: pass
    
    if tkinter_available_check:
        config_dict["cli_use_tkinter_dialog"] = confirm(
            "Prefer graphical file dialog for selecting models (if Tkinter is available)?",
            default=config_dict.get("cli_use_tkinter_dialog", False) 
        )
    else:
        config_dict["cli_use_tkinter_dialog"] = False 
        print_info("Tkinter (for graphical dialogs) not found. File selection will be manual path input.")

    print_info("\nGPU Detection Preferences (used for 'auto' GPU mode and VRAM display):")
    gpu_detect_prefs = config_dict.get("gpu_detection", tensortune_core.DEFAULT_CONFIG_TEMPLATE["gpu_detection"].copy())
    for vendor in ["nvidia", "amd", "intel", "apple"]:
        gpu_detect_prefs[vendor] = confirm(f"Enable detection for {vendor.upper()} GPUs?", default=gpu_detect_prefs.get(vendor, True))
    config_dict["gpu_detection"] = gpu_detect_prefs
    
    # Auto-update GPU backend flags based on KCPP caps and detected GPU after exe is set
    kcpp_caps_after_exe_set = tensortune_core.detect_koboldcpp_capabilities(config_dict["koboldcpp_executable"])
    _update_gpu_backend_flags_in_config_cli(kcpp_caps_after_exe_set) # Modifies config_dict directly

    config_dict['first_run_completed'] = True
    save_success, save_msg = tensortune_core.save_launcher_config(config_dict)
    if save_success: print_success("\nInitial setup complete! Configuration saved.")
    else: print_error(f"\nFailed to save initial configuration: {save_msg}")
    return save_success


def select_gguf_file_cli() -> Optional[str]:
    global last_gguf_directory, CONFIG, DEFAULT_GGUF_DIR 

    print_title("Select GGUF Model / Main Menu")

    main_menu_actions = { "s": "Select GGUF Model File", "t": "Launcher Settings", "h": "View Global Launch History", "q": "Quit Launcher" }
    print("Main Menu Options:"); [print(f"  ({k.upper()}) {d}") for k, d in main_menu_actions.items()]

    while True:
        action_choice = prompt("Your choice", choices=list(main_menu_actions.keys()), default="s").lower()
        if action_choice == 'q': return None
        if action_choice == 't':
            manage_launcher_settings_cli()
            _update_cli_globals_from_config() 
            print_title("Select GGUF Model / Main Menu"); [print(f"  ({k.upper()}) {d}") for k, d in main_menu_actions.items()]
            continue
        if action_choice == 'h':
            view_db_history_cli()
            print_title("Select GGUF Model / Main Menu"); [print(f"  ({k.upper()}) {d}") for k, d in main_menu_actions.items()]
            continue
        if action_choice == 's': break

    tkinter_available = False
    try:
        import tkinter as _tk_ 
        from tkinter import filedialog as _filedialog_
        tkinter_available = True
    except ImportError:
        pass

    use_dialog_from_config = CONFIG.get("cli_use_tkinter_dialog", False)

    if use_dialog_from_config and tkinter_available:
        try:
            root_tk = _tk_.Tk(); root_tk.withdraw(); root_tk.attributes('-topmost', True)
            start_dir_for_dialog = last_gguf_directory or DEFAULT_GGUF_DIR or os.getcwd()
            if not os.path.isdir(start_dir_for_dialog): start_dir_for_dialog = os.getcwd()

            filepath_selected = _filedialog_.askopenfilename(
                title="Select GGUF Model File",
                filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")],
                initialdir=start_dir_for_dialog
            )
            root_tk.destroy()
            if filepath_selected:
                abs_filepath = os.path.abspath(filepath_selected)
                last_gguf_directory = os.path.dirname(abs_filepath)
                CONFIG["last_used_gguf_dir"] = last_gguf_directory
                tensortune_core.save_launcher_config(CONFIG) 
                print_success(f"Selected via dialog: {os.path.basename(abs_filepath)}")
                return abs_filepath
            else:
                print_info("File selection cancelled via dialog.")
                return "main_menu"
        except Exception as e_tk:
            print_warning(f"Tkinter file dialog failed: {e_tk}. Falling back to manual path input.")
    elif use_dialog_from_config and not tkinter_available:
        print_warning("Graphical file dialog preferred but Tkinter is not available. Using manual input.")
    
    dir_for_manual_prompt = last_gguf_directory or DEFAULT_GGUF_DIR or os.getcwd()
    while True:
        filepath_manual_input = prompt(
            f"Enter full path to GGUF model file (or press Enter to cancel and return to main menu)\n"
            f"(Searches in: '{dir_for_manual_prompt}' if relative path is given)"
        ).strip()

        if not filepath_manual_input:
            print_info("File selection cancelled via manual input.")
            return "main_menu"

        potential_full_path = os.path.join(dir_for_manual_prompt, filepath_manual_input) \
            if not os.path.isabs(filepath_manual_input) else filepath_manual_input
        
        if os.path.isfile(potential_full_path) and potential_full_path.lower().endswith(".gguf"):
            abs_path_manual = os.path.abspath(potential_full_path)
            last_gguf_directory = os.path.dirname(abs_path_manual)
            CONFIG["last_used_gguf_dir"] = last_gguf_directory
            tensortune_core.save_launcher_config(CONFIG) 
            print_success(f"Selected via manual input: {os.path.basename(abs_path_manual)}")
            return abs_path_manual
        else:
            print_error(f"Path '{potential_full_path}' is not a valid .gguf file. Please try again or press Enter to cancel.")


def _display_kcpp_capabilities_cli():
    kcpp_exe_path = CONFIG.get("koboldcpp_executable", "")
    if not kcpp_exe_path:
        print_warning("KoboldCpp executable path not set. Cannot display capabilities.")
        return
    
    caps = tensortune_core.detect_koboldcpp_capabilities(kcpp_exe_path)
    print_title("Detected KoboldCpp Capabilities")
    if "error" in caps:
        print_error(f"Error detecting capabilities: {caps['error']}")
    else:
        friendly_names = {
            "cuda": "CUDA (cuBLAS)", "rocm": "ROCm (hipBLAS/rocBLAS)", "opencl": "OpenCL (CLBlast)",
            "vulkan": "Vulkan Backend", "flash_attn": "FlashAttention Support",
            "auto_quantkv": "Auto QuantKV Option", "overridetensors": "Tensor Override Support"
        }
        for key, name in friendly_names.items():
            status_str = "[green]Yes[/green]" if caps.get(key) else "[red]No[/red]"
            if dependencies['rich']['module']: console.print(f"  {name:<30}: {status_str}")
            else: print(f"  {name:<30}: {'Yes' if caps.get(key) else 'No'}")
        # Optionally, list some available_args if needed, but can be very long
        # available_args_list = caps.get("available_args", [])
        # if available_args_list and isinstance(available_args_list, list) and "Error" not in available_args_list[0]:
        #     print_info(f"  A subset of available args: {', '.join(available_args_list[:5])}...")


def manage_launcher_settings_cli():
    global CONFIG 
    print_title("Launcher Settings")
    
    while True:
        settings_options = {
            "1": "KoboldCpp Executable Path",
            "2": "Default GGUF Directory",
            "3": "Auto-Open WebUI",
            "4": "GPU Selection Mode",
            "5": "Target GPU Index/ID",
            "6": "VRAM Override Settings",
            "7": "GPU Detection Preferences",
            "8": "Use Graphical File Dialog (CLI)",
            "w": "Toggle Optional Library Startup Warnings",
            "k": "View/Re-detect KoboldCpp Capabilities",
            "gda": "Edit Global KoboldCpp Default Arguments",
            "msa": "Manage Model-Specific Arguments",
            "e": "Export Configuration to File",
            "i": "Import Configuration from File",
            "r": "Reset All Settings to Defaults",
            "b": "Back to Main Menu"
        }
        print("\nSettings Menu:")
        for k, v in settings_options.items(): print(f"  ({k}) {v}")
        
        current_suppress_status = "DISABLED (will show warnings)"
        if CONFIG.get("suppress_optional_lib_warnings", False):
            current_suppress_status = "ENABLED (warnings hidden after first run)"
        print_info(f"Current startup library warning suppression: {current_suppress_status}")

        choice = prompt("Choose an option", choices=list(settings_options.keys()), default="b").lower()

        if choice == 'b': break

        elif choice == '1':
            old_exe = CONFIG["koboldcpp_executable"]
            new_exe_input = prompt(f"KoboldCpp Executable Path", default=old_exe).strip()
            if new_exe_input: 
                if _validate_and_update_kcpp_exe_path_in_config(CONFIG, new_exe_input):
                    kcpp_caps_new = tensortune_core.detect_koboldcpp_capabilities(CONFIG["koboldcpp_executable"])
                    if _update_gpu_backend_flags_in_config_cli(kcpp_caps_new):
                        print_info("GPU backend flags in config may have been updated.")
        
        elif choice == '2':
            old_dir = CONFIG["default_gguf_dir"]
            new_dir = prompt(f"Default GGUF Directory (blank to clear)", default=old_dir if old_dir else os.getcwd()).strip()
            if new_dir and os.path.isdir(new_dir):
                CONFIG["default_gguf_dir"] = os.path.abspath(new_dir)
                print_success(f"Default GGUF directory set to: {CONFIG['default_gguf_dir']}")
            elif not new_dir: 
                 CONFIG["default_gguf_dir"] = ""
                 print_success("Default GGUF directory cleared.")
            else: 
                print_error(f"Directory '{new_dir}' is not valid. Keeping: {old_dir if old_dir else 'None'}")
        
        elif choice == '3':
            CONFIG["auto_open_webui"] = confirm("Auto-Open WebUI after successful launch?", default=CONFIG.get("auto_open_webui", True))
            print_success(f"Auto-Open WebUI set to: {CONFIG['auto_open_webui']}")

        elif choice == '4':
            current_mode = CONFIG.get("gpu_selection_mode", "auto")
            modes = ["auto", "nvidia", "amd", "intel", "apple"]
            new_mode = prompt(f"GPU Selection Mode ({'/'.join(modes)})", choices=modes, default=current_mode).lower()
            CONFIG["gpu_selection_mode"] = new_mode
            print_success(f"GPU Selection Mode set to: {new_mode}")

        elif choice == '5':
            current_idx = CONFIG.get("selected_gpu_index", 0)
            gpu_type_for_listing = CONFIG.get("gpu_selection_mode", "auto")
            
            effective_gpu_type = gpu_type_for_listing
            if gpu_type_for_listing == "auto":
                _, _, _, gpu_info_auto = tensortune_core.get_available_vram_mb(CONFIG)
                if gpu_info_auto and gpu_info_auto.get("success") and gpu_info_auto.get("type") not in ["Unknown/None_Auto", "N/A", "Unknown/None", "INVALID_TARGET_PREFS"]:
                    detected_vendor = gpu_info_auto.get("type", "auto").lower()
                    if "nvidia" in detected_vendor: effective_gpu_type = "nvidia"
                    elif "amd" in detected_vendor: effective_gpu_type = "amd"
                    elif "intel" in detected_vendor: effective_gpu_type = "intel"
                    elif "apple_metal" in detected_vendor: effective_gpu_type = "apple"
                    else: effective_gpu_type = "none" 
                    
                    if effective_gpu_type != "none": print_info(f"Auto-detected GPU type as '{effective_gpu_type}' for listing IDs.")
                    else: print_warning("Could not auto-detect a specific GPU type for ID listing.")
                else: print_warning("Could not auto-detect a specific GPU type for ID listing."); effective_gpu_type = "none" 

            gpus_found = []
            if effective_gpu_type == "nvidia": gpus_found = tensortune_core.list_nvidia_gpus()
            elif effective_gpu_type == "amd": gpus_found = tensortune_core.list_amd_gpus_windows() if platform.system() == "Windows" else tensortune_core.list_amd_gpus_linux()
            elif effective_gpu_type == "intel": gpus_found = tensortune_core.list_intel_gpus()
            elif effective_gpu_type == "apple": gpus_found = tensortune_core.list_apple_gpus()
            
            if gpus_found:
                print_info(f"Available GPUs for type '{effective_gpu_type.upper()}':")
                for gpu in gpus_found: print(f"  ID {gpu['id']}: {gpu['name']}")
                while True:
                    try:
                        new_idx_str = prompt(f"Target GPU Index/ID", default=str(current_idx))
                        new_idx = int(new_idx_str)
                        if any(gpu['id'] == new_idx for gpu in gpus_found):
                            CONFIG["selected_gpu_index"] = new_idx
                            print_success(f"Target GPU Index set to: {new_idx}")
                            break
                        else: print_error(f"Invalid index. Choose from available IDs for type '{effective_gpu_type.upper()}'.")
                    except ValueError: print_error("Invalid input. Please enter a number.")
            else: print_warning(f"No GPUs listed for type '{effective_gpu_type.upper()}'. Index remains {current_idx}.")

        elif choice == '6':
            print_info("VRAM Override Settings:")
            current_override_status = CONFIG.get("override_vram_budget", False)
            current_manual_budget = CONFIG.get("manual_vram_total_mb", 8192)
            
            if current_override_status: print_info(f"VRAM Override is ACTIVE. Manual Budget: {current_manual_budget} MB.")
            else: print_info("VRAM Override is INACTIVE. Auto-detected VRAM will be used.")

            enable_override = confirm("Enable VRAM Override?", default=current_override_status)
            CONFIG["override_vram_budget"] = enable_override
            if enable_override:
                while True:
                    try:
                        new_manual_vram_str = prompt(f"Enter Manual Total VRAM (MB)", default=str(current_manual_budget))
                        new_manual_vram = int(new_manual_vram_str)
                        if new_manual_vram >=0: 
                           CONFIG["manual_vram_total_mb"] = new_manual_vram
                           print_success(f"VRAM Override ENABLED. Manual Budget set to: {new_manual_vram} MB.")
                           break
                        else: print_error("VRAM cannot be negative.")
                    except ValueError: print_error("Invalid input. Please enter a number.")
            else: print_success("VRAM Override DISABLED.")
        
        elif choice == '7':
            print_info("GPU Detection Preferences:")
            gpu_detect_prefs = CONFIG.get("gpu_detection", tensortune_core.DEFAULT_CONFIG_TEMPLATE["gpu_detection"].copy())
            for vendor in ["nvidia", "amd", "intel", "apple"]:
                gpu_detect_prefs[vendor] = confirm(f"Enable detection for {vendor.upper()} GPUs?", default=gpu_detect_prefs.get(vendor, True))
            CONFIG["gpu_detection"] = gpu_detect_prefs
            print_success("GPU detection preferences updated.")

        elif choice == '8':
            tkinter_available_check = False
            try: __import__('tkinter'); tkinter_available_check = True
            except ImportError: pass
            
            if tkinter_available_check:
                current_pref = CONFIG.get("cli_use_tkinter_dialog", False)
                CONFIG["cli_use_tkinter_dialog"] = confirm(
                    "Use graphical file dialog for model selection in CLI (requires Tkinter)?",
                    default=current_pref
                )
                print_success(f"Graphical file dialog preference set to: {CONFIG['cli_use_tkinter_dialog']}")
            else:
                CONFIG["cli_use_tkinter_dialog"] = False 
                print_warning("Tkinter (for graphical dialogs) not found. Preference set to False.")
        
        elif choice == 'w':
            current_setting = CONFIG.get("suppress_optional_lib_warnings", False)
            new_setting = confirm("Suppress optional library warnings on startup (after first run)?", default=not current_setting)
            CONFIG["suppress_optional_lib_warnings"] = new_setting
            if new_setting:
                print_success("Optional library warnings will be suppressed on subsequent startups.")
            else:
                print_success("Optional library warnings will be shown on startup.")
                
        elif choice == 'k':
            _display_kcpp_capabilities_cli()
            if confirm("Re-detect KCPP capabilities now?", default=False):
                kcpp_caps_redetect = tensortune_core.detect_koboldcpp_capabilities(CONFIG["koboldcpp_executable"], force_redetect=True)
                _display_kcpp_capabilities_cli() 
                if _update_gpu_backend_flags_in_config_cli(kcpp_caps_redetect):
                    print_info("GPU backend flags in config may have been updated.")
        
        elif choice == 'gda':
            print_info("Editing Global KoboldCpp Default Arguments...")
            global_defaults_in_config = CONFIG.get("default_args", {}).copy()
            updated_global_defaults, _ = edit_current_args_interactive_cli(
                model_path_for_specifics=None, 
                current_session_args_overrides=global_defaults_in_config, 
                is_global_edit_mode=True 
            )
            if updated_global_defaults is not None:
                CONFIG["default_args"] = updated_global_defaults
                print_success("Global default arguments updated.")
            else:
                print_info("Global default argument edit cancelled or no changes made.")
        
        elif choice == 'msa':
            manage_model_specific_args_cli()

        elif choice == 'e': 
            export_path = prompt("Enter filepath to export configuration (e.g., launcher_backup.json)", default="kcpp_launcher_config_export.json").strip()
            if export_path:
                success_export, msg_export = tensortune_core.export_config_to_file(CONFIG.copy(), export_path)
                if success_export: print_success(msg_export)
                else: print_error(msg_export)
        
        elif choice == 'i': 
            import_path = prompt("Enter filepath of configuration to import").strip()
            if import_path and os.path.exists(import_path):
                imported_data, msg_import = tensortune_core.import_config_from_file(import_path)
                if imported_data:
                    if confirm(f"Importing will OVERWRITE current settings. Backup current config? ({tensortune_core.CONFIG_FILE})", default=True):
                        backup_path_auto = tensortune_core.CONFIG_FILE + f".backup_before_import_{time.strftime('%Y%m%d-%H%M%S')}.json"
                        try:
                            if os.path.exists(tensortune_core.CONFIG_FILE): 
                                shutil.copy2(tensortune_core.CONFIG_FILE, backup_path_auto)
                                print_success(f"Current config backed up to {backup_path_auto}")
                            else: print_warning(f"Current config file {tensortune_core.CONFIG_FILE} not found. Cannot backup.")
                        except Exception as e_backup: print_warning(f"Could not backup current config: {e_backup}")
                    
                    tensortune_core.save_launcher_config(CONFIG) 

                    temp_config_for_saving_import = imported_data.copy() 
                    db_file_val_from_imported = temp_config_for_saving_import.get("db_file", tensortune_core.DEFAULT_CONFIG_TEMPLATE["db_file"])
                    if not os.path.isabs(db_file_val_from_imported) and os.path.basename(db_file_val_from_imported) == db_file_val_from_imported:
                        temp_config_for_saving_import["db_file"] = os.path.join(tensortune_core._get_user_app_data_dir(), db_file_val_from_imported)

                    save_imported_ok, save_imported_msg = tensortune_core.save_launcher_config(temp_config_for_saving_import)
                    if save_imported_ok:
                        CONFIG, _, _ = tensortune_core.load_config()
                        _update_cli_globals_from_config()
                        print_success(f"{msg_import} New configuration saved and reloaded.")
                    else:
                        print_error(f"Failed to save imported configuration: {save_imported_msg}. Current settings remain.")
                else: print_error(msg_import)
            elif import_path: print_error(f"Import file not found: {import_path}")

        elif choice == 'r': 
             if confirm("WARNING: This will reset ALL launcher settings to defaults. Are you sure?", default=False):
                backup_path_reset = tensortune_core.CONFIG_FILE + f".backup_before_reset_{time.strftime('%Y%m%d-%H%M%S')}.json"
                try:
                    if os.path.exists(tensortune_core.CONFIG_FILE): shutil.copy2(tensortune_core.CONFIG_FILE, backup_path_reset); print_success(f"Current config backed up to {backup_path_reset}")
                    if os.path.exists(tensortune_core.CONFIG_FILE): os.remove(tensortune_core.CONFIG_FILE)
                except Exception as e_remove: print_warning(f"Could not remove/backup current config for reset: {e_remove}")

                temp_conf, _, load_msg = tensortune_core.load_config() 
                CONFIG = temp_conf
                _update_cli_globals_from_config()
                print_success(f"Configuration reset to defaults. {load_msg}")
                print_warning("You may need to run the first-time setup prompts again if paths changed.")

        if choice not in ['i', 'r', 'b']: # 'b' doesn't change settings
            save_ok, save_message = tensortune_core.save_launcher_config(CONFIG)
            if not save_ok: print_error(f"Failed to save settings: {save_message}")
            else:
                if choice != 'k': # 'k' view doesn't change settings unless re-detect modifies flags
                    print_info("Settings saved.")
        _update_cli_globals_from_config() 
        print("-" * 30)     


def manage_model_specific_args_cli():
    global CONFIG
    if "model_specific_args" not in CONFIG: CONFIG["model_specific_args"] = {}

    while True:
        print_title("Manage Model-Specific Arguments")
        model_configs = CONFIG["model_specific_args"]
        if not model_configs:
            print_info("No model-specific configurations exist.")
        else:
            print_info("Existing model-specific configurations:")
            for i, (path, args) in enumerate(model_configs.items()):
                print(f"  ({i+1}) {os.path.basename(path)} ({len(args)} overrides)")
        
        print("\nActions: (A)dd/Edit | (D#)elete # | (B)ack to Settings")
        choice = prompt("Choose action", default="b").lower()

        if choice == 'b': break
        elif choice == 'a':
            model_path_input = prompt("Enter full path to GGUF model for specific config (or Enter to cancel)").strip()
            if not model_path_input: continue
            
            abs_model_path = ""
            # Try to resolve if it's a relative path from default GGUF dir or current dir
            if not os.path.isabs(model_path_input):
                dir_to_check_relative = DEFAULT_GGUF_DIR or os.getcwd()
                potential_abs = os.path.join(dir_to_check_relative, model_path_input)
                if os.path.isfile(potential_abs):
                    abs_model_path = os.path.abspath(potential_abs)
            elif os.path.isfile(model_path_input): # Absolute path was given
                abs_model_path = os.path.abspath(model_path_input)
            
            if not abs_model_path or not abs_model_path.lower().endswith(".gguf"):
                print_error(f"Invalid GGUF path: {model_path_input}. Please provide a valid .gguf file path.")
                continue
            
            print_info(f"Editing specific args for: {os.path.basename(abs_model_path)}")
            current_model_specifics = CONFIG["model_specific_args"].get(abs_model_path, {}).copy()
            
            updated_model_specifics, _ = edit_current_args_interactive_cli(
                model_path_for_specifics=abs_model_path, 
                current_session_args_overrides=current_model_specifics, # Pass its own specifics as "session"
                is_global_edit_mode=False # We are editing model-specifics, not globals
            )
            if updated_model_specifics is not None:
                # The editor returns the *overrides* compared to global+core.
                # We need to store these overrides.
                final_overrides_to_store = {}
                baseline_for_model = tensortune_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
                baseline_for_model.update(CONFIG.get("default_args", {}))

                for key, val_from_editor in updated_model_specifics.items():
                    baseline_val_for_this_key = baseline_for_model.get(key)
                    # Normalize booleans for comparison
                    is_bool = False
                    arg_def_local = next((d for d in tensortune_core.KOBOLDCPP_ARG_DEFINITIONS if d["key"] == key), None)
                    if arg_def_local and arg_def_local.get("type_hint") in ["bool", "bool_flag"]:
                        is_bool = True
                    
                    val_to_compare_editor = val_from_editor
                    baseline_to_compare = baseline_val_for_this_key
                    if is_bool:
                        if isinstance(val_from_editor, str): val_to_compare_editor = val_from_editor.lower() == 'true'
                        if isinstance(baseline_val_for_this_key, str): baseline_to_compare = baseline_val_for_this_key.lower() == 'true'
                    
                    if str(val_to_compare_editor) != str(baseline_to_compare):
                        final_overrides_to_store[key] = val_from_editor
                
                if final_overrides_to_store:
                    CONFIG["model_specific_args"][abs_model_path] = final_overrides_to_store
                elif abs_model_path in CONFIG["model_specific_args"]: # If all overrides were removed
                    del CONFIG["model_specific_args"][abs_model_path]
                print_success(f"Model-specific arguments updated for {os.path.basename(abs_model_path)}.")
            else:
                print_info("Model-specific argument edit cancelled or no changes made.")

        elif choice.startswith('d') and choice[1:].isdigit():
            idx_to_delete = int(choice[1:]) - 1
            if 0 <= idx_to_delete < len(model_configs):
                path_to_delete = list(model_configs.keys())[idx_to_delete]
                if confirm(f"Delete specific config for {os.path.basename(path_to_delete)}?", default=True):
                    del CONFIG["model_specific_args"][path_to_delete]
                    print_success(f"Deleted specific config for {os.path.basename(path_to_delete)}.")
            else:
                print_error("Invalid number for deletion.")
        
        tensortune_core.save_launcher_config(CONFIG) # Save after any modification


def view_db_history_cli(model_filepath_filter: Optional[str] = None):
    if model_filepath_filter:
        print_info(f"Loading history for model: {os.path.basename(model_filepath_filter)} from DB: {DB_FILE}")
    else:
        print_info(f"Loading global launch history from DB: {DB_FILE}")

    all_history_entries_from_db = tensortune_core.get_history_entries(DB_FILE, limit=100)

    if not all_history_entries_from_db:
        print_info("No history records found in the database.")
        return

    entries_to_display = []
    if model_filepath_filter:
        for entry_tuple in all_history_entries_from_db:
            if entry_tuple[0] == model_filepath_filter:
                entries_to_display.append(entry_tuple)
        if not entries_to_display:
            print_info(f"No history records found for model: {os.path.basename(model_filepath_filter)}")
            return
        display_title = f"Launch History for {os.path.basename(model_filepath_filter)} (Up to 20 most recent)"
        entries_to_show_on_screen = entries_to_display[:20]
    else:
        display_title = "Global Launch History (Up to 20 most recent)"
        entries_to_show_on_screen = all_history_entries_from_db[:20]

    if dependencies['rich']['module']:
        history_table = Table(title=display_title)
        column_names = ["Model", "Size(B)", "Quant", "MoE", "VRAM@Launch", "OT Lvl", "Outcome", "VRAM Used(MB)", "Timestamp"]
        column_styles = ["cyan", "magenta", "yellow", "magenta", "green", "blue", "white", "green", "dim"]
        column_justifies = ["left", "right", "center", "center", "right", "center", "left", "right", "left"]
        for col_name, style, justify_opt in zip(column_names, column_styles, column_justifies):
            history_table.add_column(col_name, style=style, justify=justify_opt, overflow="fold", min_width=5)

        for record_data in entries_to_show_on_screen:
            model_filename = os.path.basename(record_data[0])
            size_b_val = f"{record_data[1]:.1f}" if isinstance(record_data[1], float) else (str(record_data[1]) if record_data[1] is not None else "N/A")
            quant_val = str(record_data[2]) if record_data[2] else "N/A"
            is_moe_val = "Y" if record_data[3] else "N"
            vram_at_launch_val = f"{record_data[4]}MB" if record_data[4] is not None else "N/A" 
            ot_level_val = str(record_data[5]) if record_data[5] is not None else "N/A"
            outcome_val = str(record_data[6]) if record_data[6] else "N/A"
            vram_used_val = f"{record_data[7]}MB" if record_data[7] is not None else "N/A" 
            timestamp_obj = record_data[8]
            timestamp_str_val = timestamp_obj.strftime('%y-%m-%d %H:%M') if isinstance(timestamp_obj, datetime) else str(timestamp_obj)[:16]
            history_table.add_row(model_filename, size_b_val, quant_val, is_moe_val, vram_at_launch_val, ot_level_val, outcome_val, vram_used_val, timestamp_str_val)
        console.print(history_table)
    else:
        print_title(display_title)
        header_fmt = f"{'Model':<28} | {'Sz':<5} | {'Quant':<9} | {'MoE':<3} | {'VRAM@L':<8} | {'Lvl':<3} | {'Outcome':<38} | {'VRAMUsed':<8} | {'Timestamp':<16}"
        print(header_fmt); print("-" * len(header_fmt))
        for record_data in entries_to_show_on_screen:
            model_fn = os.path.basename(record_data[0])[:26]
            size_b_str = f"{record_data[1]:.1f}" if isinstance(record_data[1], float) else (str(record_data[1]) if record_data[1] is not None else "N/A")
            quant_s = (str(record_data[2]) if record_data[2] else "N/A")[:9]
            is_moe_s = "Y" if record_data[3] else "N"
            vram_l_s = str(record_data[4]) if record_data[4] is not None else "N/A"
            ot_lvl_s = str(record_data[5]) if record_data[5] is not None else "N/A"
            outcome_s = (str(record_data[6]) if record_data[6] else "N/A")[:38]
            vram_u_s = str(record_data[7]) if record_data[7] is not None else "N/A"
            ts_obj = record_data[8]
            timestamp_s_val = ts_obj.strftime('%y-%m-%d %H:%M') if isinstance(ts_obj, datetime) else str(ts_obj)[:16]
            print(f"{model_fn:<28} | {size_b_str:<5} | {quant_s:<9} | {is_moe_s:<3} | {vram_l_s:<8} | {ot_lvl_s:<3} | {outcome_s:<38} | {vram_u_s:<8} | {timestamp_s_val:<16}")


def get_effective_session_args(model_file_path: Optional[str], session_specific_overrides: Dict[str, Any]) -> Dict[str, Any]:
    effective_args_dict = tensortune_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
    global_defaults_from_config = CONFIG.get("default_args", {})
    effective_args_dict.update(global_defaults_from_config)

    if model_file_path and "model_specific_args" in CONFIG:
        model_specifics_from_config = CONFIG["model_specific_args"].get(model_file_path, {})
        effective_args_dict.update(model_specifics_from_config)

    effective_args_dict.update(session_specific_overrides)
    return effective_args_dict


def edit_current_args_interactive_cli(model_path_for_specifics: Optional[str], 
                                     current_session_args_overrides: Dict[str, Any],
                                     is_global_edit_mode: bool = False
                                     ) -> Tuple[Optional[Dict[str, Any]], bool]:
    permanent_args_were_changed_in_model_mode = False 
    target_args_to_modify = current_session_args_overrides.copy()
    title_text = "Edit Global KCPP Default Arguments" if is_global_edit_mode else "Edit Base Startup Arguments (KCPP Parameters)"
    
    while True:
        print_title(title_text)
        baseline_args_for_comparison = tensortune_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
        if not is_global_edit_mode: 
            baseline_args_for_comparison.update(CONFIG.get("default_args", {}))
            if model_path_for_specifics:
                 baseline_args_for_comparison.update(CONFIG.get("model_specific_args", {}).get(model_path_for_specifics, {}))
        
        effective_args_for_display = baseline_args_for_comparison.copy()
        effective_args_for_display.update(target_args_to_modify)

        editable_arg_defs = [d for d in tensortune_core.KOBOLDCPP_ARG_DEFINITIONS if d["key"] not in ["--model", "--overridetensors"]]
        editable_arg_defs.sort(key=lambda x: (x.get("category", "zz_default"), x.get("name", x["key"])))
        idx_to_param_map = {}

        if dependencies['rich']['module']:
            args_table = Table(title="Effective Arguments (Core -> Global -> Model -> Session)" if not is_global_edit_mode else "Global Default Arguments (Core -> Global)")
            args_table.add_column("#", style="cyan"); args_table.add_column("Argument", style="green", max_width=20, overflow="fold");
            args_table.add_column("Current Value", style="yellow", max_width=15, overflow="fold"); args_table.add_column("Description", overflow="fold");
            
            for i, arg_def in enumerate(editable_arg_defs):
                param_key = arg_def["key"]; idx_str_display = str(i + 1); idx_to_param_map[idx_str_display] = param_key
                value_to_display = effective_args_for_display.get(param_key)
                is_bool_type_param = arg_def.get("type_hint") in ["bool", "bool_flag"]
                
                value_str_display = ""
                if is_bool_type_param: 
                    bool_val = False # Default if None or unexpected type
                    if isinstance(value_to_display, bool): bool_val = value_to_display
                    elif isinstance(value_to_display, str): bool_val = value_to_display.lower() == 'true'
                    value_str_display = "[green]ENABLED[/green]" if bool_val else "[red]DISABLED[/red]"
                else: 
                    value_str_display = str(value_to_display) if value_to_display is not None else "[dim]NOT SET[/dim]"
                
                args_table.add_row(idx_str_display, arg_def.get("name", param_key), value_str_display, arg_def.get("help", ""))
            console.print(args_table)
        else: 
            print("Effective Arguments:")
            for i, arg_def in enumerate(editable_arg_defs):
                param_key = arg_def["key"]; idx_str_display = str(i + 1); idx_to_param_map[idx_str_display] = param_key
                value_to_display = effective_args_for_display.get(param_key)
                is_bool_type_param = arg_def.get("type_hint") in ["bool", "bool_flag"]
                
                bool_val = False
                if isinstance(value_to_display, bool): bool_val = value_to_display
                elif isinstance(value_to_display, str): bool_val = value_to_display.lower() == 'true'
                value_str_display = ("ENABLED" if bool_val else "DISABLED") if is_bool_type_param else (str(value_to_display) if value_to_display is not None else "NOT SET")
                print(f"  ({idx_str_display}) {arg_def.get('name', param_key):<25}: {value_str_display:<15} - {arg_def.get('help', '')}")

        print("\nActions: (#) to edit value | (T#)oggle boolean | (U#)nset override/global")
        if not is_global_edit_mode and model_path_for_specifics:
            print("         (P)ermanently save current session args for this model")
        print("         (S)ave changes & Continue | (X) Cancel edits & Continue")
        user_choice_input = prompt("Your choice", default="s").lower().strip()

        if user_choice_input == 's': return target_args_to_modify, permanent_args_were_changed_in_model_mode
        if user_choice_input == 'x': return current_session_args_overrides, permanent_args_were_changed_in_model_mode

        if user_choice_input == 'p' and not is_global_edit_mode and model_path_for_specifics:
            print_info(f"Saving effective arguments as permanent defaults for '{os.path.basename(model_path_for_specifics)}'...")
            if confirm(f"This will overwrite existing model-specific settings for {os.path.basename(model_path_for_specifics)}. Proceed?", default=True):
                if "model_specific_args" not in CONFIG: CONFIG["model_specific_args"] = {}
                
                new_model_specific_settings = {}
                core_plus_global_baseline = tensortune_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
                core_plus_global_baseline.update(CONFIG.get("default_args", {}))

                for key, effective_val in effective_args_for_display.items():
                    baseline_val = core_plus_global_baseline.get(key)
                    arg_def_for_perm = next((d for d in editable_arg_defs if d["key"] == key), None)
                    is_bool_perm = arg_def_for_perm and arg_def_for_perm.get("type_hint") in ["bool", "bool_flag"]

                    current_val_to_compare = effective_val; baseline_val_to_compare = baseline_val
                    if is_bool_perm:
                        if isinstance(effective_val, str): current_val_to_compare = effective_val.lower() == 'true'
                        else: current_val_to_compare = bool(effective_val) # Normalize
                        if isinstance(baseline_val, str): baseline_val_to_compare = baseline_val.lower() == 'true'
                        else: baseline_val_to_compare = bool(baseline_val)
                    
                    if str(current_val_to_compare) != str(baseline_val_to_compare):
                        new_model_specific_settings[key] = effective_val
                
                CONFIG["model_specific_args"][model_path_for_specifics] = new_model_specific_settings
                if not CONFIG["model_specific_args"][model_path_for_specifics]: 
                    del CONFIG["model_specific_args"][model_path_for_specifics]

                tensortune_core.save_launcher_config(CONFIG)
                permanent_args_were_changed_in_model_mode = True
                print_success(f"Permanent arguments saved for {os.path.basename(model_path_for_specifics)}.")
                target_args_to_modify.clear() 
            else: print_info("Permanent argument save cancelled.")
            continue

        arg_to_edit_name = None; action_type = None 
        raw_idx_str = ""
        if user_choice_input.startswith('t') and user_choice_input[1:].isdigit():
            raw_idx_str = user_choice_input[1:]; action_type = 'toggle'
        elif user_choice_input.startswith('u') and user_choice_input[1:].isdigit():
            raw_idx_str = user_choice_input[1:]; action_type = 'unset'
        elif user_choice_input.isdigit():
            raw_idx_str = user_choice_input; action_type = 'edit'
        
        if action_type and raw_idx_str in idx_to_param_map:
            arg_to_edit_name = idx_to_param_map[raw_idx_str]
        
        if arg_to_edit_name:
            arg_def_for_edit = next((d for d in editable_arg_defs if d["key"] == arg_to_edit_name), None)
            if not arg_def_for_edit: print_error("Internal error: Arg definition not found."); continue
            param_display_name = arg_def_for_edit.get('name', arg_to_edit_name)
            widget_type_hint_edit = arg_def_for_edit.get("type_hint", "str")

            if action_type == 'toggle':
                if widget_type_hint_edit in ["bool", "bool_flag"]:
                    current_effective_val_for_toggle = effective_args_for_display.get(arg_to_edit_name, False)
                    actual_current_bool = False
                    if isinstance(current_effective_val_for_toggle, bool): actual_current_bool = current_effective_val_for_toggle
                    elif isinstance(current_effective_val_for_toggle, str): actual_current_bool = current_effective_val_for_toggle.lower() == 'true'
                    
                    target_args_to_modify[arg_to_edit_name] = not actual_current_bool
                    print_success(f"Toggled '{param_display_name}' to {'ENABLED' if target_args_to_modify[arg_to_edit_name] else 'DISABLED'}")
                else: print_error(f"Cannot toggle '{param_display_name}'. Not a boolean type.")
            elif action_type == 'unset':
                if arg_to_edit_name in target_args_to_modify:
                    del target_args_to_modify[arg_to_edit_name]
                print_success(f"Unset override for '{param_display_name}'. It will use baseline value.")
            elif action_type == 'edit':
                current_val_for_edit_prompt = effective_args_for_display.get(arg_to_edit_name)
                prompt_msg = f"New value for {param_display_name} (Current: {current_val_for_edit_prompt if current_val_for_edit_prompt is not None else 'Default'}):"
                new_val_str_input = prompt(prompt_msg).strip()

                if widget_type_hint_edit in ["bool", "bool_flag"]:
                    if new_val_str_input.lower() in ['true', 'yes', '1', 'on', 'enabled', 'y']: target_args_to_modify[arg_to_edit_name] = True
                    elif new_val_str_input.lower() in ['false', 'no', '0', 'off', 'disabled', 'n']: target_args_to_modify[arg_to_edit_name] = False
                    else: print_error(f"Invalid boolean. Use 'true'/'false'."); continue
                elif widget_type_hint_edit in ["int", "str_auto_num"] and new_val_str_input.lower() != "auto": # Allow 'auto' for str_auto_num
                    try: target_args_to_modify[arg_to_edit_name] = int(new_val_str_input) if new_val_str_input.isdigit() else new_val_str_input
                    except ValueError: print_error(f"Invalid integer for non-'auto'."); continue
                else: 
                    target_args_to_modify[arg_to_edit_name] = new_val_str_input
                print_success(f"Set override for '{param_display_name}' to {target_args_to_modify.get(arg_to_edit_name)}")
        elif user_choice_input not in ['s', 'x', 'p']: 
            print_error("Invalid choice.")


def _log_to_cli_live_output(text_line: str, console_obj_for_rich: Optional[Console] = None):
    if console_obj_for_rich and dependencies['rich']['module']:
        # When using Rich Progress, printing directly to console can mess with the bar.
        # It's better if the Progress bar itself includes any dynamic text via its columns.
        # For general KCPP output, printing it interleaved might be acceptable if console handles it.
        console_obj_for_rich.print(text_line.rstrip(),markup=False,highlight=False, overflow="ignore") # ignore overflow to prevent wrapping issues with progress bar
    else:
        sys.stdout.write(text_line)
        sys.stdout.flush()


def monitor_kcpp_output_thread_target_cli(
    process: subprocess.Popen,
    success_event_thread: threading.Event,
    oom_event_thread: threading.Event,
    output_lines_list_shared: List[str],
    success_regex_str_config: str,
    oom_keywords_list_config: List[str],
    target_port_for_success_check: str,
    console_obj_for_rich: Optional[Console] = None # Pass console for Rich printing
):
    global user_requested_stop_monitoring_cli
    try:
        for line_bytes_from_kcpp in iter(process.stdout.readline, b''):
            if not line_bytes_from_kcpp or user_requested_stop_monitoring_cli: break
            try: line_decoded_from_kcpp = line_bytes_from_kcpp.decode('utf-8', errors='replace')
            except UnicodeDecodeError: line_decoded_from_kcpp = line_bytes_from_kcpp.decode('latin-1', errors='replace')
            
            _log_to_cli_live_output(line_decoded_from_kcpp, console_obj_for_rich)
            
            line_strip_lower_case = line_decoded_from_kcpp.strip().lower()
            if line_strip_lower_case:
                output_lines_list_shared.append(line_decoded_from_kcpp.strip())
                
                if not success_event_thread.is_set() and not oom_event_thread.is_set():
                    success_match_found = re.search(success_regex_str_config, line_decoded_from_kcpp.strip(), re.IGNORECASE)
                    if success_match_found:
                        launched_port_from_log = target_port_for_success_check
                        try: launched_port_from_log = success_match_found.group(1)
                        except IndexError: pass
                        
                        if str(launched_port_from_log) == str(target_port_for_success_check):
                            success_event_thread.set()
                    
                    if not success_event_thread.is_set():
                        for oom_keyword in oom_keywords_list_config: 
                            if oom_keyword in line_strip_lower_case:
                                oom_event_thread.set(); break
            if success_event_thread.is_set() or oom_event_thread.is_set(): break
    except Exception as e_monitor:
        _log_to_cli_live_output(f"\nError in KCPP output monitor thread: {type(e_monitor).__name__}: {e_monitor}\n", console_obj_for_rich)
    finally:
        if process.stdout and not process.stdout.closed:
            try: process.stdout.close()
            except: pass
        log_msg_thread_end = "\nKCPP output monitoring thread exiting due to user stop request.\n" if user_requested_stop_monitoring_cli else "\nKCPP output monitoring thread finished.\n"
        _log_to_cli_live_output(log_msg_thread_end, console_obj_for_rich)


def launch_and_monitor_for_tuning_cli():
    global kcpp_process_obj, kcpp_monitor_thread, monitor_start_time # Uses these globals for process mgmt
    global current_tuning_attempt_level, current_tuning_model_analysis_local, current_tuning_model_path_local # Uses these tuning globals
    global current_tuning_session_base_args, last_successful_monitored_run_details_cli
    global user_requested_stop_monitoring_cli # This flag is global for the monitor thread to see

    # --- Local variables for this specific monitoring attempt ---
    local_vram_at_decision_for_db: Optional[float] = None
    local_last_approx_vram_used_kcpp_mb: Optional[float] = None
    local_level_of_last_monitored_run = current_tuning_attempt_level # Capture current level for this run
    local_last_proposed_command_list_for_db: List[str] = []
    # --- End of local variables ---

    if kcpp_process_obj and kcpp_process_obj.poll() is None:
        print_warning("A KoboldCpp process is already being monitored. Please stop it first or wait.")
        return "continue_tuning" # Or some other appropriate return to stay in tuning menu

    print_info(f"Tuning: Launching & Monitoring for OT Level {local_level_of_last_monitored_run}")
    user_requested_stop_monitoring_cli = False 
    kcpp_success_event.clear(); kcpp_oom_event.clear(); kcpp_output_lines_shared.clear()

    ot_string_for_launch = tensortune_core.generate_overridetensors(current_tuning_model_analysis_local, local_level_of_last_monitored_run)
    args_for_kcpp_run_list = tensortune_core.build_command(
        current_tuning_model_path_local, ot_string_for_launch,
        current_tuning_model_analysis_local, current_tuning_session_base_args,
        current_attempt_level_for_tuning=local_level_of_last_monitored_run
    )
    local_last_proposed_command_list_for_db = tensortune_core.get_command_to_run(KOBOLDCPP_EXECUTABLE, args_for_kcpp_run_list)
    
    _, _, _, gpu_info_rich_before_launch = tensortune_core.get_available_vram_mb(CONFIG)
    local_vram_at_decision_for_db = gpu_info_rich_before_launch.get("free_mb", 0.0) 

    kcpp_process_obj, launch_error_msg = tensortune_core.launch_process( local_last_proposed_command_list_for_db, capture_output=True, new_console=False, use_text_mode=False)

    if launch_error_msg or not kcpp_process_obj:
        print_error(f"Failed to launch KCPP for monitoring: {launch_error_msg or 'Unknown error'}")
        tensortune_core.save_config_to_db(
            DB_FILE, current_tuning_model_path_local, current_tuning_model_analysis_local,
            local_vram_at_decision_for_db, local_last_proposed_command_list_for_db,
            local_level_of_last_monitored_run, "LAUNCH_FAILED_SETUP_CLI", None
        )
        return "continue_tuning" # Stay in tuning menu

    print_info(f"KoboldCpp process started (PID: {kcpp_process_obj.pid}). Monitoring output...")
    effective_args_for_port_check = get_effective_session_args(current_tuning_model_path_local, current_tuning_session_base_args)
    target_port_str_for_success = effective_args_for_port_check.get("--port", "5000")
    
    final_outcome_key_from_monitor = "UNKNOWN_EXIT_CLI" 
    monitor_thread_args = ( kcpp_process_obj, kcpp_success_event, kcpp_oom_event, kcpp_output_lines_shared, KOBOLD_SUCCESS_PATTERN, OOM_ERROR_KEYWORDS, target_port_str_for_success, console if dependencies['rich']['module'] else None)

    # --- Monitoring loop (Rich or plain) as before ---
    if dependencies['rich']['module']:
        with Progress(...) as progress_live_display: # Your existing Rich progress
            # ... (existing Rich monitoring loop) ...
            # Ensure it sets final_outcome_key_from_monitor
            loading_task_id = progress_live_display.add_task("KCPP Loading...", total=float(LOADING_TIMEOUT_SECONDS))
            kcpp_monitor_thread = threading.Thread(target=monitor_kcpp_output_thread_target_cli, args=monitor_thread_args, daemon=True)
            kcpp_monitor_thread.start()
            monitor_start_time = time.monotonic()
            try:
                while True:
                    elapsed_monitor_time = time.monotonic() - monitor_start_time
                    if not progress_live_display.finished: progress_live_display.update(loading_task_id, completed=min(elapsed_monitor_time, float(LOADING_TIMEOUT_SECONDS)))
                    if user_requested_stop_monitoring_cli: final_outcome_key_from_monitor = "USER_STOPPED_MONITORING_CLI"; break
                    process_has_exited = kcpp_process_obj.poll() is not None
                    if kcpp_success_event.is_set(): final_outcome_key_from_monitor = "SUCCESS_LOAD_DETECTED_CLI"; break
                    if kcpp_oom_event.is_set(): final_outcome_key_from_monitor = "OOM_CRASH_DETECTED_CLI"; break
                    if process_has_exited and not kcpp_success_event.is_set() and not kcpp_oom_event.is_set(): final_outcome_key_from_monitor = "PREMATURE_EXIT_CLI"; break
                    if elapsed_monitor_time > float(LOADING_TIMEOUT_SECONDS): final_outcome_key_from_monitor = "TIMEOUT_NO_SIGNAL_CLI"; break
                    time.sleep(0.25)
            except KeyboardInterrupt: user_requested_stop_monitoring_cli = True; final_outcome_key_from_monitor = "USER_STOPPED_MONITORING_CLI"; print_warning("\nMonitoring interrupted by user."); raise # Re-raise to be caught by main try-except
    else: # No Rich
        # ... (existing plain monitoring loop) ...
        # Ensure it sets final_outcome_key_from_monitor
        kcpp_monitor_thread = threading.Thread(target=monitor_kcpp_output_thread_target_cli, args=monitor_thread_args, daemon=True)
        kcpp_monitor_thread.start()
        monitor_start_time = time.monotonic()
        try:
            spinner_chars = "|/-\\"
            idx = 0
            while True:
                if user_requested_stop_monitoring_cli: final_outcome_key_from_monitor = "USER_STOPPED_MONITORING_CLI"; break
                process_has_exited = kcpp_process_obj.poll() is not None
                if kcpp_success_event.is_set(): final_outcome_key_from_monitor = "SUCCESS_LOAD_DETECTED_CLI"; break
                if kcpp_oom_event.is_set(): final_outcome_key_from_monitor = "OOM_CRASH_DETECTED_CLI"; break
                if process_has_exited and not kcpp_success_event.is_set() and not kcpp_oom_event.is_set(): final_outcome_key_from_monitor = "PREMATURE_EXIT_CLI"; break
                elapsed_monitor_time = time.monotonic() - monitor_start_time
                if elapsed_monitor_time > float(LOADING_TIMEOUT_SECONDS): final_outcome_key_from_monitor = "TIMEOUT_NO_SIGNAL_CLI"; break
                time.sleep(0.25)
                idx = (idx + 1) % len(spinner_chars)
                sys.stdout.write(f"\rKCPP Loading... {spinner_chars[idx]} ({elapsed_monitor_time:.1f}s / {LOADING_TIMEOUT_SECONDS}s)   ")
                sys.stdout.flush()
        except KeyboardInterrupt: user_requested_stop_monitoring_cli = True; final_outcome_key_from_monitor = "USER_STOPPED_MONITORING_CLI"; print_warning("\nMonitoring interrupted by user."); raise
        finally:
            if not dependencies['rich']['module']: sys.stdout.write("\r" + " " * 80 + "\r"); sys.stdout.flush()
    # --- End of Monitoring loop ---

    print_info(f"Monitoring completed. Initial Outcome: {final_outcome_key_from_monitor}")
    if final_outcome_key_from_monitor in ["TIMEOUT_NO_SIGNAL_CLI", "OOM_CRASH_DETECTED_CLI", "PREMATURE_EXIT_CLI", "USER_STOPPED_MONITORING_CLI"] or "OOM" in final_outcome_key_from_monitor.upper() or "CRASH" in final_outcome_key_from_monitor.upper():
        if kcpp_process_obj and kcpp_process_obj.poll() is None:
            print_info("Terminating KCPP process due to unfavorable outcome or user stop...")
            tensortune_core.kill_process(kcpp_process_obj.pid, force=True)
        kcpp_process_obj = None # Clear the global process object

    db_outcome_to_save_str = final_outcome_key_from_monitor
    if final_outcome_key_from_monitor == "SUCCESS_LOAD_DETECTED_CLI":
        print_info(f"API detected. Waiting {VRAM_STABILIZATION_WAIT_S}s for VRAM to stabilize...")
        time.sleep(max(2.0, float(VRAM_STABILIZATION_WAIT_S)))
        
        _, _, _, gpu_info_rich_after_load = tensortune_core.get_available_vram_mb(CONFIG)
        budgeted_free_vram_after_load = gpu_info_rich_after_load.get("free_mb_budgeted", 0.0)
        actual_hw_free_vram_after_load = gpu_info_rich_after_load.get("free_mb", 0.0)
        actual_hw_total_vram_after_load = gpu_info_rich_after_load.get("total_mb", 0.0)

        if actual_hw_total_vram_after_load > 0 and local_vram_at_decision_for_db is not None and actual_hw_free_vram_after_load is not None :
            vram_used_by_kcpp_actual_hw = local_vram_at_decision_for_db - actual_hw_free_vram_after_load
            local_last_approx_vram_used_kcpp_mb = max(0, min(vram_used_by_kcpp_actual_hw, actual_hw_total_vram_after_load)) # Store locally
            print_info(f"Budgeted VRAM after load: {budgeted_free_vram_after_load:.0f}MB free. Approx Actual KCPP VRAM usage: {local_last_approx_vram_used_kcpp_mb:.0f}MB")
            
            if budgeted_free_vram_after_load < MIN_VRAM_FREE_AFTER_LOAD_MB:
                print_warning(f"Budgeted VRAM tight! {budgeted_free_vram_after_load:.0f}MB < {MIN_VRAM_FREE_AFTER_LOAD_MB}MB target.")
                db_outcome_to_save_str = "SUCCESS_LOAD_VRAM_TIGHT_CLI"
            else:
                print_success("Budgeted VRAM usage OK.")
                db_outcome_to_save_str = "SUCCESS_LOAD_VRAM_OK_CLI"
            
            if gpu_info_rich_after_load.get("override_active", False) and local_last_approx_vram_used_kcpp_mb is not None and local_last_approx_vram_used_kcpp_mb > gpu_info_rich_after_load.get("total_mb_budgeted", 0):
                print_warning(f"NOTE: Actual KCPP VRAM usage ({local_last_approx_vram_used_kcpp_mb:.0f}MB) exceeded manual VRAM budget ({gpu_info_rich_after_load.get('total_mb_budgeted', 0):.0f}MB).")
            last_successful_monitored_run_details_cli = { "level": local_level_of_last_monitored_run, "outcome": db_outcome_to_save_str, "vram_used_mb": f"{local_last_approx_vram_used_kcpp_mb:.0f}" if local_last_approx_vram_used_kcpp_mb is not None else "N/A" }
        else:
            db_outcome_to_save_str = "SUCCESS_LOAD_NO_VRAM_CHECK_CLI"
            last_successful_monitored_run_details_cli = { "level": local_level_of_last_monitored_run, "outcome": db_outcome_to_save_str, "vram_used_mb": "N/A"}

    tensortune_core.save_config_to_db(
        DB_FILE, current_tuning_model_path_local, current_tuning_model_analysis_local,
        local_vram_at_decision_for_db, local_last_proposed_command_list_for_db,
        local_level_of_last_monitored_run, db_outcome_to_save_str,
        local_last_approx_vram_used_kcpp_mb # Pass the locally calculated value
    )
    # Pass the local VRAM info and command list to the post-monitoring choices
    return handle_post_monitoring_choices_cli(
        db_outcome_to_save_str,
        kcpp_process_obj, # Pass the current kcpp_process_obj (might be None if it exited)
        local_vram_at_decision_for_db,
        local_last_approx_vram_used_kcpp_mb,
        local_level_of_last_monitored_run,
        local_last_proposed_command_list_for_db
    )


def handle_post_monitoring_choices_cli(
    outcome_from_monitor: str,
    monitored_kcpp_instance: Optional[subprocess.Popen],
    # ADDED PARAMETERS from the monitoring session:
    vram_at_decision_from_monitor: Optional[float],
    approx_vram_used_from_monitor: Optional[float],
    level_from_monitor: int,
    command_list_from_monitor: List[str]
):
    global current_tuning_attempt_level, tuning_in_progress, kcpp_process_obj # kcpp_process_obj is the monitored one
    global last_launched_process_info # For non-monitored KCPP instance
    # level_of_last_monitored_run is now level_from_monitor

    kcpp_process_obj = monitored_kcpp_instance # Keep track of the process passed from monitor
    choices_dict = {}; default_choice_key = ""
    print_title("Post-Monitoring Options")
    print_info(f"Outcome of monitored launch (Level {level_from_monitor}): {outcome_from_monitor}")
    if approx_vram_used_from_monitor is not None:
        print_info(f"Approx. Actual KCPP VRAM Used by monitored run: {approx_vram_used_from_monitor:.0f} MB")
    
    kcpp_is_still_running = kcpp_process_obj and kcpp_process_obj.poll() is None

    # --- Logic for choices_dict and default_choice_key as before ---
    if "SUCCESS_LOAD_VRAM_OK" in outcome_from_monitor:
        print_success("KCPP loaded successfully (Budgeted VRAM OK).")
        choices_dict = { "u": "✅ Accept & Use this KCPP instance", "s": "💾 Save as Good, Auto-Adjust for More GPU & Continue Tuning", "g": "⚙️ Manually Try More GPU (This Session) & Continue Tuning", "c": "⚙️ Manually Try More CPU (This Session) & Continue Tuning", "q": "↩️ Save Outcome & Return to Tuning Menu (Manual Adjust)" }
        default_choice_key = "u" if kcpp_is_still_running else "s"
    elif "SUCCESS_LOAD_VRAM_TIGHT" in outcome_from_monitor:
        print_warning("KCPP loaded, but Budgeted VRAM is tight!")
        choices_dict = { "a": "⚠️ Auto-Adjust for More CPU & Continue Tuning", "l": "🚀 Launch This Config Anyway (Risky)", "q": "↩️ Save Outcome & Return to Tuning Menu (Manual Adjust)" }
        default_choice_key = "a"
    elif "OOM" in outcome_from_monitor or "CRASH" in outcome_from_monitor or "PREMATURE_EXIT" in outcome_from_monitor or "USER_STOPPED" in outcome_from_monitor:
        print_error("KCPP failed to load properly (OOM/Crash/Premature Exit/User Stop).")
        choices_dict = { "c": "⚙️ Auto-Adjust for More CPU & Continue Tuning", "q": "↩️ Save Outcome & Return to Tuning Menu (Manual Adjust)"}
        default_choice_key = "c"
    elif "TIMEOUT" in outcome_from_monitor:
        print_warning("KCPP launch timed out (no success/OOM signal).")
        choices_dict = { "c": "⚙️ Auto-Adjust for More CPU (Assume OOM) & Continue Tuning", "q": "↩️ Save Outcome & Return to Tuning Menu (Manual Adjust)"}
        default_choice_key = "c"
    else: 
        if kcpp_is_still_running:
            print_info("KCPP loaded (status: unknown VRAM or generic success).")
            choices_dict = {"u": "✅ Keep KCPP Running for Use", "q": "↩️ Stop KCPP, Save Outcome & Return to Tuning Menu (Manual Adjust)"}
            default_choice_key = "u"
        else: 
            print_warning("KCPP status unclear or it has already exited.")
            choices_dict = {"q": "↩️ Save Outcome & Return to Tuning Menu (Manual Adjust)"}
            default_choice_key = "q"
    # --- End of choice dict logic ---

    for key, desc in choices_dict.items(): print(f"  ({key.upper()}) {desc}")
    user_action_choice = prompt("Your choice?", choices=list(choices_dict.keys()), default=default_choice_key).lower()
    
    db_outcome_suffix_for_action = "_CLI" 
    should_stop_monitored_kcpp = True 
    
    if user_action_choice in ['u', 'l'] and kcpp_is_still_running:
        should_stop_monitored_kcpp = False 

    if kcpp_is_still_running and should_stop_monitored_kcpp:
        print_info(f"Stopping monitored KCPP instance (PID: {kcpp_process_obj.pid})...")
        tensortune_core.kill_process(kcpp_process_obj.pid, force=True)
        kcpp_process_obj = None # Clear the monitored process

    # Determine the suffix for the DB outcome based on user's choice
    if user_action_choice == 'u': db_outcome_suffix_for_action = "_USER_ACCEPTED_TUNED_CLI"
    elif user_action_choice == 'l': db_outcome_suffix_for_action = "_USER_LAUNCHED_RISKY_CLI"
    elif user_action_choice == 's': db_outcome_suffix_for_action = "_USER_SAVED_GOOD_GPU_CLI"
    elif user_action_choice == 'g': db_outcome_suffix_for_action = "_USER_WANTS_MORE_GPU_CLI"
    elif user_action_choice == 'a': db_outcome_suffix_for_action = "_USER_AUTO_ADJUST_CPU_CLI"
    elif user_action_choice == 'c': 
        if "FAIL" in outcome_from_monitor.upper() or "OOM" in outcome_from_monitor.upper() or "TIMEOUT" in outcome_from_monitor.upper() or "USER_STOPPED" in outcome_from_monitor.upper():
            db_outcome_suffix_for_action = "_USER_TRIED_CPU_AFTER_FAIL_CLI"
        else:
            db_outcome_suffix_for_action = "_USER_WANTS_MORE_CPU_CLI"
    elif user_action_choice == 'q': db_outcome_suffix_for_action = "_USER_RETURNED_MENU_CLI"
    
    # Update the DB record for the monitored run with the user's decision outcome
    # The initial save was done in launch_and_monitor_for_tuning_cli, here we might update it
    # if the outcome reflects a user choice.
    final_outcome_for_db_update = outcome_from_monitor + db_outcome_suffix_for_action
    tensortune_core.save_config_to_db(
        DB_FILE, current_tuning_model_path_local, current_tuning_model_analysis_local,
        vram_at_decision_from_monitor, # VRAM before this specific monitored run
        command_list_from_monitor,    # Command for this specific monitored run
        level_from_monitor,           # Level of this specific monitored run
        final_outcome_for_db_update,
        approx_vram_used_from_monitor # VRAM used by this specific monitored run
    )

    # Perform the chosen action
    if user_action_choice == 'u': # Accept & Use
        if kcpp_is_still_running and kcpp_process_obj: # If kept running
            print_info("Keeping current KoboldCpp instance running for use.")
            last_launched_process_info["pid"] = kcpp_process_obj.pid 
            last_launched_process_info["process_obj"] = kcpp_process_obj # Transfer to main launcher tracking
            last_launched_process_info["command_list"] = command_list_from_monitor
            kcpp_process_obj = None # Clear from monitored slot
            
            effective_args_for_webui = get_effective_session_args(current_tuning_model_path_local, current_tuning_session_base_args)
            port_for_webui = effective_args_for_webui.get("--port", "5000")
            if AUTO_OPEN_WEBUI: webbrowser.open(f"http://localhost:{port_for_webui}")
            
            tuning_in_progress = False 
            session_control_result = kcpp_control_loop_cli(port_for_webui, is_monitored_instance_being_controlled=False) # Now it's a "last_launched"
            return session_control_result 
        else: # Monitored process was stopped or failed, so relaunch
            print_warning("KCPP is not running. Re-launching the accepted configuration for use...")
            launched_proc = launch_kobold_for_use_cli(
                command_list_from_monitor, # Use the command from the successful monitor
                final_outcome_for_db_update, # Log this re-launch with the accept outcome
                level_from_monitor,
                vram_at_launch_decision_param=vram_at_decision_from_monitor, # Use VRAM from before monitored run
                approx_vram_used_mb_for_db_param=approx_vram_used_from_monitor
            )
            if launched_proc:
                effective_args_relaunch = get_effective_session_args(current_tuning_model_path_local, current_tuning_session_base_args)
                port_relaunch = effective_args_relaunch.get("--port", "5000")
                session_ctrl_res = kcpp_control_loop_cli(port_relaunch)
                tuning_in_progress = False; return session_ctrl_res
            else:
                print_error("Re-launch failed. Returning to tuning menu.")
                # DB already updated by launch_kobold_for_use_cli if launch fails
                return "continue_tuning"

    elif user_action_choice == 'l': # Launch Risky
        print_info("Attempting to launch risky configuration for use...")
        # The monitored KCPP instance (if any) would have been stopped already unless this logic changes
        if kcpp_process_obj and kcpp_process_obj.poll() is None: # Ensure it's stopped before new launch
            tensortune_core.kill_process(kcpp_process_obj.pid, force=True); kcpp_process_obj = None
        
        launched_proc_risky = launch_kobold_for_use_cli(
            command_list_from_monitor,
            final_outcome_for_db_update, # Outcome already reflects risky choice
            level_from_monitor,
            vram_at_launch_decision_param=vram_at_decision_from_monitor,
            approx_vram_used_mb_for_db_param=approx_vram_used_from_monitor
        )
        if launched_proc_risky:
            effective_args_relaunch_risky = get_effective_session_args(current_tuning_model_path_local, current_tuning_session_base_args)
            port_relaunch_risky = effective_args_relaunch_risky.get("--port", "5000")
            session_ctrl_res_relaunch = kcpp_control_loop_cli(port_relaunch_risky)
            tuning_in_progress = False; return session_ctrl_res_relaunch
        else:
            print_error("Risky re-launch failed. Returning to tuning menu.")
            return "continue_tuning"

    # Adjust OT level for continued tuning based on other choices
    if user_action_choice == 's': # Save as Good, More GPU
        if current_tuning_attempt_level > current_tuning_min_level: current_tuning_attempt_level -= 1
        else: print_warning("Already at Max GPU, cannot go further.")
    elif user_action_choice == 'g': # Manually More GPU
        if current_tuning_attempt_level > current_tuning_min_level: current_tuning_attempt_level -= 1
        else: print_warning("Already at Max GPU, cannot go further.")
    elif user_action_choice == 'c' or user_action_choice == 'a': # More CPU (Auto or Manual after fail)
        if current_tuning_attempt_level < current_tuning_max_level: current_tuning_attempt_level += 1
        else: print_warning("Already at Max CPU, cannot go further.")
    
    # 'q' (Return to tuning menu) or other continue tuning actions
    return "continue_tuning"


def launch_kobold_for_use_cli(
    command_list_to_run: List[str],
    db_outcome_on_success: str,
    level_for_db_record: Optional[int] = None,
    # ADDED PARAMETERS:
    vram_at_launch_decision_param: Optional[float] = None,
    approx_vram_used_mb_for_db_param: Optional[float] = None
):
    global last_launched_process_info, gguf_file_global, current_model_analysis_global # CLI global states
    global current_tuning_model_path_local, current_tuning_model_analysis_local, tuning_in_progress # Tuning states

    # Stop any previously launched (non-monitored) KCPP process
    if last_launched_process_info.get("process_obj") and last_launched_process_info["process_obj"].poll() is None:
        print_info(f"Stopping previously launched KCPP (PID: {last_launched_process_info['pid']})...")
        tensortune_core.kill_process(last_launched_process_info["pid"])
    last_launched_process_info = {"pid": None, "process_obj": None, "command_list": []}

    print_info(f"Launching KoboldCpp for use...")
    
    # Determine VRAM at decision: Use passed param, or get fresh if not provided
    current_vram_at_this_launch_decision = vram_at_launch_decision_param
    if current_vram_at_this_launch_decision is None:
        _, _, _, gpu_info_rich_direct_launch = tensortune_core.get_available_vram_mb(CONFIG)
        current_vram_at_this_launch_decision = gpu_info_rich_direct_launch.get("free_mb", 0.0)
    
    # Determine model path and analysis based on whether we are in a tuning session or main menu
    model_p_for_log = current_tuning_model_path_local if tuning_in_progress and current_tuning_model_path_local else gguf_file_global
    model_a_for_log = current_tuning_model_analysis_local if tuning_in_progress and current_tuning_model_analysis_local.get('filepath') else current_model_analysis_global
    
    final_level_for_db_log = level_for_db_record
    if final_level_for_db_log is None: # Default if not provided (e.g. direct launch from main menu)
        final_level_for_db_log = current_tuning_attempt_level if tuning_in_progress else 0

    # Use the passed parameter for approx_vram_used_mb_for_db
    vram_used_for_this_db_entry = approx_vram_used_mb_for_db_param

    # Ensure model_p_for_log and model_a_for_log are valid before DB save
    if not model_p_for_log or not model_a_for_log.get('filepath'):
        print_error("CRITICAL: Cannot save to DB - model path or analysis is missing for the launch record.")
    else:
        tensortune_core.save_config_to_db(
            DB_FILE, model_p_for_log, model_a_for_log,
            current_vram_at_this_launch_decision,
            command_list_to_run, final_level_for_db_log,
            db_outcome_on_success,
            vram_used_for_this_db_entry
        )
    
    launched_kcpp_process, launch_err_msg = tensortune_core.launch_process(command_list_to_run, capture_output=False, new_console=True)

    if launch_err_msg or not launched_kcpp_process:
        print_error(f"Failed to launch KoboldCPP: {launch_err_msg or 'Unknown error'}")
        if model_p_for_log and model_a_for_log.get('filepath'): # Only log if we have model info
            tensortune_core.save_config_to_db(
                DB_FILE, model_p_for_log, model_a_for_log,
                current_vram_at_this_launch_decision, command_list_to_run,
                final_level_for_db_log, "LAUNCH_FOR_USE_FAILED_CLI",
                vram_used_for_this_db_entry
            )
        return None
    else:
        print_success(f"KoboldCpp launched in new console (PID: {launched_kcpp_process.pid}).")
        last_launched_process_info["pid"] = launched_kcpp_process.pid
        last_launched_process_info["process_obj"] = launched_kcpp_process
        last_launched_process_info["command_list"] = command_list_to_run
        
        if AUTO_OPEN_WEBUI:
            # Determine port from effective arguments
            args_dict_from_cmd = tensortune_core.args_list_to_dict(
                command_list_to_run[2:] if command_list_to_run[0].lower() == sys.executable.lower() and len(command_list_to_run) > 1 and command_list_to_run[1].lower().endswith((".py", ".pyw")) else command_list_to_run[1:]
            )
            # Get base args considering global, model-specific (if any for model_p_for_log)
            base_args_for_final_launch = get_effective_session_args(model_p_for_log, {}) 
            effective_launch_args_for_port = base_args_for_final_launch.copy()
            effective_launch_args_for_port.update(args_dict_from_cmd) # Command line args take precedence

            port_to_open_webui = effective_launch_args_for_port.get("--port", "5000")
            print_info(f"Attempting to open Web UI at http://localhost:{port_to_open_webui} in a few seconds...")
            threading.Timer(3.0, lambda: webbrowser.open(f"http://localhost:{port_to_open_webui}")).start()
        return launched_kcpp_process


def kcpp_control_loop_cli(port_to_use_for_webui: str, is_monitored_instance_being_controlled: bool = False) -> str:
    global last_launched_process_info, kcpp_process_obj

    process_object_to_control = kcpp_process_obj if is_monitored_instance_being_controlled else last_launched_process_info.get("process_obj")
    pid_to_control_val = (kcpp_process_obj.pid if kcpp_process_obj else None) if is_monitored_instance_being_controlled else last_launched_process_info.get("pid")

    while True:
        if process_object_to_control and process_object_to_control.poll() is not None:
            print_warning(f"KoboldCpp process (PID: {pid_to_control_val}) seems to have exited on its own.")
            if is_monitored_instance_being_controlled: kcpp_process_obj = None
            else: last_launched_process_info = {"pid": None, "process_obj": None, "command_list": []}
            return "new_gguf" 

        print_title("KCPP Control Options")
        active_pid_display_str = f" (PID: {pid_to_control_val})" if pid_to_control_val else " (No active PID)"
        print(f"  (S)top KCPP{active_pid_display_str} & Select New Model")
        print(f"  (Q)uit Launcher (Stops KCPP{active_pid_display_str} & Exits Script)")
        print(f"  (E)xit Launcher (Leave KCPP{active_pid_display_str} Running & Exits Script)")
        print(f"  (W)Open WebUI (http://localhost:{port_to_use_for_webui})")
        
        control_choice = prompt("KCPP Control", choices=['s', 'q', 'e', 'w'], default='s').lower().strip()

        if control_choice == 's':
            if process_object_to_control and pid_to_control_val:
                print_info(f"Stopping KCPP (PID: {pid_to_control_val})...")
                tensortune_core.kill_process(pid_to_control_val)
            if is_monitored_instance_being_controlled: kcpp_process_obj = None
            else: last_launched_process_info = {"pid": None, "process_obj": None, "command_list": []}
            return "new_gguf"
        elif control_choice == 'q':
            if process_object_to_control and pid_to_control_val:
                print_info(f"Stopping KCPP (PID: {pid_to_control_val}) and quitting launcher...")
                tensortune_core.kill_process(pid_to_control_val)
            return "quit_script"
        elif control_choice == 'e':
            print_info(f"Exiting launcher. KCPP{active_pid_display_str} will be left running.")
            return "quit_script_leave_running"
        elif control_choice == 'w':
            print_info(f"Opening Web UI at http://localhost:{port_to_use_for_webui}")
            webbrowser.open(f"http://localhost:{port_to_use_for_webui}")
        else: print_error("Invalid control choice.")


def run_model_tuning_session_cli() -> str:
    global tuning_in_progress, current_tuning_attempt_level, current_tuning_min_level, current_tuning_max_level
    global current_tuning_session_base_args, current_tuning_model_path_local, current_tuning_model_analysis_local
    global gguf_file_global, current_model_analysis_global, level_of_last_monitored_run, last_successful_monitored_run_details_cli
    global vram_at_decision_for_db, last_approx_vram_used_kcpp_mb 

    if not gguf_file_global or not current_model_analysis_global.get('filepath'):
        print_error("No model selected or analyzed. Please select a model first.")
        return "new_gguf"

    tuning_in_progress = True
    current_tuning_model_path_local = gguf_file_global
    current_tuning_model_analysis_local = current_model_analysis_global.copy()
    current_tuning_session_base_args = get_effective_session_args(current_tuning_model_path_local, {}) 
    last_successful_monitored_run_details_cli = None
    vram_at_decision_for_db = None 
    last_approx_vram_used_kcpp_mb = None 

    print_title(f"Starting Auto-Tuning Session for: {os.path.basename(current_tuning_model_path_local)}")
    print_info(f"Analysis: Size ~{current_tuning_model_analysis_local.get('size_b', 'N/A')}B, "
               f"Quant ~{current_tuning_model_analysis_local.get('quant', 'N/A')}, "
               f"MoE: {'Yes' if current_tuning_model_analysis_local.get('is_moe') else 'No'}, "
               f"Layers: {current_tuning_model_analysis_local.get('num_layers', 'N/A')}, "
               f"Est. Full VRAM: {current_tuning_model_analysis_local.get('estimated_vram_gb_full_gpu', 'N/A')}GB")

    _, _, _, current_gpu_full_info = tensortune_core.get_available_vram_mb(CONFIG)
    current_budgeted_free_vram_mb = current_gpu_full_info.get("free_mb_budgeted", 0.0)
    current_actual_hw_free_vram_mb = current_gpu_full_info.get("free_mb", 0.0) 

    is_moe = current_tuning_model_analysis_local.get('is_moe', False)
    estimated_vram_needed_gb = current_tuning_model_analysis_local.get('estimated_vram_gb_full_gpu', 0)
    estimated_vram_needed_mb = float(estimated_vram_needed_gb) * 1024 if isinstance(estimated_vram_needed_gb, (int,float)) else 0.0

    if is_moe: current_tuning_min_level, current_tuning_max_level, initial_heuristic_level = -25, 10, -10
    else:
        current_tuning_min_level, current_tuning_max_level = -17, 9
        size_b_val = current_tuning_model_analysis_local.get('size_b', 0)
        size_b_float = float(size_b_val) if isinstance(size_b_val, (int,float)) else 0.0
        if size_b_float >= 30: initial_heuristic_level = -3
        elif size_b_float >= 20: initial_heuristic_level = -5
        else: initial_heuristic_level = -7
    
    effective_vram_budget_for_heuristic_mb = current_gpu_full_info.get("total_mb_budgeted", 0.0) - VRAM_SAFETY_BUFFER_MB - MIN_VRAM_FREE_AFTER_LOAD_MB

    if estimated_vram_needed_mb > 0 and current_budgeted_free_vram_mb > 0 :
        if estimated_vram_needed_mb > effective_vram_budget_for_heuristic_mb * 1.1:
            initial_heuristic_level = max(initial_heuristic_level, -3 if not is_moe else -6)
            print_info(f"Heuristic: Est. VRAM ({estimated_vram_needed_mb:.0f}MB) > budget ({effective_vram_budget_for_heuristic_mb:.0f}MB). OT -> CPU.")
        elif estimated_vram_needed_mb < effective_vram_budget_for_heuristic_mb * 0.7:
            initial_heuristic_level = min(initial_heuristic_level, -12 if not is_moe else -18)
            print_info(f"Heuristic: Ample VRAM budget. OT -> GPU.")

    best_historical_config = tensortune_core.find_best_historical_config(DB_FILE, current_tuning_model_analysis_local, current_actual_hw_free_vram_mb, CONFIG)

    if best_historical_config and "attempt_level" in best_historical_config:
        print_info(f"Found historical config. Level: {best_historical_config['attempt_level']}, Outcome: {best_historical_config['outcome']}")
        hist_lvl, hist_outcome_str = best_historical_config['attempt_level'], best_historical_config.get('outcome', "")
        approx_hist_vram_used = best_historical_config.get('approx_vram_used_kcpp_mb') 

        if approx_hist_vram_used is not None and (float(approx_hist_vram_used) + VRAM_SAFETY_BUFFER_MB < current_actual_hw_free_vram_mb):
            initial_heuristic_level = max(current_tuning_min_level, hist_lvl -1 if hist_lvl > current_tuning_min_level else hist_lvl)
            print_info(f"Historical success used {approx_hist_vram_used:.0f}MB (actual) fits current actual VRAM ({current_actual_hw_free_vram_mb:.0f}MB). Starting near: {initial_heuristic_level}")
        elif hist_outcome_str.startswith("SUCCESS_LOAD_VRAM_OK") or hist_outcome_str.startswith("SUCCESS_USER_CONFIRMED") or hist_outcome_str.endswith("_USER_SAVED_GOOD_GPU_CLI"):
            initial_heuristic_level = max(current_tuning_min_level, hist_lvl - 1 if hist_lvl > current_tuning_min_level else hist_lvl)
        elif hist_outcome_str.endswith("_USER_AUTO_ADJUST_CPU_CLI") or hist_outcome_str.endswith("_USER_TRIED_CPU_AFTER_FAIL_CLI") or "OOM" in hist_outcome_str.upper() or "TIGHT" in hist_outcome_str.upper():
             initial_heuristic_level = min(current_tuning_max_level, hist_lvl + 1 if hist_lvl < current_tuning_max_level else hist_lvl)
        else: initial_heuristic_level = hist_lvl
        
        remembered_args_list_from_db = best_historical_config.get("args_list", [])
        if remembered_args_list_from_db:
            remembered_args_dict_parsed = tensortune_core.args_list_to_dict(remembered_args_list_from_db)
            remembered_args_dict_parsed.pop("--model", None); remembered_args_dict_parsed.pop("--overridetensors", None)
            current_tuning_session_base_args.update(remembered_args_dict_parsed) 
            print_info(f"Applied remembered arguments to session base. OT Level target: {initial_heuristic_level}")
    else:
        print_info(f"No suitable historical config found. Starting with heuristic OT Level: {initial_heuristic_level}")

    current_tuning_attempt_level = max(current_tuning_min_level, min(initial_heuristic_level, current_tuning_max_level))
    level_of_last_monitored_run = current_tuning_attempt_level

    while tuning_in_progress:
        print("\n" + "=" * 70)
        current_tuning_attempt_level = max(current_tuning_min_level, min(current_tuning_attempt_level, current_tuning_max_level))
        ot_string_generated = tensortune_core.generate_overridetensors(current_tuning_model_analysis_local, current_tuning_attempt_level)
        strategy_description = tensortune_core.get_offload_description(current_tuning_model_analysis_local, current_tuning_attempt_level, ot_string_generated)
        gpu_layers_for_level = tensortune_core.get_gpu_layers_for_level(current_tuning_model_analysis_local, current_tuning_attempt_level)
        total_model_layers = current_tuning_model_analysis_local.get('num_layers', 32)

        if last_successful_monitored_run_details_cli:
            ls_level = last_successful_monitored_run_details_cli.get("level", "N/A")
            ls_outcome = last_successful_monitored_run_details_cli.get("outcome", "N/A")
            ls_vram = last_successful_monitored_run_details_cli.get("vram_used_mb", "N/A")
            print_info(f"Last Monitored Success: Level {ls_level}, Outcome: {ls_outcome}, Approx Actual KCPP VRAM Used: {ls_vram}MB")

        if dependencies['rich']['module']:
            strategy_table = Table(title="Current Tensor Offload Strategy")
            strategy_table.add_column("Setting", style="cyan"); strategy_table.add_column("Value", style="yellow", overflow="fold")
            strategy_table.add_row("Model Being Tuned", os.path.basename(current_tuning_model_path_local))
            strategy_table.add_row("OT Level", f"{current_tuning_attempt_level}")
            range_desc_str = f"{current_tuning_min_level}=MaxGPU ... {current_tuning_max_level}={'SuperMaxCPU' if current_tuning_model_analysis_local.get('is_moe') else 'MaxCPU'}"
            strategy_table.add_row("Level Range", range_desc_str); strategy_table.add_row("Strategy Desc", strategy_description)
            strategy_table.add_row("OT Regex", ot_string_generated or "None")
            strategy_table.add_row("Effective GPU Layers", f"{gpu_layers_for_level}/{total_model_layers}")
            console.print(strategy_table)
        else:
            print_info(f"Model: {os.path.basename(current_tuning_model_path_local)}")
            print(f"🛠️ OT Level: {current_tuning_attempt_level}\n   Range: {current_tuning_min_level}=MaxGPU to {current_tuning_max_level}={'SuperMaxCPU' if current_tuning_model_analysis_local.get('is_moe') else 'MaxCPU'}\n   Strategy: {strategy_description}\n   Regex: {(ot_string_generated or 'None')}\n   GPU Layers: {gpu_layers_for_level}/{total_model_layers}")

        args_for_kcpp_display_list = tensortune_core.build_command(current_tuning_model_path_local, ot_string_generated, current_tuning_model_analysis_local, current_tuning_session_base_args, current_attempt_level_for_tuning=current_tuning_attempt_level)
        display_full_command_list = tensortune_core.get_command_to_run(KOBOLDCPP_EXECUTABLE, args_for_kcpp_display_list)
        
        _, _, vram_info_message_str, current_gpu_rich_info_tuning = tensortune_core.get_available_vram_mb(CONFIG)
        if dependencies['rich']['module']: console.print(Panel(f"{vram_info_message_str}", title="Current GPU Info", style="green" if current_gpu_rich_info_tuning.get("success") else "red", expand=False))
        else: print(f"    GPU Status: {vram_info_message_str}")
        
        print_title("Proposed Command for This OT Level"); print_command(tensortune_core.format_command_for_display(display_full_command_list))
        
        menu_options_text = "(L)aunch & Monitor | (S)kip Tune & Launch Now | (G)PU More (↓Lvl) | (C)PU More (↑Lvl) | (E)dit Session Args | (P)ermanent Model Args | (H)istory (This Model) | (N)ew GGUF | (Q)uit Tuning"
        print_title("Tuning Actions"); print(menu_options_text)
        user_tuning_choice = prompt("Your choice", choices=['l','s','g','c','e', 'p', 'h','n','q'], default='l').lower().strip()

        if user_tuning_choice == 'l':
            post_monitoring_action_result = launch_and_monitor_for_tuning_cli()
            if post_monitoring_action_result == "quit_script_leave_running": tuning_in_progress = False; return "quit_script_leave_running"
            elif post_monitoring_action_result == "new_gguf": tuning_in_progress = False; return "new_gguf"
            elif post_monitoring_action_result == "quit_script": tuning_in_progress = False; return "quit_script"
        elif user_tuning_choice == 's':
            print_info("Skipping further tuning, launching current configuration directly...")
            _, _, _, gpu_info_direct_skip = tensortune_core.get_available_vram_mb(CONFIG)
            vram_at_decision_for_db = gpu_info_direct_skip.get("free_mb", 0.0) 
            last_approx_vram_used_kcpp_mb = None 

            launched_kcpp_proc = launch_kobold_for_use_cli(display_full_command_list, "SUCCESS_USER_DIRECT_LAUNCH_CLI", level_for_db_record=current_tuning_attempt_level, approx_vram_used_mb_for_db=None)
            if launched_kcpp_proc:
                effective_args_for_direct_launch = get_effective_session_args(current_tuning_model_path_local, current_tuning_session_base_args)
                port_for_direct_launch = effective_args_for_direct_launch.get("--port", "5000")
                session_control_outcome = kcpp_control_loop_cli(port_for_direct_launch)
                tuning_in_progress = False; return session_control_outcome
            else: print_error("Direct launch failed. Returning to tuning menu."); continue
        elif user_tuning_choice == 'g':
            if current_tuning_attempt_level > current_tuning_min_level: current_tuning_attempt_level -=1; print_success(f"Shifted towards GPU. New OT Level: {current_tuning_attempt_level}")
            else: print_warning(f"Already at Max GPU (Level {current_tuning_min_level}).")
        elif user_tuning_choice == 'c':
            if current_tuning_attempt_level < current_tuning_max_level: current_tuning_attempt_level +=1; print_success(f"Shifted towards CPU. New OT Level: {current_tuning_attempt_level}")
            else: print_warning(f"Already at Max CPU (Level {current_tuning_max_level}).")
        elif user_tuning_choice == 'e':
            updated_session_args_overrides, permanent_change_in_editor = edit_current_args_interactive_cli(current_tuning_model_path_local, current_tuning_session_base_args)
            if updated_session_args_overrides is not None: current_tuning_session_base_args = updated_session_args_overrides
            if permanent_change_in_editor: 
                 current_tuning_session_base_args = get_effective_session_args(current_tuning_model_path_local, {}) 
                 print_info("Permanent arguments were changed. Session overrides for this model reset, effective base updated.")
        elif user_tuning_choice == 'p':
            print_info("Opening argument editor for permanent model-specific settings...")
            _, permanent_save_made_here = edit_current_args_interactive_cli(current_tuning_model_path_local, current_tuning_session_base_args) 
            if permanent_save_made_here:
                 current_tuning_session_base_args = get_effective_session_args(current_tuning_model_path_local, {})
                 print_info("Permanent arguments changed. Session overrides for this model reset, effective base updated.")
        elif user_tuning_choice == 'h': view_db_history_cli(model_filepath_filter=current_tuning_model_path_local)
        elif user_tuning_choice == 'n': tuning_in_progress = False; return "new_gguf"
        elif user_tuning_choice == 'q': tuning_in_progress = False; return "new_gguf" 
        else: print_error("Invalid input.")
    
    tuning_in_progress = False
    return "new_gguf"


def main_cli():
    global CONFIG, gguf_file_global, current_model_analysis_global, last_launched_process_info
    # Note: vram_at_decision_for_db and last_approx_vram_used_kcpp_mb are NOT declared global here.
    # They will be handled as local variables within this function's scope for non-tuning launches,
    # or passed from tuning functions.

    # 1. Initialize Core and Load/Create Configuration
    core_init_data = tensortune_core.initialize_launcher()
    CONFIG = core_init_data["config"]
    _update_cli_globals_from_config()

    print_info(f"Using configuration file: {tensortune_core.CONFIG_FILE}")
    print_info(f"Using database file: {DB_FILE}")

    if not core_init_data["initialized"]:
        if not core_init_data["config_loaded"]: print_warning(f"Config issue: {core_init_data['config_message']}")
        if not core_init_data["db_success"]: print_warning(f"DB issue: {core_init_data['db_message']}")

    # 2. Handle First Run Prompts if necessary
    if not CONFIG.get("first_run_completed", False):
        if not handle_first_run_prompts_cli(CONFIG):
            print_error("Initial setup failed. Exiting."); sys.exit(1)
        _update_cli_globals_from_config()

    # 3. Print Initial Welcome and Core Information
    core_version_display = CONFIG.get('launcher_core_version', 'N/A')
    print_title(f"TensorTune v{LAUNCHER_CLI_VERSION} (Core: {core_version_display})")

    sys_info_data = core_init_data.get("system_info", {})
    print_info(f"OS: {sys_info_data.get('os_name','N/A')} {sys_info_data.get('os_version','N/A')} | Python: {sys_info_data.get('python_version','N/A').split()[0]}")

    # 4. Print GPU Information
    gpu_info_data_rich = core_init_data.get("gpu_info", {})
    gpu_message_to_display = gpu_info_data_rich.get('message', 'Could not detect GPU details.')
    if gpu_info_data_rich.get("success"):
        print_info(f"GPU Info: {gpu_message_to_display}")
    else:
        print_warning(f"GPU Info: {gpu_message_to_display}")
    if gpu_info_data_rich.get("override_active"):
        print_info(f"  VRAM Budget Override Active: Total {gpu_info_data_rich.get('total_mb_budgeted',0):.0f}MB")

    # 5. Conditionally Print Library Status Warnings
    show_lib_warnings_on_startup = True
    if CONFIG.get("first_run_completed", False) and CONFIG.get("suppress_optional_lib_warnings", False):
        show_lib_warnings_on_startup = False

    if show_lib_warnings_on_startup:
        print_info("Checking status of optional support libraries (can be suppressed in settings after first run):")
        optional_libs_to_check_status = {
            "PyADLX (AMD)": tensortune_core.pyadlx_load_error_reason,
            "WMI (Windows)": tensortune_core.wmi_load_error_reason,
            "PyZE (Intel)": tensortune_core.pyze_load_error_reason,
            "Appdirs": tensortune_core.appdirs_load_error_reason,
            "Metal (Apple)": tensortune_core.metal_load_error_reason,
        }
        any_optional_warnings_printed = False
        for lib_name, error_reason in optional_libs_to_check_status.items():
            if error_reason:
                if lib_name == "WMI (Windows)" and platform.system() != "win32": continue
                if lib_name == "Metal (Apple)" and platform.system() != "darwin": continue
                print_warning(f"  ! {lib_name}: {error_reason}")
                any_optional_warnings_printed = True
        
        critical_issues_found = False
        if tensortune_core.psutil_load_error_reason:
            print_error(f"  CRITICAL! Psutil: {tensortune_core.psutil_load_error_reason} (Impacts auto-threads, process management)")
            critical_issues_found = True
        if CONFIG.get("gpu_detection", {}).get("nvidia", True) and tensortune_core.pynvml_load_error_reason:
            print_error(f"  CRITICAL! PyNVML (NVIDIA): {tensortune_core.pynvml_load_error_reason} (Required for NVIDIA VRAM monitoring)")
            critical_issues_found = True

        if not any_optional_warnings_printed and not critical_issues_found and CONFIG.get("first_run_completed", False):
            print_success("  All checked optional and critical support libraries seem to be available or not applicable for your setup.")
            
    elif CONFIG.get("first_run_completed", False):
        print_info("Optional library status warnings are currently suppressed (configurable in settings).")
        critical_issues_found_suppressed_mode = False
        if tensortune_core.psutil_load_error_reason:
            print_error(f"  CRITICAL! Psutil: {tensortune_core.psutil_load_error_reason} (Impacts auto-threads, process management)")
            critical_issues_found_suppressed_mode = True
        if CONFIG.get("gpu_detection", {}).get("nvidia", True) and tensortune_core.pynvml_load_error_reason:
            print_error(f"  CRITICAL! PyNVML (NVIDIA): {tensortune_core.pynvml_load_error_reason} (Required for NVIDIA VRAM monitoring)")
            critical_issues_found_suppressed_mode = True
        if critical_issues_found_suppressed_mode:
            print_warning("  Note: Above CRITICAL library issues are always shown even if optional warnings are suppressed.")

    # 6. Print KoboldCpp Capabilities
    kcpp_caps = core_init_data.get("koboldcpp_capabilities", {})
    if "error" in kcpp_caps:
        print_warning(f"KCPP Caps Error: {kcpp_caps['error']}")
    else:
        print_info(f"KCPP Caps: CUDA:{kcpp_caps.get('cuda', False)}, ROCm:{kcpp_caps.get('rocm',False)}, FlashAttn:{kcpp_caps.get('flash_attn',False)}, OverrideTensors:{kcpp_caps.get('overridetensors',False)}")

    # 7. Validate KoboldCpp Executable Path
    if not _validate_and_update_kcpp_exe_path_in_config(CONFIG, KOBOLDCPP_EXECUTABLE):
        print_error(f"Initial KoboldCpp executable path '{KOBOLDCPP_EXECUTABLE}' is not valid.")
        if confirm("Would you like to correct the KoboldCpp executable path now in settings?", default=True):
            manage_launcher_settings_cli()
            _update_cli_globals_from_config()
            if not _validate_and_update_kcpp_exe_path_in_config(CONFIG, KOBOLDCPP_EXECUTABLE):
                print_error("FATAL: KoboldCpp path still not valid after attempting to fix. Exiting."); sys.exit(1)
            kcpp_caps_after_fix = tensortune_core.detect_koboldcpp_capabilities(CONFIG["koboldcpp_executable"])
            if _update_gpu_backend_flags_in_config_cli(kcpp_caps_after_fix):
                tensortune_core.save_launcher_config(CONFIG)
                _update_cli_globals_from_config()
                print_info("KCPP capabilities re-checked and GPU backend flags potentially updated.")
        else:
            print_error("FATAL: KoboldCpp path not valid and not corrected. Exiting."); sys.exit(1)

    # 8. Main Application Loop
    while True:
        gguf_selection_result = select_gguf_file_cli()
        if gguf_selection_result is None: break 
        if gguf_selection_result == "main_menu": continue

        gguf_file_global = gguf_selection_result
        current_model_analysis_global = tensortune_core.analyze_filename(gguf_file_global)
        
        print_title("Model Actions")
        print_info(f"Selected Model: {os.path.basename(gguf_file_global)}")
        print_info(f"  Analysis: Size ~{current_model_analysis_global.get('size_b', 'N/A')}B, Quant ~{current_model_analysis_global.get('quant', 'N/A')}, MoE: {'Yes' if current_model_analysis_global.get('is_moe') else 'No'}")

        model_actions_menu = {
            "t": "Start Auto-Tune / Use OT Strategy",
            "b": "Launch Best Remembered Config",
            "d": "Direct Launch (Settings Defaults)",
            "s": "Back to Model Selection / Main Menu"
        }
        print("\nModel Actions Menu:")
        for k_menu, v_menu in model_actions_menu.items(): print(f"  ({k_menu.upper()}) {v_menu}")
        model_action_choice = prompt("Choose action for this model", choices=list(model_actions_menu.keys()), default="t").lower()

        if model_action_choice == 's': continue 

        session_outcome = ""
        if model_action_choice == 't':
            session_outcome = run_model_tuning_session_cli() 
        
        elif model_action_choice == 'b': # Launch Best Remembered
            print_info("Attempting to launch using the best remembered configuration...")
            _, _, _, current_gpu_full_info_best = tensortune_core.get_available_vram_mb(CONFIG)
            # This is a local variable for this specific launch context
            vram_at_decision_for_best_launch = current_gpu_full_info_best.get("free_mb", 0.0)
            
            best_historical_config = tensortune_core.find_best_historical_config(DB_FILE, current_model_analysis_global, vram_at_decision_for_best_launch, CONFIG)

            if best_historical_config and best_historical_config.get("args_list"):
                print_info(f"Found best remembered config - Level: {best_historical_config['attempt_level']}, Outcome: {best_historical_config['outcome']}")
                remembered_args_list = best_historical_config["args_list"]
                remembered_args_dict = tensortune_core.args_list_to_dict(remembered_args_list)
                historical_ot_string = remembered_args_dict.pop("--overridetensors", None)
                
                base_args_for_launch = get_effective_session_args(gguf_file_global, {})
                final_effective_args_dict = base_args_for_launch.copy()
                remembered_args_dict.pop("--model", None) 
                final_effective_args_dict.update(remembered_args_dict) 
                historical_attempt_level = best_historical_config.get("attempt_level", 0)

                final_command_args_list = tensortune_core.build_command(gguf_file_global, historical_ot_string, current_model_analysis_global, final_effective_args_dict, current_attempt_level_for_tuning=historical_attempt_level)
                command_list_to_execute = tensortune_core.get_command_to_run(KOBOLDCPP_EXECUTABLE, final_command_args_list)
                
                approx_vram_used_from_hist = best_historical_config.get("approx_vram_used_kcpp_mb")

                launched_proc = launch_kobold_for_use_cli(
                    command_list_to_execute,
                    "SUCCESS_USER_LAUNCHED_BEST_REMEMBERED_CLI",
                    historical_attempt_level,
                    vram_at_launch_decision_param=vram_at_decision_for_best_launch, # Pass local value
                    approx_vram_used_mb_for_db_param=approx_vram_used_from_hist  # Pass local value
                )
                if launched_proc:
                    port_best = final_effective_args_dict.get("--port", "5000")
                    session_outcome = kcpp_control_loop_cli(port_best)
                else:
                    session_outcome = "new_gguf"
            else:
                print_warning("No suitable remembered configuration found. Suggesting Direct Launch or Auto-Tune.")
                session_outcome = "new_gguf"
        
        elif model_action_choice == 'd': # Direct Launch
            print_info("Direct Launch with current default settings...")
            effective_args_direct = get_effective_session_args(gguf_file_global, {})
            
            _, _, _, gpu_info_direct_launch = tensortune_core.get_available_vram_mb(CONFIG)
            # This is a local variable for this specific launch context
            vram_at_decision_for_direct_launch = gpu_info_direct_launch.get("free_mb", 0.0)
            approx_vram_used_for_direct_launch = None # No monitoring for direct launch

            args_list_direct = tensortune_core.build_command(gguf_file_global, None, current_model_analysis_global, effective_args_direct, current_attempt_level_for_tuning=0)
            command_list_direct = tensortune_core.get_command_to_run(KOBOLDCPP_EXECUTABLE, args_list_direct)
            
            launched_proc_direct = launch_kobold_for_use_cli(
                command_list_direct,
                "SUCCESS_USER_DIRECT_SETTINGS_CLI",
                0,
                vram_at_launch_decision_param=vram_at_decision_for_direct_launch, # Pass local value
                approx_vram_used_mb_for_db_param=approx_vram_used_for_direct_launch # Pass local value
            )
            if launched_proc_direct:
                port_direct = effective_args_direct.get("--port", "5000")
                session_outcome = kcpp_control_loop_cli(port_direct)
            else:
                session_outcome = "new_gguf"

        if session_outcome == "quit_script": break
        if session_outcome == "quit_script_leave_running":
            print_info("Exiting launcher. KoboldCpp may still be running as per user choice.")
            return
    
    print_title("TensorTune Finished")



if __name__ == "__main__":
    if dependencies['pynvml']['module'] is None and platform.system() == "Windows":
        print_warning("PyNVML (for NVIDIA GPU monitoring) is not installed. NVIDIA GPU VRAM info might be limited.")
        print_warning("You can install it with: pip install pynvml")

    tensortune_core.atexit.register(tensortune_core._cleanup_nvml) 

    try:
        main_cli()
    except KeyboardInterrupt:
        print_warning("\nLauncher terminated by user (Ctrl+C).")
        user_requested_stop_monitoring_cli = True 
        if kcpp_process_obj and kcpp_process_obj.poll() is None:
            print_info(f"Stopping monitored KCPP process (PID: {kcpp_process_obj.pid}) due to Ctrl+C...")
            tensortune_core.kill_process(kcpp_process_obj.pid, force=True)
    except Exception as e_fatal_script:
        print_error(f"\nFATAL SCRIPT ERROR: {type(e_fatal_script).__name__}: {e_fatal_script}")
        import traceback
        traceback.print_exc()
    finally:
        print_info("Exiting. Cleaning up any lingering KoboldCpp process launched by this session...")
        if last_launched_process_info.get("process_obj") and last_launched_process_info["process_obj"].poll() is None:
            if last_launched_process_info.get("pid"):
                print_info(f"Stopping last directly launched KCPP process (PID: {last_launched_process_info['pid']})...")
                tensortune_core.kill_process(last_launched_process_info["pid"], force=True)

        if kcpp_process_obj and kcpp_process_obj.poll() is None: 
             print_info(f"Stopping lingering monitored KCPP process (PID: {kcpp_process_obj.pid})...")
             tensortune_core.kill_process(kcpp_process_obj.pid, force=True)

        _kcpp_exe_for_cleanup = KOBOLDCPP_EXECUTABLE 
        if not _kcpp_exe_for_cleanup and CONFIG: 
             _kcpp_exe_for_cleanup = CONFIG.get("koboldcpp_executable", "")
        elif not CONFIG: 
            _temp_conf_cleanup, _, _ = tensortune_core.load_config() 
            _kcpp_exe_for_cleanup = _temp_conf_cleanup.get("koboldcpp_executable", "")

        if _kcpp_exe_for_cleanup:
            kcpp_exe_basename = os.path.basename(_kcpp_exe_for_cleanup)
            print_info(f"Performing cleanup sweep for processes like '{kcpp_exe_basename}'...")
            tensortune_core.kill_processes_by_name(kcpp_exe_basename)
            if kcpp_exe_basename.lower().endswith((".py", ".pyw")): 
                 tensortune_core.kill_processes_by_name("python", cmdline_substr_filter=kcpp_exe_basename)
        print_info("Launcher exited.")
