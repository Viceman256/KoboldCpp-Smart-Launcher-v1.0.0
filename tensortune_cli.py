#!/usr/bin/env python3
# TensorTune - VRAM Auto-Tuning Edition (CLI)
# Version 1.0.0 (CLI - TensorTune)

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
import koboldcpp_core

# Handle dependency imports with helpful error messages
dependencies = {
    'rich': {'required': False, 'module': None, 'purpose': 'improved terminal UI'},
    'psutil': {'required': False, 'module': None, 'purpose': 'system resource monitoring (used by core)'},
    'pynvml': {'required': False, 'module': None, 'purpose': 'NVIDIA GPU VRAM detection (used by core)'},
    'tkinter': {'required': False, 'module': None, 'purpose': 'file open dialog (optional, less used in CLI)'}
}

for dep_name, dep_info in dependencies.items():
    try:
        if dep_name == 'tkinter': # Tkinter is less critical for CLI
            pass # Don't try to import eagerly, only if specifically called by a function
        else:
            dependencies[dep_name]['module'] = __import__(dep_name)
    except ImportError:
        if dep_info['required']: # Should not happen as psutil/pynvml are core's concern
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
    def prompt(text, choices=None, default=None, password=False): # Added password for consistency, though not used here.
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

# --- Global Configuration Variables (populated from koboldcpp_core) ---
CONFIG: Dict[str, Any] = {}
KOBOLDCPP_EXECUTABLE = "" # Populated from CONFIG
DB_FILE = "" # Populated from CONFIG (absolute path)
DEFAULT_GGUF_DIR = "" # Populated from CONFIG
AUTO_OPEN_WEBUI = True # Populated from CONFIG
VRAM_SAFETY_BUFFER_MB = 0 # Populated from CONFIG
MIN_VRAM_FREE_AFTER_LOAD_MB = 0 # Populated from CONFIG
LOADING_TIMEOUT_SECONDS = 60 # Populated from CONFIG
KOBOLD_SUCCESS_PATTERN = "" # Populated from CONFIG
OOM_ERROR_KEYWORDS: List[str] = [] # Populated from CONFIG
LAUNCHER_CLI_VERSION = "1.0.0" # Updated version

# --- Runtime State Variables ---
last_gguf_directory = "" # Populated from CONFIG
last_launched_process_info: Dict[str, Any] = {"pid": None, "process_obj": None, "command_list": []}
gguf_file_global = ""
current_model_analysis_global: Dict[str, Any] = {}

tuning_in_progress = False
current_tuning_attempt_level = 0
current_tuning_min_level = 0
current_tuning_max_level = 0
current_tuning_session_base_args: Dict[str, Any] = {} # Overrides for current session
current_tuning_model_path_local = ""
current_tuning_model_analysis_local: Dict[str, Any] = {}
last_proposed_command_list_for_db: List[str] = []
# vram_at_decision_for_db: actual HW free VRAM before KCPP launch (from get_available_vram_mb free_mb, not budgeted)
vram_at_decision_for_db: Optional[float] = None
# last_approx_vram_used_kcpp_mb: actual KCPP VRAM used, calculated from (HW free before - HW free after)
last_approx_vram_used_kcpp_mb: Optional[float] = None
last_successful_monitored_run_details_cli: Optional[Dict[str, Any]] = None # For CLI display

kcpp_monitor_thread: Optional[threading.Thread] = None
kcpp_process_obj: Optional[subprocess.Popen] = None
kcpp_success_event = threading.Event()
kcpp_oom_event = threading.Event()
kcpp_output_lines_shared: List[str] = []
monitor_start_time: float = 0.0
level_of_last_monitored_run = 0


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

    detected_exe_candidate = None
    default_exe_name_platform = "koboldcpp.exe" if platform.system() == "Windows" else "./koboldcpp"
    if os.path.exists(default_exe_name_platform):
        detected_exe_candidate = os.path.abspath(default_exe_name_platform)
    elif shutil.which("koboldcpp"): # Check PATH
        detected_exe_candidate = shutil.which("koboldcpp")
    elif shutil.which(os.path.basename(default_exe_name_platform)): # Check PATH for basename if ./ fails
        detected_exe_candidate = shutil.which(os.path.basename(default_exe_name_platform))


    current_exe_in_config = config_dict.get("koboldcpp_executable", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["koboldcpp_executable"])
    exe_prompt_default = detected_exe_candidate or current_exe_in_config

    if detected_exe_candidate: print_success(f"Auto-detected KoboldCpp candidate: {detected_exe_candidate}")

    user_exe_path_input = prompt(f"Enter path to KoboldCpp executable/script", default=exe_prompt_default).strip()
    user_exe_path_abs = ""
    while True:
        potential_abs_path = os.path.abspath(user_exe_path_input)

        if os.path.exists(potential_abs_path) and (os.access(potential_abs_path, os.X_OK) or potential_abs_path.lower().endswith(".py")):
            user_exe_path_abs = potential_abs_path
            break
        
        found_in_path_shutil = shutil.which(user_exe_path_input)
        if found_in_path_shutil:
            user_exe_path_abs = os.path.abspath(found_in_path_shutil)
            print_success(f"Found '{os.path.basename(user_exe_path_input)}' in PATH: {user_exe_path_abs}")
            break
        
        # Try relative to script if .py
        if user_exe_path_input.lower().endswith(".py"):
            try:
                launcher_dir = os.path.dirname(os.path.abspath(__file__)) if not getattr(sys, 'frozen', False) else os.path.dirname(sys.executable)
                potential_relative_script_path = os.path.join(launcher_dir, user_exe_path_input)
                if os.path.exists(potential_relative_script_path):
                    user_exe_path_abs = os.path.abspath(potential_relative_script_path)
                    print_success(f"Found '{os.path.basename(user_exe_path_input)}' relative to launcher: {user_exe_path_abs}")
                    break
            except NameError: pass # __file__ not defined

        print_error(f"Path '{user_exe_path_input}' (resolved to '{potential_abs_path}') not found or not executable. Please enter a valid path.")
        user_exe_path_input = prompt("Re-enter KoboldCpp path:", default=exe_prompt_default).strip()
    config_dict["koboldcpp_executable"] = user_exe_path_abs

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

    db_file_location_absolute = config_dict["db_file"] # Already absolute from core
    print_info(f"History database will be stored at: {db_file_location_absolute}")

    default_port_from_core_template = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"]["--port"]
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
    try:
        __import__('tkinter')
        tkinter_available_check = True
    except ImportError:
        pass
    
    if tkinter_available_check:
        config_dict["cli_use_tkinter_dialog"] = confirm(
            "Prefer graphical file dialog for selecting models (if Tkinter is available)?",
            default=config_dict.get("cli_use_tkinter_dialog", False) # Default to current or False
        )
    else:
        config_dict["cli_use_tkinter_dialog"] = False # Can't use it if not available
        print_info("Tkinter (for graphical dialogs) not found. File selection will be manual path input.")


    print_info("\nGPU Detection Preferences (used for 'auto' GPU mode and VRAM display):")    
    
    print_info("\nGPU Detection Preferences (used for 'auto' GPU mode and VRAM display):")
    gpu_detect_prefs = config_dict.get("gpu_detection", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["gpu_detection"].copy())
    for vendor in ["nvidia", "amd", "intel", "apple"]:
        gpu_detect_prefs[vendor] = confirm(f"Enable detection for {vendor.upper()} GPUs?", default=gpu_detect_prefs.get(vendor, True))
    config_dict["gpu_detection"] = gpu_detect_prefs

    config_dict['first_run_completed'] = True
    save_success, save_msg = koboldcpp_core.save_launcher_config(config_dict)
    if save_success: print_success("\nInitial setup complete! Configuration saved.")
    else: print_error(f"\nFailed to save initial configuration: {save_msg}")
    return save_success


def select_gguf_file_cli() -> Optional[str]:
    global last_gguf_directory, CONFIG, DEFAULT_GGUF_DIR # CONFIG is already global

    print_title("Select GGUF Model / Main Menu")

    main_menu_actions = { "s": "Select GGUF Model File", "t": "Launcher Settings", "h": "View Global Launch History", "q": "Quit Launcher" }
    print("Main Menu Options:"); [print(f"  ({k.upper()}) {d}") for k, d in main_menu_actions.items()]

    while True:
        action_choice = prompt("Your choice", choices=list(main_menu_actions.keys()), default="s").lower()
        if action_choice == 'q': return None
        if action_choice == 't':
            manage_launcher_settings_cli()
            print_title("Select GGUF Model / Main Menu"); [print(f"  ({k.upper()}) {d}") for k, d in main_menu_actions.items()]
            continue
        if action_choice == 'h':
            view_db_history_cli()
            print_title("Select GGUF Model / Main Menu"); [print(f"  ({k.upper()}) {d}") for k, d in main_menu_actions.items()]
            continue
        if action_choice == 's': break

    tkinter_available = False
    try:
        import tkinter as _tk_ # Use alias to avoid conflicts if user's script also imports tkinter
        from tkinter import filedialog as _filedialog_
        tkinter_available = True
    except ImportError:
        pass

    # Check config setting for dialog preference
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
                koboldcpp_core.save_launcher_config(CONFIG) # Save last_used_gguf_dir
                print_success(f"Selected via dialog: {os.path.basename(abs_filepath)}")
                return abs_filepath
            else:
                print_info("File selection cancelled via dialog.")
                return "main_menu"
        except Exception as e_tk:
            print_warning(f"Tkinter file dialog failed: {e_tk}. Falling back to manual path input.")
    elif use_dialog_from_config and not tkinter_available:
        print_warning("Graphical file dialog preferred but Tkinter is not available. Using manual input.")
    
    # Manual path input (fallback or if user chose not to use dialog, or Tkinter not available)
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
            koboldcpp_core.save_launcher_config(CONFIG) # Save last_used_gguf_dir
            print_success(f"Selected via manual input: {os.path.basename(abs_path_manual)}")
            return abs_path_manual
        else:
            print_error(f"Path '{potential_full_path}' is not a valid .gguf file. Please try again or press Enter to cancel.")


def manage_launcher_settings_cli():
    global CONFIG, KOBOLDCPP_EXECUTABLE, DEFAULT_GGUF_DIR, AUTO_OPEN_WEBUI
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
            "e": "Export Configuration to File",
            "i": "Import Configuration from File",
            "r": "Reset All Settings to Defaults",
            "b": "Back to Main Menu"
        }
        print("\nSettings Menu:") # Added newline for better spacing
        for k, v in settings_options.items(): print(f"  ({k}) {v}")
        
        choice = prompt("Choose an option", choices=list(settings_options.keys()), default="b").lower()

        if choice == 'b': break

        elif choice == '1':
            old_exe = CONFIG["koboldcpp_executable"]
            new_exe_input = prompt(f"KoboldCpp Executable Path", default=old_exe).strip()
            # Resolve path (check existence, shutil.which, relative to script)
            resolved_new_exe = ""
            if os.path.exists(new_exe_input) and (os.access(new_exe_input, os.X_OK) or new_exe_input.lower().endswith(".py")):
                resolved_new_exe = os.path.abspath(new_exe_input)
            elif shutil.which(new_exe_input):
                resolved_new_exe = os.path.abspath(shutil.which(new_exe_input))
            elif new_exe_input.lower().endswith(".py"):
                try:
                    launcher_dir = os.path.dirname(os.path.abspath(__file__)) if not getattr(sys, 'frozen', False) else os.path.dirname(sys.executable)
                    potential_relative_script_path = os.path.join(launcher_dir, new_exe_input)
                    if os.path.exists(potential_relative_script_path):
                        resolved_new_exe = os.path.abspath(potential_relative_script_path)
                except NameError: pass

            if resolved_new_exe:
                CONFIG["koboldcpp_executable"] = resolved_new_exe
                KOBOLDCPP_EXECUTABLE = CONFIG["koboldcpp_executable"]
                print_success(f"Executable path set to: {KOBOLDCPP_EXECUTABLE}")
            elif new_exe_input: # If user provided input but it wasn't valid
                print_error(f"Path '{new_exe_input}' is not valid. Keeping: {old_exe}")
            # If new_exe_input was empty, old_exe is kept by default from prompt
        
        elif choice == '2':
            old_dir = CONFIG["default_gguf_dir"]
            new_dir = prompt(f"Default GGUF Directory (blank to clear)", default=old_dir if old_dir else os.getcwd()).strip()
            if new_dir and os.path.isdir(new_dir):
                CONFIG["default_gguf_dir"] = os.path.abspath(new_dir)
                DEFAULT_GGUF_DIR = CONFIG["default_gguf_dir"]
                print_success(f"Default GGUF directory set to: {DEFAULT_GGUF_DIR}")
            elif not new_dir: # User entered blank
                 CONFIG["default_gguf_dir"] = ""
                 DEFAULT_GGUF_DIR = ""
                 print_success("Default GGUF directory cleared.")
            else: # User entered something, but it's not a valid directory
                print_error(f"Directory '{new_dir}' is not valid. Keeping: {old_dir if old_dir else 'None'}")
        
        elif choice == '3':
            CONFIG["auto_open_webui"] = confirm("Auto-Open WebUI after successful launch?", default=CONFIG.get("auto_open_webui", True))
            AUTO_OPEN_WEBUI = CONFIG["auto_open_webui"]
            print_success(f"Auto-Open WebUI set to: {AUTO_OPEN_WEBUI}")

        elif choice == '4':
            current_mode = CONFIG.get("gpu_selection_mode", "auto")
            modes = ["auto", "nvidia", "amd", "intel", "apple"]
            new_mode = prompt(f"GPU Selection Mode ({'/'.join(modes)})", choices=modes, default=current_mode).lower()
            CONFIG["gpu_selection_mode"] = new_mode
            print_success(f"GPU Selection Mode set to: {new_mode}")

        elif choice == '5':
            current_idx = CONFIG.get("selected_gpu_index", 0)
            gpu_type_for_listing = CONFIG.get("gpu_selection_mode", "auto")
            
            # Get actual detected type if mode is 'auto'
            effective_gpu_type = gpu_type_for_listing
            if gpu_type_for_listing == "auto":
                _, _, _, gpu_info_auto = koboldcpp_core.get_available_vram_mb(CONFIG)
                if gpu_info_auto and gpu_info_auto.get("success") and gpu_info_auto.get("type") not in ["Unknown/None_Auto", "N/A", "Unknown/None"]:
                    effective_gpu_type = gpu_info_auto.get("type", "auto").lower()
                    if effective_gpu_type == "apple_metal": effective_gpu_type = "apple"
                    print_info(f"Auto-detected GPU type as '{effective_gpu_type}' for listing IDs.")
                else:
                    print_warning("Could not auto-detect a specific GPU type for ID listing.")
                    effective_gpu_type = "none" # Prevent listing attempts for invalid type

            gpus_found = []
            if effective_gpu_type == "nvidia": gpus_found = koboldcpp_core.list_nvidia_gpus()
            elif effective_gpu_type == "amd": gpus_found = koboldcpp_core.list_amd_gpus_windows() if platform.system() == "Windows" else koboldcpp_core.list_amd_gpus_linux()
            elif effective_gpu_type == "intel": gpus_found = koboldcpp_core.list_intel_gpus()
            elif effective_gpu_type == "apple": gpus_found = koboldcpp_core.list_apple_gpus()
            
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
            elif gpu_type_for_listing == "auto" and effective_gpu_type != "none": # Auto mode, but specific type was detected
                print_info(f"In 'auto' mode (detected as '{effective_gpu_type.upper()}'), GPU index is typically 0 if only one GPU of this type. Listing was for this detected type.")
                new_idx_str = prompt(f"Target GPU Index (usually 0 for primary of type '{effective_gpu_type.upper()}')", default=str(current_idx))
                try: CONFIG["selected_gpu_index"] = int(new_idx_str); print_success(f"Target GPU Index set to: {int(new_idx_str)}")
                except: print_error("Invalid index.");
            else: # No GPUs for selected type, or auto-detection failed to find a specific type
                print_warning(f"No GPUs listed for type '{effective_gpu_type.upper()}'. Cannot select specific ID. Index remains {current_idx}.")

        elif choice == '6':
            print_info("VRAM Override Settings:")
            current_override_status = CONFIG.get("override_vram_budget", False)
            current_manual_budget = CONFIG.get("manual_vram_total_mb", 8192)
            
            if current_override_status:
                print_info(f"VRAM Override is currently ACTIVE. Manual Budget: {current_manual_budget} MB.")
            else:
                print_info("VRAM Override is currently INACTIVE. Auto-detected VRAM will be used.")

            enable_override = confirm("Do you want to enable VRAM Override?", default=current_override_status)
            
            if enable_override:
                CONFIG["override_vram_budget"] = True
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
            else:
                CONFIG["override_vram_budget"] = False
                # Optionally, you might want to reset manual_vram_total_mb to default when override is disabled
                # CONFIG["manual_vram_total_mb"] = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["manual_vram_total_mb"]
                print_success("VRAM Override DISABLED. Launcher will use auto-detected VRAM values.")
        
        elif choice == '7':
            print_info("GPU Detection Preferences:")
            gpu_detect_prefs = CONFIG.get("gpu_detection", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["gpu_detection"].copy())
            for vendor in ["nvidia", "amd", "intel", "apple"]:
                gpu_detect_prefs[vendor] = confirm(f"Enable detection for {vendor.upper()} GPUs?", default=gpu_detect_prefs.get(vendor, True))
            CONFIG["gpu_detection"] = gpu_detect_prefs
            print_success("GPU detection preferences updated.")

        elif choice == '8':
            tkinter_available_check = False
            try:
                __import__('tkinter')
                tkinter_available_check = True
            except ImportError:
                pass
            
            if tkinter_available_check:
                current_pref = CONFIG.get("cli_use_tkinter_dialog", False)
                CONFIG["cli_use_tkinter_dialog"] = confirm(
                    "Use graphical file dialog for model selection in CLI (requires Tkinter)?",
                    default=current_pref
                )
                print_success(f"Graphical file dialog preference set to: {CONFIG['cli_use_tkinter_dialog']}")
            else:
                CONFIG["cli_use_tkinter_dialog"] = False # Ensure it's false if Tkinter not found
                print_warning("Tkinter (for graphical dialogs) not found. Preference set to False.")
                
        elif choice == 'e': # Export
            export_path = prompt("Enter filepath to export configuration (e.g., launcher_backup.json)", default="kcpp_launcher_config_export.json").strip()
            if export_path:
                success_export, msg_export = koboldcpp_core.export_config_to_file(CONFIG.copy(), export_path)
                if success_export: print_success(msg_export)
                else: print_error(msg_export)
        
        elif choice == 'i': # Import
            import_path = prompt("Enter filepath of configuration to import").strip()
            if import_path and os.path.exists(import_path):
                imported_data, msg_import = koboldcpp_core.import_config_from_file(import_path)
                if imported_data:
                    if confirm(f"Importing will OVERWRITE current settings. Backup current config? ({koboldcpp_core.CONFIG_FILE})", default=True):
                        backup_path_auto = koboldcpp_core.CONFIG_FILE + f".backup_before_import_{time.strftime('%Y%m%d-%H%M%S')}.json"
                        try:
                            if os.path.exists(koboldcpp_core.CONFIG_FILE): # Check if source file exists
                                shutil.copy2(koboldcpp_core.CONFIG_FILE, backup_path_auto)
                                print_success(f"Current config backed up to {backup_path_auto}")
                            else:
                                print_warning(f"Current config file {koboldcpp_core.CONFIG_FILE} not found. Cannot backup.")
                        except Exception as e_backup:
                            print_warning(f"Could not backup current config: {e_backup}")
                    
                    # Before overwriting CONFIG, save the current one if there are pending changes from this menu session
                    save_ok_before_import, save_msg_before_import = koboldcpp_core.save_launcher_config(CONFIG)
                    if not save_ok_before_import:
                        print_warning(f"Note: Failed to save current session's pending setting changes before import: {save_msg_before_import}")

                    # Now apply imported data and save it as the new primary config
                    temp_config_for_saving_import = imported_data.copy() # Use a copy
                    # Ensure db_file path is made absolute if it's relative after import
                    db_file_val_from_imported = temp_config_for_saving_import.get("db_file", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["db_file"])
                    if not os.path.isabs(db_file_val_from_imported):
                        # If db_file is just a basename, make it absolute to user_data_dir
                        # If it's a relative path, it's trickier; for CLI, perhaps just keep it and let core handle on next full load
                        # For now, let's assume core's save_launcher_config will handle this correctly if it's a basename.
                        # If it's a truly relative path like "subdir/db.sqlite", user should be aware.
                         if os.path.basename(db_file_val_from_imported) == db_file_val_from_imported: # Is it just a filename?
                            temp_config_for_saving_import["db_file"] = os.path.join(koboldcpp_core._get_user_app_data_dir(), db_file_val_from_imported)

                    save_imported_ok, save_imported_msg = koboldcpp_core.save_launcher_config(temp_config_for_saving_import)
                    if save_imported_ok:
                        # Reload CONFIG from the newly saved file to ensure consistency
                        CONFIG, _, _ = koboldcpp_core.load_config()
                        # Re-populate global vars that depend on CONFIG
                        KOBOLDCPP_EXECUTABLE = CONFIG["koboldcpp_executable"]
                        DB_FILE = CONFIG["db_file"]
                        DEFAULT_GGUF_DIR = CONFIG.get("default_gguf_dir", "")
                        AUTO_OPEN_WEBUI = CONFIG.get("auto_open_webui", True)
                        print_success(f"{msg_import} New configuration saved and reloaded.")
                    else:
                        print_error(f"Failed to save imported configuration: {save_imported_msg}. Current settings remain.")
                else:
                    print_error(msg_import)
            elif import_path:
                print_error(f"Import file not found: {import_path}")

        elif choice == 'r': # Reset
             if confirm("WARNING: This will reset ALL launcher settings to defaults. Are you sure?", default=False):
                backup_path_reset = koboldcpp_core.CONFIG_FILE + f".backup_before_reset_{time.strftime('%Y%m%d-%H%M%S')}.json"
                try:
                    if os.path.exists(koboldcpp_core.CONFIG_FILE):
                        shutil.copy2(koboldcpp_core.CONFIG_FILE, backup_path_reset)
                        print_success(f"Current config backed up to {backup_path_reset}")
                    if os.path.exists(koboldcpp_core.CONFIG_FILE): # Check again before removing
                        os.remove(koboldcpp_core.CONFIG_FILE)
                except Exception as e_remove:
                    print_warning(f"Could not remove/backup current config for reset: {e_remove}")

                temp_conf, _, load_msg = koboldcpp_core.load_config() # This saves defaults if file was missing
                CONFIG = temp_conf
                KOBOLDCPP_EXECUTABLE = CONFIG["koboldcpp_executable"]
                DB_FILE = CONFIG["db_file"]
                DEFAULT_GGUF_DIR = CONFIG.get("default_gguf_dir", "")
                AUTO_OPEN_WEBUI = CONFIG.get("auto_open_webui", True)
                # ... re-populate other globals as needed ...
                print_success(f"Configuration reset to defaults. {load_msg}")
                print_warning("You may need to run the first-time setup prompts again if paths changed.")


        # Save changes made in this settings menu session (unless it was import/reset which handle saving)
        if choice not in ['i', 'r']: # 'e' should save current config state.
            save_ok, save_message = koboldcpp_core.save_launcher_config(CONFIG)
            if not save_ok:
                print_error(f"Failed to save settings: {save_message}")
            else:
                if choice != 'b': 
                     print_info("Settings saved.")
        print("-" * 30)


def view_db_history_cli(model_filepath_filter: Optional[str] = None):
    if model_filepath_filter:
        print_info(f"Loading history for model: {os.path.basename(model_filepath_filter)} from DB: {DB_FILE}")
    else:
        print_info(f"Loading global launch history from DB: {DB_FILE}")

    all_history_entries_from_db = koboldcpp_core.get_history_entries(DB_FILE, limit=100)

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
            vram_at_launch_val = f"{record_data[4]}MB" if record_data[4] is not None else "N/A" # Actual HW VRAM
            ot_level_val = str(record_data[5]) if record_data[5] is not None else "N/A"
            outcome_val = str(record_data[6]) if record_data[6] else "N/A"
            vram_used_val = f"{record_data[7]}MB" if record_data[7] is not None else "N/A" # Actual KCPP VRAM
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
    effective_args_dict = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
    global_defaults_from_config = CONFIG.get("default_args", {})
    effective_args_dict.update(global_defaults_from_config)

    if model_file_path and "model_specific_args" in CONFIG:
        model_specifics_from_config = CONFIG["model_specific_args"].get(model_file_path, {})
        effective_args_dict.update(model_specifics_from_config)

    effective_args_dict.update(session_specific_overrides)
    return effective_args_dict


def edit_current_args_interactive_cli(model_path_for_specifics: Optional[str], current_session_args_overrides: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], bool]:
    permanent_args_were_changed = False
    editable_arg_defs = [
        d for d in koboldcpp_core.KOBOLDCPP_ARG_DEFINITIONS
        if d["key"] not in ["--model", "--overridetensors"]
    ]
    editable_arg_defs.sort(key=lambda x: (x.get("category", "zz_default"), x.get("name", x["key"])))

    temp_session_overrides = current_session_args_overrides.copy()

    while True:
        print_title("Edit Base Startup Arguments (KCPP Parameters)")
        effective_args_for_display = get_effective_session_args(model_path_for_specifics, temp_session_overrides)
        
        idx_to_param_map = {}

        if dependencies['rich']['module']:
            args_table = Table(title="Effective Arguments (Core -> Global -> Model -> Session)")
            args_table.add_column("#", style="cyan"); args_table.add_column("Argument", style="green", max_width=20, overflow="fold");
            args_table.add_column("Current Value", style="yellow", max_width=15, overflow="fold"); args_table.add_column("Description", overflow="fold");
            
            for i, arg_def in enumerate(editable_arg_defs):
                param_key = arg_def["key"]
                idx_str_display = str(i + 1); idx_to_param_map[idx_str_display] = param_key
                value_to_display = effective_args_for_display.get(param_key)
                is_bool_type_param = arg_def.get("type_hint") in ["bool", "bool_flag"]
                
                if is_bool_type_param:
                    if value_to_display is True: value_str_display = "ENABLED"
                    elif value_to_display is False: value_str_display = "DISABLED"
                    else: value_str_display = "NOT SET (KCPP default)"
                else:
                    value_str_display = str(value_to_display) if value_to_display is not None else "NOT SET (KCPP default)"
                
                help_text_for_arg = arg_def.get("help", "No description available.")
                args_table.add_row(idx_str_display, arg_def.get("name", param_key), value_str_display, help_text_for_arg)
            console.print(args_table)
        else: # Basic print
            print("Effective Arguments for This Session (Core -> Global -> Model -> Session):")
            for i, arg_def in enumerate(editable_arg_defs):
                param_key = arg_def["key"]
                idx_str_display = str(i + 1); idx_to_param_map[idx_str_display] = param_key
                value_to_display = effective_args_for_display.get(param_key)
                is_bool_type_param = arg_def.get("type_hint") in ["bool", "bool_flag"]
                value_str_display = ("ENABLED" if value_to_display is True else ("DISABLED" if value_to_display is False else str(value_to_display))) if is_bool_type_param else (str(value_to_display) if value_to_display is not None else "NOT SET")
                help_text_for_arg = arg_def.get("help", "")
                print(f"  ({idx_str_display}) {arg_def.get('name', param_key):<25}: {value_str_display:<15} - {help_text_for_arg}")

        print("\nActions: (#) to edit | (T#)oggle bool | (P)ermanently save current session args for this model")
        print("         (S)ave session changes & Continue | (X) Cancel session edits & Continue to launch")
        user_choice_input = prompt("Your choice", default="s").lower().strip()

        if user_choice_input == 's': return temp_session_overrides, permanent_args_were_changed
        if user_choice_input == 'x': return current_session_args_overrides, permanent_args_were_changed

        if user_choice_input == 'p':
            if not model_path_for_specifics:
                print_error("No model selected to save permanent args for. Select a model first."); continue
            
            print_info(f"Current effective arguments (including session changes) for '{os.path.basename(model_path_for_specifics)}' will be saved as its new permanent defaults.")
            if confirm(f"Save these as permanent defaults for {os.path.basename(model_path_for_specifics)}?", default=True):
                if "model_specific_args" not in CONFIG: CONFIG["model_specific_args"] = {}
                CONFIG["model_specific_args"][model_path_for_specifics] = {} 
                
                model_specifics_to_set = CONFIG["model_specific_args"][model_path_for_specifics]
                
                global_baseline_args_for_perm = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
                global_baseline_args_for_perm.update(CONFIG.get("default_args", {}))

                for arg_def_perm in editable_arg_defs:
                    param_key_perm = arg_def_perm["key"]
                    value_from_effective_display = effective_args_for_display.get(param_key_perm)
                    global_baseline_for_this_arg = global_baseline_args_for_perm.get(param_key_perm)
                    
                    if value_from_effective_display is not None:
                        is_bool_type = arg_def_perm.get("type_hint") in ["bool", "bool_flag"]
                        is_different = False
                        if is_bool_type:
                            norm_global_bool = False
                            if isinstance(global_baseline_for_this_arg, bool): norm_global_bool = global_baseline_for_this_arg
                            elif isinstance(global_baseline_for_this_arg, str): norm_global_bool = global_baseline_for_this_arg.lower() == 'true'
                            if value_from_effective_display != norm_global_bool: is_different = True
                        else:
                            if str(value_from_effective_display) != str(global_baseline_for_this_arg): is_different = True
                        
                        if is_different:
                            model_specifics_to_set[param_key_perm] = value_from_effective_display

                if not CONFIG["model_specific_args"][model_path_for_specifics]:
                    del CONFIG["model_specific_args"][model_path_for_specifics]
                
                koboldcpp_core.save_launcher_config(CONFIG)
                permanent_args_were_changed = True
                print_success(f"Permanent arguments saved for {os.path.basename(model_path_for_specifics)}.")
                temp_session_overrides.clear()
            else: print_info("Permanent argument save cancelled.")
            continue

        arg_to_edit_name = None; is_toggle_action = False
        if user_choice_input.startswith('t') and user_choice_input[1:].isdigit():
            raw_idx_str = user_choice_input[1:]
            if raw_idx_str in idx_to_param_map:
                arg_to_edit_name = idx_to_param_map[raw_idx_str]; is_toggle_action = True
        elif user_choice_input.isdigit():
            if user_choice_input in idx_to_param_map:
                arg_to_edit_name = idx_to_param_map[user_choice_input]

        if arg_to_edit_name:
            arg_def_for_edit = next((d for d in editable_arg_defs if d["key"] == arg_to_edit_name), None)
            if not arg_def_for_edit: print_error("Internal error: Arg definition not found."); continue

            widget_type_hint_edit = arg_def_for_edit.get("type_hint", "str")

            if is_toggle_action:
                if widget_type_hint_edit in ["bool", "bool_flag"]:
                    current_effective_val_for_toggle = effective_args_for_display.get(arg_to_edit_name, False)
                    current_bool_val = False
                    if isinstance(current_effective_val_for_toggle, bool): current_bool_val = current_effective_val_for_toggle
                    elif isinstance(current_effective_val_for_toggle, str): current_bool_val = current_effective_val_for_toggle.lower() == 'true'
                    
                    temp_session_overrides[arg_to_edit_name] = not current_bool_val
                    print_success(f"Toggled session override for {arg_def_for_edit.get('name', arg_to_edit_name)} to {'ENABLED' if temp_session_overrides[arg_to_edit_name] else 'DISABLED'}")
                else: print_error(f"Cannot toggle '{arg_def_for_edit.get('name', arg_to_edit_name)}'. It is not a boolean type argument.")
                continue
            
            current_val_for_edit_prompt = effective_args_for_display.get(arg_to_edit_name)
            prompt_message = f"New value for {arg_def_for_edit.get('name', arg_to_edit_name)} (Current: {current_val_for_edit_prompt if current_val_for_edit_prompt is not None else 'Default'} | Enter 'unset' to remove session override):"
            new_val_str_input = prompt(prompt_message).strip()

            if new_val_str_input.lower() == 'unset':
                if arg_to_edit_name in temp_session_overrides:
                    del temp_session_overrides[arg_to_edit_name]
                print_success(f"Removed session override for {arg_def_for_edit.get('name', arg_to_edit_name)}. It will use model/global default.")
            elif widget_type_hint_edit in ["bool", "bool_flag"]:
                if new_val_str_input.lower() in ['true', 'yes', '1', 'on', 'enabled']: temp_session_overrides[arg_to_edit_name] = True
                elif new_val_str_input.lower() in ['false', 'no', '0', 'off', 'disabled']: temp_session_overrides[arg_to_edit_name] = False
                else: print_error(f"Invalid boolean value for {arg_def_for_edit.get('name', arg_to_edit_name)}. Use 'true' or 'false'."); continue
            elif widget_type_hint_edit in ["int", "str_auto_num"] and new_val_str_input.lower() != "auto":
                try:
                    temp_session_overrides[arg_to_edit_name] = new_val_str_input
                except ValueError:
                    print_error(f"Invalid numeric value for {arg_def_for_edit.get('name', arg_to_edit_name)}.")
                    continue
            else:
                temp_session_overrides[arg_to_edit_name] = new_val_str_input
            
            if new_val_str_input.lower() != 'unset':
                 print_success(f"Set session override for {arg_def_for_edit.get('name', arg_to_edit_name)} to {temp_session_overrides.get(arg_to_edit_name)}")
        else: print_error("Invalid choice. Please enter a number from the list or a valid action.")


def _log_to_cli_live_output(text_line: str, live_obj: Optional[Any] = None, progress_obj: Optional[Any] = None):
    print(text_line.strip())


def monitor_kcpp_output_thread_target_cli(
    process: subprocess.Popen,
    success_event_thread: threading.Event,
    oom_event_thread: threading.Event,
    output_lines_list_shared: List[str],
    success_regex_str_config: str,
    oom_keywords_list_config: List[str],
    target_port_for_success_check: str,
    live_display_obj: Optional[Any] = None, 
    progress_bar_obj: Optional[Any] = None 
):
    try:
        for line_bytes_from_kcpp in iter(process.stdout.readline, b''):
            if not line_bytes_from_kcpp: break
            try:
                line_decoded_from_kcpp = line_bytes_from_kcpp.decode('utf-8', errors='replace')
            except UnicodeDecodeError:
                line_decoded_from_kcpp = line_bytes_from_kcpp.decode('latin-1', errors='replace')
            
            _log_to_cli_live_output(line_decoded_from_kcpp, live_display_obj, progress_bar_obj)
            
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
                        for oom_keyword in oom_keywords_list_config: # oom_keywords_list_config is already lowercased
                            if oom_keyword in line_strip_lower_case:
                                oom_event_thread.set()
                                break
            
            if success_event_thread.is_set() or oom_event_thread.is_set():
                break
    except Exception as e_monitor:
        err_msg_monitor = f"\nError in KCPP output monitor thread: {type(e_monitor).__name__}: {e_monitor}\n"
        _log_to_cli_live_output(err_msg_monitor, live_display_obj, progress_bar_obj)
    finally:
        if process.stdout and not process.stdout.closed:
            try: process.stdout.close()
            except: pass
        _log_to_cli_live_output("\nKCPP output monitoring thread finished.\n", live_display_obj, progress_bar_obj)


def launch_and_monitor_for_tuning_cli():
    global kcpp_process_obj, kcpp_monitor_thread, monitor_start_time
    global last_proposed_command_list_for_db, vram_at_decision_for_db, last_approx_vram_used_kcpp_mb
    global level_of_last_monitored_run, last_successful_monitored_run_details_cli

    if kcpp_process_obj and kcpp_process_obj.poll() is None:
        print_warning("A KoboldCpp process is already being monitored. Please stop it first or wait.")
        return "continue_tuning"

    print_info(f"Tuning: Launching & Monitoring for OT Level {current_tuning_attempt_level}")
    kcpp_success_event.clear(); kcpp_oom_event.clear(); kcpp_output_lines_shared.clear()
    last_approx_vram_used_kcpp_mb = None 
    level_of_last_monitored_run = current_tuning_attempt_level

    ot_string_for_launch = koboldcpp_core.generate_overridetensors(current_tuning_model_analysis_local, current_tuning_attempt_level)
    args_for_kcpp_run_list = koboldcpp_core.build_command(
        current_tuning_model_path_local, ot_string_for_launch,
        current_tuning_model_analysis_local, current_tuning_session_base_args
    )
    last_proposed_command_list_for_db = koboldcpp_core.get_command_to_run(KOBOLDCPP_EXECUTABLE, args_for_kcpp_run_list)
    
    # Get actual HW free VRAM for DB logging, not budgeted
    _, _, _, gpu_info_rich_before_launch = koboldcpp_core.get_available_vram_mb(
        CONFIG, # Pass current global config
        target_gpu_type=CONFIG.get("gpu_selection_mode", "auto") if CONFIG.get("gpu_selection_mode", "auto") != "auto" else None,
        target_gpu_index=CONFIG.get("selected_gpu_index", 0)
    )
    vram_at_decision_for_db = gpu_info_rich_before_launch.get("free_mb", 0.0) # Actual HW free VRAM

    kcpp_process_obj, launch_error_msg = koboldcpp_core.launch_process(
        last_proposed_command_list_for_db, capture_output=True, new_console=False, use_text_mode=False
    )

    if launch_error_msg or not kcpp_process_obj:
        print_error(f"Failed to launch KCPP for monitoring: {launch_error_msg or 'Unknown error'}")
        koboldcpp_core.save_config_to_db(
            DB_FILE, current_tuning_model_path_local, current_tuning_model_analysis_local,
            vram_at_decision_for_db, last_proposed_command_list_for_db,
            level_of_last_monitored_run, "LAUNCH_FAILED_SETUP_CLI", None
        )
        return "continue_tuning"

    print_info(f"KoboldCpp process started (PID: {kcpp_process_obj.pid}). Monitoring output...")
    effective_args_for_port_check = get_effective_session_args(current_tuning_model_path_local, current_tuning_session_base_args)
    target_port_str_for_success = effective_args_for_port_check.get("--port", "5000")

    rich_progress_live_obj = None; loading_task_id = None
    if dependencies['rich']['module']:
        rich_progress_live_obj = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), TimeRemainingColumn(), console=console, transient=True)
        loading_task_id = rich_progress_live_obj.add_task("KCPP Loading...", total=float(LOADING_TIMEOUT_SECONDS))
        rich_progress_live_obj.start()

    kcpp_monitor_thread = threading.Thread(
        target=monitor_kcpp_output_thread_target_cli,
        args=(kcpp_process_obj, kcpp_success_event, kcpp_oom_event, kcpp_output_lines_shared, KOBOLD_SUCCESS_PATTERN, OOM_ERROR_KEYWORDS, target_port_str_for_success, rich_progress_live_obj, loading_task_id),
        daemon=True
    )
    kcpp_monitor_thread.start()
    monitor_start_time = time.monotonic()
    final_outcome_key_from_monitor = "UNKNOWN_EXIT_CLI"
    try:
        while True:
            elapsed_monitor_time = time.monotonic() - monitor_start_time
            if rich_progress_live_obj and loading_task_id is not None and not rich_progress_live_obj.finished and rich_progress_live_obj.tasks[loading_task_id].started:
                rich_progress_live_obj.update(loading_task_id, completed=min(elapsed_monitor_time, float(LOADING_TIMEOUT_SECONDS)))
            
            process_has_exited = kcpp_process_obj.poll() is not None
            
            if kcpp_success_event.is_set(): final_outcome_key_from_monitor = "SUCCESS_LOAD_DETECTED_CLI"; break
            if kcpp_oom_event.is_set(): final_outcome_key_from_monitor = "OOM_CRASH_DETECTED_CLI"; break
            if process_has_exited and not kcpp_success_event.is_set() and not kcpp_oom_event.is_set():
                final_outcome_key_from_monitor = "PREMATURE_EXIT_CLI"; break
            if elapsed_monitor_time > LOADING_TIMEOUT_SECONDS:
                final_outcome_key_from_monitor = "TIMEOUT_NO_SIGNAL_CLI"; break
            time.sleep(0.25)
    finally:
        if rich_progress_live_obj and not rich_progress_live_obj.finished : rich_progress_live_obj.stop()

    print_info(f"Monitoring completed. Initial Outcome: {final_outcome_key_from_monitor}")
    if final_outcome_key_from_monitor in ["TIMEOUT_NO_SIGNAL_CLI", "OOM_CRASH_DETECTED_CLI", "PREMATURE_EXIT_CLI"] or \
       "OOM" in final_outcome_key_from_monitor.upper() or "CRASH" in final_outcome_key_from_monitor.upper():
        if kcpp_process_obj and kcpp_process_obj.poll() is None:
            print_info("Terminating KCPP process due to unfavorable outcome...")
            koboldcpp_core.kill_process(kcpp_process_obj.pid, force=True)
        kcpp_process_obj = None

    db_outcome_to_save_str = final_outcome_key_from_monitor
    if final_outcome_key_from_monitor == "SUCCESS_LOAD_DETECTED_CLI":
        print_info("API detected. Waiting a few seconds for VRAM to stabilize...")
        vram_stabilization_wait_s = CONFIG.get("vram_stabilization_wait_s", 3.0)
        time.sleep(max(2.0, float(vram_stabilization_wait_s)))
        
        # Get full GPU info, including actual HW and budgeted VRAM after load
        _, _, _, gpu_info_rich_after_load = koboldcpp_core.get_available_vram_mb(
            CONFIG, # Pass current global config
            target_gpu_type=CONFIG.get("gpu_selection_mode", "auto") if CONFIG.get("gpu_selection_mode", "auto") != "auto" else None,
            target_gpu_index=CONFIG.get("selected_gpu_index", 0)
        )
        
        budgeted_free_vram_after_load = gpu_info_rich_after_load.get("free_mb_budgeted", 0.0)
        actual_hw_free_vram_after_load = gpu_info_rich_after_load.get("free_mb", 0.0)
        actual_hw_total_vram_after_load = gpu_info_rich_after_load.get("total_mb", 0.0)

        if actual_hw_total_vram_after_load > 0 and vram_at_decision_for_db is not None:
            # Calculate actual KCPP VRAM usage
            vram_used_by_kcpp_actual_hw = vram_at_decision_for_db - actual_hw_free_vram_after_load
            last_approx_vram_used_kcpp_mb = max(0, min(vram_used_by_kcpp_actual_hw, actual_hw_total_vram_after_load))
            
            print_info(f"Budgeted VRAM after load: {budgeted_free_vram_after_load:.0f}MB free. Approx Actual KCPP VRAM usage: {last_approx_vram_used_kcpp_mb:.0f}MB")
            
            if budgeted_free_vram_after_load < MIN_VRAM_FREE_AFTER_LOAD_MB:
                print_warning(f"Budgeted VRAM tight! {budgeted_free_vram_after_load:.0f}MB < {MIN_VRAM_FREE_AFTER_LOAD_MB}MB target.")
                db_outcome_to_save_str = "SUCCESS_LOAD_VRAM_TIGHT_CLI"
            else:
                print_success("Budgeted VRAM usage OK.")
                db_outcome_to_save_str = "SUCCESS_LOAD_VRAM_OK_CLI"
            
            last_successful_monitored_run_details_cli = {
                "level": level_of_last_monitored_run,
                "outcome": db_outcome_to_save_str,
                "vram_used_mb": f"{last_approx_vram_used_kcpp_mb:.0f}" if last_approx_vram_used_kcpp_mb is not None else "N/A"
            }
        else:
            db_outcome_to_save_str = "SUCCESS_LOAD_NO_VRAM_CHECK_CLI"
            last_successful_monitored_run_details_cli = {
                "level": level_of_last_monitored_run, "outcome": db_outcome_to_save_str, "vram_used_mb": "N/A"
            }

    koboldcpp_core.save_config_to_db(
        DB_FILE, current_tuning_model_path_local, current_tuning_model_analysis_local,
        vram_at_decision_for_db, # Actual HW free VRAM before launch
        last_proposed_command_list_for_db,
        level_of_last_monitored_run, db_outcome_to_save_str, 
        last_approx_vram_used_kcpp_mb # Actual KCPP VRAM usage
    )
    return handle_post_monitoring_choices_cli(db_outcome_to_save_str, kcpp_process_obj)


def handle_post_monitoring_choices_cli(outcome_from_monitor: str, monitored_kcpp_instance: Optional[subprocess.Popen]):
    global current_tuning_attempt_level, tuning_in_progress, kcpp_process_obj
    global last_launched_process_info, level_of_last_monitored_run

    kcpp_process_obj = monitored_kcpp_instance
    choices_dict = {}; default_choice_key = ""
    print_title("Post-Monitoring Options")
    print_info(f"Outcome of monitored launch: {outcome_from_monitor}")
    if last_approx_vram_used_kcpp_mb is not None:
        print_info(f"Approx. Actual KCPP VRAM Used: {last_approx_vram_used_kcpp_mb:.0f} MB")
    
    kcpp_is_still_running = kcpp_process_obj and kcpp_process_obj.poll() is None

    if "SUCCESS_LOAD_VRAM_OK" in outcome_from_monitor:
        print_success("KCPP loaded successfully (Budgeted VRAM OK).")
        choices_dict = { "u": "✅ Accept & Use this KCPP instance", "s": "💾 Save as Good, Auto-Adjust for More GPU & Continue Tuning", "g": "⚙️ Manually Try More GPU (This Session) & Continue Tuning", "c": "⚙️ Manually Try More CPU (This Session) & Continue Tuning", "q": "↩️ Save Outcome & Return to Tuning Menu (Manual Adjust)" }
        default_choice_key = "u" if kcpp_is_still_running else "s"
    elif "SUCCESS_LOAD_VRAM_TIGHT" in outcome_from_monitor:
        print_warning("KCPP loaded, but Budgeted VRAM is tight!")
        choices_dict = { "a": "⚠️ Auto-Adjust for More CPU & Continue Tuning", "l": "🚀 Launch This Config Anyway (Risky)", "q": "↩️ Save Outcome & Return to Tuning Menu (Manual Adjust)" }
        default_choice_key = "a"
    elif "OOM" in outcome_from_monitor or "CRASH" in outcome_from_monitor or "PREMATURE_EXIT" in outcome_from_monitor:
        print_error("KCPP failed to load properly (OOM/Crash/Premature Exit).")
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
            print_warning("KCPP status unclear or it has already exited without specific error.")
            choices_dict = {"q": "↩️ Save Outcome & Return to Tuning Menu (Manual Adjust)"}
            default_choice_key = "q"

    for key, desc in choices_dict.items(): print(f"  ({key.upper()}) {desc}")
    user_action_choice = prompt("Your choice?", choices=list(choices_dict.keys()), default=default_choice_key).lower()
    
    db_outcome_suffix_for_action = "_CLI"
    should_stop_monitored_kcpp = True
    
    if user_action_choice in ['u', 'l'] and kcpp_is_still_running:
        should_stop_monitored_kcpp = False

    if kcpp_is_still_running and should_stop_monitored_kcpp:
        print_info(f"Stopping monitored KCPP instance (PID: {kcpp_process_obj.pid})...")
        koboldcpp_core.kill_process(kcpp_process_obj.pid, force=True)
        kcpp_process_obj = None

    if user_action_choice == 'u':
        if kcpp_is_still_running:
            print_info("Keeping current KoboldCpp instance running for use.")
            db_outcome_suffix_for_action = "_USER_ACCEPTED_TUNED_CLI"
            last_launched_process_info["pid"] = kcpp_process_obj.pid
            last_launched_process_info["process_obj"] = kcpp_process_obj
            last_launched_process_info["command_list"] = last_proposed_command_list_for_db
            kcpp_process_obj = None 
            
            effective_args_for_webui = get_effective_session_args(current_tuning_model_path_local, current_tuning_session_base_args)
            port_for_webui = effective_args_for_webui.get("--port", "5000")
            if AUTO_OPEN_WEBUI: webbrowser.open(f"http://localhost:{port_for_webui}")
            
            final_outcome_for_db_update = outcome_from_monitor + db_outcome_suffix_for_action
            koboldcpp_core.save_config_to_db(DB_FILE, current_tuning_model_path_local, current_tuning_model_analysis_local, vram_at_decision_for_db, last_proposed_command_list_for_db, level_of_last_monitored_run, final_outcome_for_db_update, last_approx_vram_used_kcpp_mb)
            
            tuning_in_progress = False
            session_control_result = kcpp_control_loop_cli(port_for_webui, is_monitored_instance_being_controlled=False)
            return session_control_result
        else:
            print_warning("KCPP is not running (or was stopped). Cannot accept and use. Returning to tuning menu.")
            db_outcome_suffix_for_action = "_ATTEMPTED_USE_BUT_NOT_RUNNING_CLI"
    elif user_action_choice == 'l':
        db_outcome_suffix_for_action = "_USER_LAUNCHED_RISKY_CLI"
        final_outcome_for_db_relaunch = outcome_from_monitor + db_outcome_suffix_for_action
        launched_proc_risky = launch_kobold_for_use_cli(last_proposed_command_list_for_db, final_outcome_for_db_relaunch, level_for_db_record=level_of_last_monitored_run)
        if launched_proc_risky:
            effective_args_relaunch = get_effective_session_args(current_tuning_model_path_local, current_tuning_session_base_args)
            port_relaunch = effective_args_relaunch.get("--port", "5000")
            session_ctrl_res_relaunch = kcpp_control_loop_cli(port_relaunch)
            tuning_in_progress = False
            return session_ctrl_res_relaunch
        else: print_error("Risky re-launch failed. Returning to tuning menu.")

    if user_action_choice == 's':
        db_outcome_suffix_for_action = "_USER_SAVED_GOOD_GPU_CLI"
        if current_tuning_attempt_level > current_tuning_min_level: current_tuning_attempt_level -= 1
        else: print_warning("Already at Max GPU, cannot go further.")
    elif user_action_choice == 'g':
        db_outcome_suffix_for_action = "_USER_WANTS_MORE_GPU_CLI"
        if current_tuning_attempt_level > current_tuning_min_level: current_tuning_attempt_level -= 1
        else: print_warning("Already at Max GPU, cannot go further.")
    elif user_action_choice == 'c' or user_action_choice == 'a':
        if user_action_choice == 'a': db_outcome_suffix_for_action = "_USER_AUTO_ADJUST_CPU_CLI"
        else: db_outcome_suffix_for_action = "_USER_TRIED_CPU_AFTER_FAIL_CLI" if "FAIL" in outcome_from_monitor.upper() or "OOM" in outcome_from_monitor.upper() or "TIMEOUT" in outcome_from_monitor.upper() else "_USER_WANTS_MORE_CPU_CLI"
        if current_tuning_attempt_level < current_tuning_max_level: current_tuning_attempt_level += 1
        else: print_warning("Already at Max CPU, cannot go further.")
    elif user_action_choice == 'q':
        db_outcome_suffix_for_action = "_USER_RETURNED_MENU_CLI"

    if db_outcome_suffix_for_action:
        final_outcome_for_db_update_choice = outcome_from_monitor + db_outcome_suffix_for_action
        koboldcpp_core.save_config_to_db(DB_FILE, current_tuning_model_path_local, current_tuning_model_analysis_local, vram_at_decision_for_db, last_proposed_command_list_for_db, level_of_last_monitored_run, final_outcome_for_db_update_choice, last_approx_vram_used_kcpp_mb)
    
    return "continue_tuning"


def launch_kobold_for_use_cli(command_list_to_run: List[str], db_outcome_on_success: str, level_for_db_record: Optional[int] = None):
    global last_launched_process_info, gguf_file_global, current_model_analysis_global

    if last_launched_process_info["process_obj"] and last_launched_process_info["process_obj"].poll() is None:
        print_info(f"Stopping previously launched KCPP (PID: {last_launched_process_info['pid']})...")
        koboldcpp_core.kill_process(last_launched_process_info["pid"])
    last_launched_process_info = {"pid": None, "process_obj": None, "command_list": []}

    print_info(f"Launching KoboldCpp for use...")
    
    # Get actual HW VRAM for DB logging
    _, _, _, gpu_info_rich_direct_launch = koboldcpp_core.get_available_vram_mb(
        CONFIG,
        target_gpu_type=CONFIG.get("gpu_selection_mode", "auto") if CONFIG.get("gpu_selection_mode", "auto") != "auto" else None,
        target_gpu_index=CONFIG.get("selected_gpu_index", 0)
    )
    vram_before_this_launch = gpu_info_rich_direct_launch.get("free_mb", 0.0) # Actual HW free VRAM
    
    model_p_for_log = current_tuning_model_path_local if tuning_in_progress else gguf_file_global
    model_a_for_log = current_tuning_model_analysis_local if tuning_in_progress else current_model_analysis_global
    
    final_level_for_db_log = level_for_db_record if level_for_db_record is not None else \
                             (current_tuning_attempt_level if tuning_in_progress else 0)

    koboldcpp_core.save_config_to_db(DB_FILE, model_p_for_log, model_a_for_log, 
                                   vram_before_this_launch, command_list_to_run, 
                                   final_level_for_db_log, db_outcome_on_success, None)
    
    launched_kcpp_process, launch_err_msg = koboldcpp_core.launch_process(command_list_to_run, capture_output=False, new_console=True)

    if launch_err_msg or not launched_kcpp_process:
        print_error(f"Failed to launch KoboldCPP: {launch_err_msg or 'Unknown error'}")
        koboldcpp_core.save_config_to_db(DB_FILE, model_p_for_log, model_a_for_log, 
                                       vram_before_this_launch, command_list_to_run, 
                                       final_level_for_db_log, "LAUNCH_FOR_USE_FAILED_CLI", None)
        return None
    else:
        print_success(f"KoboldCpp launched in new console (PID: {launched_kcpp_process.pid}).")
        last_launched_process_info["pid"] = launched_kcpp_process.pid
        last_launched_process_info["process_obj"] = launched_kcpp_process
        last_launched_process_info["command_list"] = command_list_to_run
        
        if AUTO_OPEN_WEBUI:
            args_dict_from_cmd = koboldcpp_core.args_list_to_dict(command_list_to_run[1:] if len(command_list_to_run)>1 and (command_list_to_run[0].lower().endswith("python.exe") or command_list_to_run[0].lower().endswith("python")) else command_list_to_run)
            port_to_open_webui = args_dict_from_cmd.get("--port", CONFIG.get("default_args", {}).get("--port", "5000"))
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
                koboldcpp_core.kill_process(pid_to_control_val)
            if is_monitored_instance_being_controlled: kcpp_process_obj = None
            else: last_launched_process_info = {"pid": None, "process_obj": None, "command_list": []}
            return "new_gguf"
        elif control_choice == 'q':
            if process_object_to_control and pid_to_control_val:
                print_info(f"Stopping KCPP (PID: {pid_to_control_val}) and quitting launcher...")
                koboldcpp_core.kill_process(pid_to_control_val)
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

    if not gguf_file_global or not current_model_analysis_global.get('filepath'):
        print_error("No model selected or analyzed. Please select a model first.")
        return "new_gguf"

    tuning_in_progress = True
    current_tuning_model_path_local = gguf_file_global
    current_tuning_model_analysis_local = current_model_analysis_global.copy()
    current_tuning_session_base_args = get_effective_session_args(current_tuning_model_path_local, {})
    last_successful_monitored_run_details_cli = None

    print_title(f"Starting Auto-Tuning Session for: {os.path.basename(current_tuning_model_path_local)}")
    print_info(f"Analysis: Size ~{current_tuning_model_analysis_local.get('size_b', 'N/A')}B, "
               f"Quant ~{current_tuning_model_analysis_local.get('quant', 'N/A')}, "
               f"MoE: {'Yes' if current_tuning_model_analysis_local.get('is_moe') else 'No'}, "
               f"Layers: {current_tuning_model_analysis_local.get('num_layers', 'N/A')}, "
               f"Est. Full VRAM: {current_tuning_model_analysis_local.get('estimated_vram_gb_full_gpu', 'N/A')}GB")

    # Get current VRAM info (both budgeted and actual HW free VRAM)
    _, _, _, current_gpu_full_info = koboldcpp_core.get_available_vram_mb(
        CONFIG,
        target_gpu_type=CONFIG.get("gpu_selection_mode", "auto") if CONFIG.get("gpu_selection_mode", "auto") != "auto" else None,
        target_gpu_index=CONFIG.get("selected_gpu_index", 0)
    )
    current_budgeted_free_vram_mb = current_gpu_full_info.get("free_mb_budgeted", 0.0)
    current_actual_hw_free_vram_mb = current_gpu_full_info.get("free_mb", 0.0) # For historical lookup

    is_moe = current_tuning_model_analysis_local.get('is_moe', False)
    estimated_vram_needed_gb = current_tuning_model_analysis_local.get('estimated_vram_gb_full_gpu', 0)
    estimated_vram_needed_mb = float(estimated_vram_needed_gb) * 1024 if isinstance(estimated_vram_needed_gb, (int,float)) else 0.0

    if is_moe:
        current_tuning_min_level, current_tuning_max_level, initial_heuristic_level = -25, 10, -10
    else:
        current_tuning_min_level, current_tuning_max_level = -17, 9
        size_b_val = current_tuning_model_analysis_local.get('size_b', 0)
        size_b_float = 0.0
        if isinstance(size_b_val, (int,float)): size_b_float = float(size_b_val)
        if size_b_float >= 30: initial_heuristic_level = -3
        elif size_b_float >= 20: initial_heuristic_level = -5
        else: initial_heuristic_level = -7
    
    safety_buffer_mb_cfg = CONFIG.get("vram_safety_buffer_mb", 768)
    min_free_after_load_mb_cfg = CONFIG.get("min_vram_free_after_load_success_mb", 512)
    # Effective VRAM budget for heuristic is based on budgeted total VRAM
    effective_vram_budget_for_heuristic_mb = current_gpu_full_info.get("total_mb_budgeted", 0.0) - safety_buffer_mb_cfg - min_free_after_load_mb_cfg


    if estimated_vram_needed_mb > 0 and current_budgeted_free_vram_mb > 0 :
        if estimated_vram_needed_mb > effective_vram_budget_for_heuristic_mb * 1.1:
            initial_heuristic_level = max(initial_heuristic_level, -3 if not is_moe else -6)
            print_info(f"Heuristic: Estimated VRAM ({estimated_vram_needed_mb:.0f}MB) exceeds budget ({effective_vram_budget_for_heuristic_mb:.0f}MB). Adjusting OT towards CPU.")
        elif estimated_vram_needed_mb < effective_vram_budget_for_heuristic_mb * 0.7:
            initial_heuristic_level = min(initial_heuristic_level, -12 if not is_moe else -18)
            print_info(f"Heuristic: Ample VRAM budget. Adjusting OT towards GPU.")

    # Use actual hardware free VRAM for historical lookup comparison
    best_historical_config = koboldcpp_core.find_best_historical_config(DB_FILE, current_tuning_model_analysis_local, current_actual_hw_free_vram_mb, CONFIG)

    if best_historical_config and "attempt_level" in best_historical_config:
        print_info(f"Found historical config. Level: {best_historical_config['attempt_level']}, Outcome: {best_historical_config['outcome']}")
        hist_lvl, hist_outcome_str = best_historical_config['attempt_level'], best_historical_config.get('outcome', "")
        approx_hist_vram_used = best_historical_config.get('approx_vram_used_kcpp_mb') # This is actual KCPP usage

        if approx_hist_vram_used is not None and (float(approx_hist_vram_used) + safety_buffer_mb_cfg < current_actual_hw_free_vram_mb):
            initial_heuristic_level = max(current_tuning_min_level, hist_lvl -1 if hist_lvl > current_tuning_min_level else hist_lvl)
            print_info(f"Historical success used {approx_hist_vram_used:.0f}MB (actual) which fits current actual VRAM ({current_actual_hw_free_vram_mb:.0f}MB). Starting near historical level: {initial_heuristic_level}")
        elif hist_outcome_str.startswith("SUCCESS_LOAD_VRAM_OK") or hist_outcome_str.startswith("SUCCESS_USER_CONFIRMED") or hist_outcome_str.endswith("_USER_SAVED_GOOD_GPU_CLI"):
            initial_heuristic_level = max(current_tuning_min_level, hist_lvl - 1 if hist_lvl > current_tuning_min_level else hist_lvl)
        elif hist_outcome_str.endswith("_USER_AUTO_ADJUST_CPU_CLI") or hist_outcome_str.endswith("_USER_TRIED_CPU_AFTER_FAIL_CLI") or \
             "OOM" in hist_outcome_str.upper() or "TIGHT" in hist_outcome_str.upper():
             initial_heuristic_level = min(current_tuning_max_level, hist_lvl + 1 if hist_lvl < current_tuning_max_level else hist_lvl)
        else: initial_heuristic_level = hist_lvl
        
        remembered_args_list_from_db = best_historical_config.get("args_list", [])
        if remembered_args_list_from_db:
            remembered_args_dict_parsed = koboldcpp_core.args_list_to_dict(remembered_args_list_from_db)
            remembered_args_dict_parsed.pop("--model", None); remembered_args_dict_parsed.pop("--overridetensors", None)
            current_tuning_session_base_args.update(remembered_args_dict_parsed)
            print_info(f"Applied remembered arguments to current session base. OT Level target adjusted to: {initial_heuristic_level}")
    else:
        print_info(f"No suitable historical config found. Starting with heuristic OT Level: {initial_heuristic_level}")

    current_tuning_attempt_level = max(current_tuning_min_level, min(initial_heuristic_level, current_tuning_max_level))
    level_of_last_monitored_run = current_tuning_attempt_level

    while tuning_in_progress:
        print("\n" + "=" * 70)
        current_tuning_attempt_level = max(current_tuning_min_level, min(current_tuning_attempt_level, current_tuning_max_level))
        ot_string_generated = koboldcpp_core.generate_overridetensors(current_tuning_model_analysis_local, current_tuning_attempt_level)
        strategy_description = koboldcpp_core.get_offload_description(current_tuning_model_analysis_local, current_tuning_attempt_level, ot_string_generated)
        gpu_layers_for_level = koboldcpp_core.get_gpu_layers_for_level(current_tuning_model_analysis_local, current_tuning_attempt_level)
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
            strategy_table.add_row("OT Regex", ot_string_generated or "None (Max GPU layers if --gpulayers is high)")
            strategy_table.add_row("Effective GPU Layers", f"{gpu_layers_for_level}/{total_model_layers}")
            console.print(strategy_table)
        else:
            print_info(f"Model: {os.path.basename(current_tuning_model_path_local)}")
            print(f"🛠️ OT Level: {current_tuning_attempt_level}\n   Range: {current_tuning_min_level}=MaxGPU to {current_tuning_max_level}={'SuperMaxCPU' if current_tuning_model_analysis_local.get('is_moe') else 'MaxCPU'}\n   Strategy: {strategy_description}\n   Regex: {(ot_string_generated or 'None')}\n   GPU Layers: {gpu_layers_for_level}/{total_model_layers}")

        args_for_kcpp_display_list = koboldcpp_core.build_command(current_tuning_model_path_local, ot_string_generated, current_tuning_model_analysis_local, current_tuning_session_base_args)
        display_full_command_list = koboldcpp_core.get_command_to_run(KOBOLDCPP_EXECUTABLE, args_for_kcpp_display_list)
        
        # Use rich GPU info for display
        _, _, vram_info_message_str, current_gpu_rich_info = koboldcpp_core.get_available_vram_mb(
             CONFIG,
             target_gpu_type=CONFIG.get("gpu_selection_mode", "auto") if CONFIG.get("gpu_selection_mode", "auto") != "auto" else None,
             target_gpu_index=CONFIG.get("selected_gpu_index", 0)
        )
        if dependencies['rich']['module']:
            console.print(Panel(f"{vram_info_message_str}", title="Current GPU Info", style="green" if current_gpu_rich_info.get("success") else "red", expand=False))
        else: print(f"    GPU Status: {vram_info_message_str}")
        
        print_title("Proposed Command for This OT Level"); print_command(koboldcpp_core.format_command_for_display(display_full_command_list))
        
        menu_options_text = "(L)aunch & Monitor | (S)kip Tune & Launch Now | (G)PU More (↓Lvl) | (C)PU More (↑Lvl) | (E)dit Session Args | (P)ermanent Model Args | (H)istory (This Model) | (N)ew GGUF | (Q)uit Tuning"
        print_title("Tuning Actions"); print(menu_options_text)
        user_tuning_choice = prompt("Your choice", choices=['l','s','g','c','e', 'p', 'h','n','q'], default='l').lower().strip()

        if user_tuning_choice == 'l':
            post_monitoring_action_result = launch_and_monitor_for_tuning_cli()
            if post_monitoring_action_result == "quit_script_leave_running":
                tuning_in_progress = False; return "quit_script_leave_running"
            elif post_monitoring_action_result == "new_gguf":
                tuning_in_progress = False; return "new_gguf"
            elif post_monitoring_action_result == "quit_script":
                tuning_in_progress = False; return "quit_script"
        elif user_tuning_choice == 's':
            print_info("Skipping further tuning, launching current configuration directly...")
            launched_kcpp_proc = launch_kobold_for_use_cli(display_full_command_list, "SUCCESS_USER_DIRECT_LAUNCH_CLI", level_for_db_record=current_tuning_attempt_level)
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
                 print_info("Permanent arguments were changed via editor. Session overrides reset, effective base updated.")
        elif user_tuning_choice == 'p':
            print_info("Opening argument editor. Use the 'P' option inside the editor to save changes permanently for this model.")
            _, permanent_save_made_here = edit_current_args_interactive_cli(current_tuning_model_path_local, current_tuning_session_base_args)
            if permanent_save_made_here:
                 current_tuning_session_base_args = get_effective_session_args(current_tuning_model_path_local, {})
                 print_info("Permanent arguments changed. Session overrides reset, effective base updated.")
        elif user_tuning_choice == 'h':
            view_db_history_cli(model_filepath_filter=current_tuning_model_path_local)
        elif user_tuning_choice == 'n':
            tuning_in_progress = False; return "new_gguf"
        elif user_tuning_choice == 'q':
            tuning_in_progress = False; return "new_gguf"
        else: print_error("Invalid input.")
    
    tuning_in_progress = False
    return "new_gguf"


def main_cli():
    global CONFIG, KOBOLDCPP_EXECUTABLE, DB_FILE, DEFAULT_GGUF_DIR, AUTO_OPEN_WEBUI
    global VRAM_SAFETY_BUFFER_MB, MIN_VRAM_FREE_AFTER_LOAD_MB, LOADING_TIMEOUT_SECONDS
    global KOBOLD_SUCCESS_PATTERN, OOM_ERROR_KEYWORDS
    global gguf_file_global, current_model_analysis_global, last_gguf_directory, last_launched_process_info

    core_init_data = koboldcpp_core.initialize_launcher()
    CONFIG = core_init_data["config"]

    KOBOLDCPP_EXECUTABLE = CONFIG["koboldcpp_executable"]
    DB_FILE = CONFIG["db_file"]
    DEFAULT_GGUF_DIR = CONFIG.get("default_gguf_dir", "")
    AUTO_OPEN_WEBUI = CONFIG.get("auto_open_webui", True)
    VRAM_SAFETY_BUFFER_MB = CONFIG.get("vram_safety_buffer_mb", 768)
    MIN_VRAM_FREE_AFTER_LOAD_MB = CONFIG.get("min_vram_free_after_load_success_mb", 512)
    LOADING_TIMEOUT_SECONDS = CONFIG.get("loading_timeout_seconds", 60)
    KOBOLD_SUCCESS_PATTERN = CONFIG.get("kobold_success_pattern", r"Starting Kobold API on port (\d+)")
    OOM_ERROR_KEYWORDS = [k.lower() for k in CONFIG.get("oom_error_keywords", [])]
    last_gguf_directory = CONFIG.get("last_used_gguf_dir", "")

    print_info(f"Using configuration file: {koboldcpp_core.CONFIG_FILE}")
    print_info(f"Using database file: {DB_FILE}")

    if not core_init_data["initialized"]:
        if not core_init_data["config_loaded"]: print_warning(f"Config issue: {core_init_data['config_message']}")
        if not core_init_data["db_success"]: print_warning(f"DB issue: {core_init_data['db_message']}")

    if not CONFIG.get("first_run_completed", False):
        if not handle_first_run_prompts_cli(CONFIG):
            print_error("Initial setup failed. Exiting."); sys.exit(1)
        # Re-populate after first run, as CONFIG might have changed
        KOBOLDCPP_EXECUTABLE = CONFIG["koboldcpp_executable"]
        DEFAULT_GGUF_DIR = CONFIG.get("default_gguf_dir", "")
        AUTO_OPEN_WEBUI = CONFIG.get("auto_open_webui", True)

    core_version_display = CONFIG.get('launcher_core_version', 'N/A')
    print_title(f"TensorTune v{LAUNCHER_CLI_VERSION} (Core: {core_version_display})")
    sys_info_data = core_init_data.get("system_info", {})
    gpu_info_data_rich = core_init_data.get("gpu_info", {}) # This is the rich dict
    kcpp_caps = core_init_data.get("koboldcpp_capabilities", {})

    print_info(f"OS: {sys_info_data.get('os_name','N/A')} {sys_info_data.get('os_version','N/A')} | Python: {sys_info_data.get('python_version','N/A').split()[0]}")
    
    gpu_message_to_display = gpu_info_data_rich.get('message', 'Could not detect GPU details.')
    if gpu_info_data_rich.get("success"):
        print_info(f"GPU Info: {gpu_message_to_display}")
        if gpu_info_data_rich.get("override_active"):
            print_info(f"  VRAM Budget Override Active: Total {gpu_info_data_rich.get('total_mb_budgeted',0):.0f}MB")
    else: print_warning(f"GPU Info: {gpu_message_to_display}")
    
    if "error" in kcpp_caps: print_warning(f"KCPP Caps Error: {kcpp_caps['error']}")
    else: print_info(f"KCPP Caps: CUDA:{kcpp_caps.get('cuda', False)}, ROCm:{kcpp_caps.get('rocm',False)}, FlashAttn:{kcpp_caps.get('flash_attn',False)}, OverrideTensors:{kcpp_caps.get('overridetensors',False)}")

    # Verify KCPP executable path
    if not os.path.exists(KOBOLDCPP_EXECUTABLE):
        resolved_exe_path = shutil.which(KOBOLDCPP_EXECUTABLE) or shutil.which(os.path.basename(KOBOLDCPP_EXECUTABLE))
        if resolved_exe_path:
            print_info(f"KoboldCpp executable '{KOBOLDCPP_EXECUTABLE}' updated via PATH to: {resolved_exe_path}")
            KOBOLDCPP_EXECUTABLE = os.path.abspath(resolved_exe_path)
            CONFIG["koboldcpp_executable"] = KOBOLDCPP_EXECUTABLE
            koboldcpp_core.save_launcher_config(CONFIG)
        elif KOBOLDCPP_EXECUTABLE.lower().endswith(".py"):
            try:
                launcher_dir = os.path.dirname(os.path.abspath(__file__)) if not getattr(sys, 'frozen', False) else os.path.dirname(sys.executable)
                potential_relative_path = os.path.join(launcher_dir, KOBOLDCPP_EXECUTABLE)
                if os.path.exists(potential_relative_path):
                    KOBOLDCPP_EXECUTABLE = os.path.abspath(potential_relative_path)
                    CONFIG["koboldcpp_executable"] = KOBOLDCPP_EXECUTABLE
                    koboldcpp_core.save_launcher_config(CONFIG)
                    print_info(f"Using KoboldCpp Python script found relative to launcher: {KOBOLDCPP_EXECUTABLE}")
                else:
                    print_error(f"FATAL: KoboldCpp target '{KOBOLDCPP_EXECUTABLE}' not found and not in PATH or relative to launcher. Please check the path."); sys.exit(1)
            except NameError:
                 print_error(f"FATAL: Could not determine launcher script directory for '{KOBOLDCPP_EXECUTABLE}'. Path check failed."); sys.exit(1)
        else:
             print_error(f"FATAL: KoboldCpp target '{KOBOLDCPP_EXECUTABLE}' not found and not in PATH. Please check the path."); sys.exit(1)

    while True:
        gguf_selection_result = select_gguf_file_cli()
        if gguf_selection_result is None:
            break 
        if gguf_selection_result == "main_menu":
            continue

        gguf_file_global = gguf_selection_result
        current_model_analysis_global = koboldcpp_core.analyze_filename(gguf_file_global)
        
        tuning_session_outcome = run_model_tuning_session_cli() 

        if tuning_session_outcome == "quit_script":
            break
        if tuning_session_outcome == "quit_script_leave_running":
            print_info("Exiting launcher. KoboldCpp may still be running as per user choice."); return
    
    print_title("TensorTune Finished")


if __name__ == "__main__":
    if dependencies['pynvml']['module'] is None and platform.system() == "Windows":
        print_warning("PyNVML (for NVIDIA GPU monitoring) is not installed. NVIDIA GPU VRAM info might be limited.")
        print_warning("You can install it with: pip install pynvml")

    koboldcpp_core.atexit.register(koboldcpp_core._cleanup_nvml) # Ensure NVML is shutdown

    try:
        main_cli()
    except KeyboardInterrupt:
        print_warning("\nLauncher terminated by user (Ctrl+C).")
    except Exception as e_fatal_script:
        print_error(f"\nFATAL SCRIPT ERROR: {type(e_fatal_script).__name__}: {e_fatal_script}")
        import traceback
        traceback.print_exc()
    finally:
        print_info("Exiting. Cleaning up any lingering KoboldCpp process launched by this session...")
        if last_launched_process_info.get("process_obj") and last_launched_process_info["process_obj"].poll() is None:
            if last_launched_process_info.get("pid"):
                print_info(f"Stopping last directly launched KCPP process (PID: {last_launched_process_info['pid']})...")
                koboldcpp_core.kill_process(last_launched_process_info["pid"], force=True)

        if not CONFIG: # If main_cli didn't run enough to populate CONFIG
            _temp_conf, _, _ = koboldcpp_core.load_config() 
            _kcpp_exe_for_cleanup = _temp_conf.get("koboldcpp_executable", "")
        else:
            _kcpp_exe_for_cleanup = KOBOLDCPP_EXECUTABLE

        if _kcpp_exe_for_cleanup:
            kcpp_exe_basename = os.path.basename(_kcpp_exe_for_cleanup)
            print_info(f"Performing cleanup sweep for processes like '{kcpp_exe_basename}'...")
            koboldcpp_core.kill_processes_by_name(kcpp_exe_basename)
            if kcpp_exe_basename.lower().endswith(".py"):
                 koboldcpp_core.kill_processes_by_name("python", cmdline_substr_filter=kcpp_exe_basename)
        print_info("Launcher exited.")