#!/usr/bin/env python3
# KoboldCpp Smart Launcher - VRAM Auto-Tuning Edition
# Version 11.2.0 (CLI - Aligned with GUI features)

import json
import sys
import os
import subprocess
import re
import time
import threading
import signal 
import sqlite3
from datetime import datetime, timezone
import pathlib
import webbrowser
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import platform
from pathlib import Path

# Import the core script
import koboldcpp_core

# Handle dependency imports with helpful error messages
dependencies = {
    'rich': {'required': False, 'module': None, 'purpose': 'improved terminal UI'},
    'psutil': {'required': False, 'module': None, 'purpose': 'system resource monitoring'},
    'pynvml': {'required': False, 'module': None, 'purpose': 'NVIDIA GPU VRAM detection'},
    'tkinter': {'required': False, 'module': None, 'purpose': 'file open dialog'}
}

for dep_name, dep_info in dependencies.items():
    try:
        if dep_name == 'tkinter':
            import tkinter as _tk_
            from tkinter import filedialog as _filedialog_
            dependencies[dep_name]['module'] = 'imported_for_filedialog'
        else:
            dependencies[dep_name]['module'] = __import__(dep_name)
    except ImportError:
        if dep_info['required']:
            print(f"ERROR: Required dependency '{dep_name}' not found. Please install with: pip install {dep_name}")
            print(f"Purpose: {dep_info['purpose']}")
            sys.exit(1)

if dependencies['rich']['module']:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
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
    def prompt(text, choices=None, default=None):
        if choices: return Prompt.ask(text, choices=choices, default=default)
        return Prompt.ask(text, default=default)
    def confirm(text, default=True): return Confirm.ask(text, default=default)
else:
    def print_title(text): print(f"\n{'='*10} {text} {'='* (50-len(text) if len(text) < 50 else 0)}\n")
    def print_success(text): print(f"✓ {text}")
    def print_error(text): print(f"✗ {text}")
    def print_warning(text): print(f"! {text}")
    def print_info(text): print(f"ℹ {text}")
    def print_command(text): print(f"\n```bash\n{text}\n```\n")
    def prompt(text, choices=None, default=None):
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

psutil = dependencies['psutil']['module']
pynvml_module = dependencies['pynvml']['module']

CONFIG: Dict[str, Any] = {}
KOBOLDCPP_EXECUTABLE = ""
DB_FILE = ""
DEFAULT_GGUF_DIR = ""
AUTO_OPEN_WEBUI = True
VRAM_SAFETY_BUFFER_MB = 0
MIN_VRAM_FREE_AFTER_LOAD_MB = 0
LOADING_TIMEOUT_SECONDS = 60
KOBOLD_SUCCESS_PATTERN = ""
OOM_ERROR_KEYWORDS: List[str] = []
LAUNCHER_CLI_VERSION = "11.2.0" 

last_gguf_directory = ""
last_launched_process_info: Dict[str, Any] = {"pid": None, "process_obj": None, "command_list": []}
gguf_file_global = ""
current_model_analysis_global: Dict[str, Any] = {}

tuning_in_progress = False
current_tuning_attempt_level = 0
current_tuning_min_level = 0
current_tuning_max_level = 0
current_tuning_session_base_args: Dict[str, Any] = {}
current_tuning_model_path_local = ""
current_tuning_model_analysis_local: Dict[str, Any] = {}
last_proposed_command_list_for_db: List[str] = []
vram_at_decision_for_db: Optional[float] = None
last_approx_vram_used_kcpp_mb: Optional[float] = None

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

    print_title(" KoboldCpp Launcher Initial Setup ")
    if not config_dict.get('first_run_intro_shown', False):
        print("""
Welcome to the KoboldCpp Smart Launcher CLI!
This tool helps you manage and launch KoboldCpp with auto-tuned settings.
We need to configure a few things for the first run.
        """)
        config_dict['first_run_intro_shown'] = True

    import shutil
    detected_exe = None
    default_exe_name = "koboldcpp.exe" if platform.system() == "Windows" else "./koboldcpp"
    if os.path.exists(default_exe_name):
        detected_exe = os.path.abspath(default_exe_name)
    elif shutil.which("koboldcpp"):
        detected_exe = shutil.which("koboldcpp")
    
    current_exe_in_config = config_dict.get("koboldcpp_executable", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["koboldcpp_executable"])
    default_prompt_exe = detected_exe or current_exe_in_config
    
    if detected_exe: print_success(f"Auto-detected KoboldCpp candidate: {detected_exe}")
    exe_prompt_msg = f"Enter path to KoboldCpp executable/script"
    user_exe = prompt(exe_prompt_msg, default=default_prompt_exe).strip()
    while not os.path.exists(user_exe):
        found_in_path = shutil.which(user_exe)
        if found_in_path:
            user_exe = os.path.abspath(found_in_path)
            print_success(f"Found '{os.path.basename(user_exe)}' in PATH: {user_exe}")
            break
        print_error(f"Path '{user_exe}' not found or not executable. Please enter a valid path.")
        user_exe = prompt("Re-enter KoboldCpp path:", default=default_prompt_exe).strip()
    config_dict["koboldcpp_executable"] = os.path.abspath(user_exe)

    current_gguf_dir = config_dict.get("default_gguf_dir", os.getcwd())
    user_gguf_dir = prompt("Enter default GGUF model directory (or '.' for current, blank for none)",
                           default=current_gguf_dir if current_gguf_dir and os.path.isdir(current_gguf_dir) else os.getcwd()).strip()
    if user_gguf_dir == ".": user_gguf_dir = os.getcwd()
    if user_gguf_dir and not os.path.isdir(user_gguf_dir):
        print_warning(f"Directory '{user_gguf_dir}' invalid. No default GGUF dir set.")
        config_dict["default_gguf_dir"] = ""
    else:
        config_dict["default_gguf_dir"] = os.path.abspath(user_gguf_dir) if user_gguf_dir else ""

    db_file_from_core = config_dict.get("db_file", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["db_file"])
    print_info(f"History database will be stored at: {os.path.abspath(db_file_from_core)}")

    current_port = config_dict.get("default_args", {}).get("--port", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"]["--port"])
    while True:
        user_port_str = prompt(f"Default KoboldCpp port?", default=str(current_port))
        try:
            user_port_int = int(user_port_str)
            if 1 <= user_port_int <= 65535:
                if "default_args" not in config_dict: config_dict["default_args"] = {}
                config_dict["default_args"]["--port"] = str(user_port_int); break
            else: print_error("Port must be between 1 and 65535.")
        except ValueError: print_error("Invalid port number.")

    config_dict["auto_open_webui"] = confirm("Auto-open Web UI in browser after successful launch?", default=config_dict.get("auto_open_webui", True))

    config_dict['first_run_completed'] = True
    success, msg = koboldcpp_core.save_launcher_config(config_dict)
    if success: print_success("\nInitial setup complete! Configuration saved.")
    else: print_error(f"\nFailed to save initial configuration: {msg}")
    return success

def select_gguf_file_cli() -> Optional[str]:
    global last_gguf_directory, CONFIG, DEFAULT_GGUF_DIR
    print_title("Select GGUF Model")
    
    main_menu_actions = {
        "s": "Select GGUF Model File",
        "v": "View Global Launch History",
        "q": "Quit Launcher"
    }
    print("Main Menu Options:")
    for key, desc in main_menu_actions.items():
        print(f"  ({key.upper()}) {desc}")
    
    while True:
        action_choice = prompt("Your choice", choices=list(main_menu_actions.keys()), default="s").lower()
        if action_choice == 'q': return None
        if action_choice == 'v':
            view_db_history_cli()
            continue
        if action_choice == 's':
            break

    if dependencies['tkinter']['module']:
        try:
            root = _tk_.Tk(); root.withdraw(); root.attributes('-topmost', True)
            start_dir = last_gguf_directory or DEFAULT_GGUF_DIR or os.getcwd()
            if not os.path.isdir(start_dir): start_dir = os.getcwd()
            filepath = _filedialog_.askopenfilename(title="Select GGUF Model File",
                                                  filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")],
                                                  initialdir=start_dir)
            root.destroy()
            if filepath:
                last_gguf_directory = os.path.dirname(filepath)
                CONFIG["last_used_gguf_dir"] = last_gguf_directory
                koboldcpp_core.save_launcher_config(CONFIG)
                print_success(f"Selected: {os.path.basename(filepath)}")
                return os.path.abspath(filepath)
            print_info("File selection cancelled via dialog.")
            return "main_menu" 
        except Exception as e:
            print_warning(f"Tkinter file dialog failed: {e}. Falling back to manual input.")

    default_dir_prompt = last_gguf_directory or DEFAULT_GGUF_DIR or os.getcwd()
    while True:
        filepath_manual = prompt(f"Enter full path to GGUF model file (or press Enter to cancel and return to main menu)\n(Searches in: {default_dir_prompt} if relative path is given)").strip()
        if not filepath_manual:
            print_info("File selection cancelled via manual input.")
            return "main_menu"

        if os.path.isabs(filepath_manual):
            potential_path = filepath_manual
        else:
            potential_path = os.path.join(default_dir_prompt, filepath_manual)

        if os.path.isfile(potential_path) and potential_path.lower().endswith(".gguf"):
            abs_path = os.path.abspath(potential_path)
            last_gguf_directory = os.path.dirname(abs_path)
            CONFIG["last_used_gguf_dir"] = last_gguf_directory
            print_success(f"Selected: {os.path.basename(abs_path)}")
            return abs_path
        print_error(f"Path '{potential_path}' is not a valid .gguf file. Please try again.")

def view_db_history_cli(model_filepath_filter: Optional[str] = None):
    if model_filepath_filter:
        print_info(f"Loading history for model: {os.path.basename(model_filepath_filter)} from: {DB_FILE}")
    else:
        print_info(f"Loading global history from: {DB_FILE}")
        
    all_history_entries = koboldcpp_core.get_history_entries(DB_FILE, limit=100)
    
    if not all_history_entries:
        print_info("No history records found.")
        return

    filtered_entries = []
    if model_filepath_filter:
        for entry in all_history_entries:
            if entry[0] == model_filepath_filter:
                filtered_entries.append(entry)
        if not filtered_entries:
            print_info(f"No history records found for model: {os.path.basename(model_filepath_filter)}")
            return
        display_entries = filtered_entries[:20]
        title = f"Launch History for {os.path.basename(model_filepath_filter)} (Up to 20 most recent)"
    else:
        display_entries = all_history_entries[:20]
        title = "Global Launch History (Up to 20 most recent)"

    if dependencies['rich']['module']:
        table = Table(title=title)
        cols = ["Model", "Size", "Quant", "MoE", "VRAM@L", "OT Lvl", "Outcome", "VRAM Used", "Timestamp", "Key Args"]
        styles = ["cyan", "magenta", None, "magenta", "green", "yellow", "blue", "green", "dim", "yellow"]
        justifies = ["left", "right", "center", "center", "right", "center", "left", "right", "left", "left"]
        for col, style, justify in zip(cols, styles, justifies):
            table.add_column(col, style=style, justify=justify, overflow="fold")

        for r_data in display_entries:
            model_name = os.path.basename(r_data[0])
            size_b = f"{r_data[1]:.1f}B" if isinstance(r_data[1], float) else (str(r_data[1]) + "B" if r_data[1] else "N/A")
            quant = str(r_data[2]) if r_data[2] else "N/A"
            is_moe = "Y" if r_data[3] else "N"
            vram_at_l = f"{r_data[4]}MB" if r_data[4] is not None else "N/A"
            ot_lvl = str(r_data[5]) if r_data[5] is not None else "N/A"
            outcome = str(r_data[6]) if r_data[6] else "N/A"
            vram_used = f"{r_data[7]}MB" if r_data[7] is not None else "N/A"
            ts_obj = r_data[8]
            timestamp_str = ts_obj.strftime('%y-%m-%d %H:%M') if isinstance(ts_obj, datetime) else str(ts_obj)[:16]
            key_args_str = "N/A" 
            table.add_row(model_name, size_b, quant, is_moe, vram_at_l, ot_lvl, outcome, vram_used, timestamp_str, key_args_str)
        console.print(table)
    else: 
        print_title(title)
        header = f"{'Model':<28} | {'Sz':<5} | {'Quant':<9} | {'MoE':<3} | {'VRAM@L':<7} | {'Lvl':<3} | {'Outcome':<38} | {'VRAMUsed':<8} | {'Timestamp':<16}"
        print(header); print("-" * len(header))
        for r_data in display_entries:
            model_name = os.path.basename(r_data[0])[:26]
            size_b = f"{r_data[1]:.1f}" if isinstance(r_data[1], float) else (str(r_data[1]) if r_data[1] else "N/A")
            quant_str = (str(r_data[2]) if r_data[2] else "N/A")[:9]
            is_moe_str = "Y" if r_data[3] else "N"
            vram_at_l_str = str(r_data[4]) if r_data[4] is not None else "N/A"
            ot_lvl_str = str(r_data[5]) if r_data[5] is not None else "N/A"
            outcome_str = (str(r_data[6]) if r_data[6] else "N/A")[:38]
            vram_used_str = str(r_data[7]) if r_data[7] is not None else "N/A"
            ts_obj = r_data[8]
            timestamp_s = ts_obj.strftime('%y-%m-%d %H:%M') if isinstance(ts_obj, datetime) else str(ts_obj)[:16]
            print(f"{model_name:<28} | {size_b:<5} | {quant_str:<9} | {is_moe_str:<3} | {vram_at_l_str:<7} | {ot_lvl_str:<3} | {outcome_str:<38} | {vram_used_str:<8} | {timestamp_s:<16}")

def get_effective_session_args(model_file_path: Optional[str], session_overrides: Dict[str, Any]) -> Dict[str, Any]:
    effective_args = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
    effective_args.update(CONFIG.get("default_args", {}))
    if model_file_path and "model_specific_args" in CONFIG:
        model_specifics = CONFIG["model_specific_args"].get(model_file_path, {})
        effective_args.update(model_specifics)
    effective_args.update(session_overrides)
    return effective_args

def edit_current_args_interactive_cli(model_path_for_specifics: Optional[str], current_session_args_dict: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], bool]:
    permanent_args_changed = False
    arg_info_cli = {
        "--threads": {"help": "CPU threads. 'auto' or number.", "type": "str"},
        "--nblas": {"help": "BLAS threads. 'auto' or number.", "type": "str"},
        "--contextsize": {"help": "Context tokens (e.g., 4096, 16384).", "type": "int_str"},
        "--promptlimit": {"help": "Max prompt tokens (e.g. 4000).", "type": "int_str"},
        "--usecublas": {"help": "Use CUBLAS (NVIDIA). (true/false)", "type": "bool"},
        "--flashattention": {"help": "Use FlashAttention. (true/false)", "type": "bool"},
        "--nommap": {"help": "Disable memory mapping. (true/false)", "type": "bool"},
        "--lowvram": {"help": "Enable low VRAM mode. (true/false)", "type": "bool"},
        "--port": {"help": "Network port (e.g., 5000).", "type": "int_str"},
        "--defaultgenamt": {"help": "Default tokens to generate (e.g., 2048).", "type": "int_str"},
        "--gpulayers": {"help": "GPU layers. 'auto', 'off', or number (e.g. 35, 999 for max).", "type": "str"},
        "--quantkv": {"help": "K/V cache quant. 'auto', 'off', or number (0=f32, 1=Q8_0).", "type": "str"},
        "--blasbatchsize": {"help": "BLAS batch size. 'auto', 'off', or number (e.g. 128, 512).", "type": "str"}
    }
    # Ensure editable_arg_keys only contains args present in the core defaults template for consistency
    editable_arg_keys = sorted([k for k in arg_info_cli.keys() if k in koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"]])
    
    while True:
        print_title("Edit Base Startup Arguments")
        effective_display_args = get_effective_session_args(model_path_for_specifics, current_session_args_dict)
        idx_map = {}

        if dependencies['rich']['module']:
            table = Table(title="Effective Arguments for This Session")
            table.add_column("#", style="cyan"); table.add_column("Argument", style="green");
            table.add_column("Current Value", style="yellow"); table.add_column("Description");
            for i, arg_name in enumerate(editable_arg_keys):
                idx_str = str(i + 1); idx_map[idx_str] = arg_name
                val_disp = effective_display_args.get(arg_name)
                val_str = "ENABLED" if val_disp is True else ("DISABLED" if val_disp is False else (str(val_disp) if val_disp is not None else "NOT SET (KCPP default)"))
                table.add_row(idx_str, arg_name, val_str, arg_info_cli[arg_name]['help'])
            console.print(table)
        else:
            print("Effective Arguments for This Session:")
            for i, arg_name in enumerate(editable_arg_keys):
                idx_str = str(i + 1); idx_map[idx_str] = arg_name
                val_disp = effective_display_args.get(arg_name)
                val_str = "ENABLED" if val_disp is True else ("DISABLED" if val_disp is False else (str(val_disp) if val_disp is not None else "NOT SET (KCPP default)"))
                print(f"  ({idx_str}) {arg_name:<18}: {val_str:<15} - {arg_info_cli[arg_name]['help']}")

        print("\nActions: (number) to edit | (T#)oggle bool | (P)ermanently save current args for this model")
        print("         (S)ave session changes & Continue | (X) Cancel session edits & Continue")
        choice = prompt("Your choice:", default="s").lower().strip()

        if choice == 's': return current_session_args_dict, permanent_args_changed
        if choice == 'x': return None, permanent_args_changed

        if choice == 'p':
            if not model_path_for_specifics:
                print_error("No model selected to save permanent args for. Select a model first."); continue
            
            print_info(f"Current effective arguments for '{os.path.basename(model_path_for_specifics)}' will be saved as its new defaults.")
            if confirm(f"Save these as permanent defaults for {os.path.basename(model_path_for_specifics)}?", default=True):
                if "model_specific_args" not in CONFIG: CONFIG["model_specific_args"] = {}
                if model_path_for_specifics not in CONFIG["model_specific_args"]:
                    CONFIG["model_specific_args"][model_path_for_specifics] = {}
                
                model_perms_to_set = CONFIG["model_specific_args"][model_path_for_specifics]
                global_baseline_args = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
                global_baseline_args.update(CONFIG.get("default_args", {}))

                for arg_key in editable_arg_keys:
                    session_val_candidate = effective_display_args.get(arg_key)
                    global_default_val = global_baseline_args.get(arg_key)
                    
                    if session_val_candidate is None:
                        if arg_key in model_perms_to_set: del model_perms_to_set[arg_key]
                    elif session_val_candidate != global_default_val:
                        model_perms_to_set[arg_key] = session_val_candidate
                    else: 
                        if arg_key in model_perms_to_set: del model_perms_to_set[arg_key]
                
                if not CONFIG["model_specific_args"][model_path_for_specifics]:
                    del CONFIG["model_specific_args"][model_path_for_specifics]
                
                koboldcpp_core.save_launcher_config(CONFIG)
                permanent_args_changed = True
                print_success(f"Permanent args saved for {os.path.basename(model_path_for_specifics)}.")
                current_session_args_dict.clear() 
            else: print_info("Permanent edit cancelled.")
            continue

        arg_to_edit_key = None; is_toggle = False
        if choice.startswith('t') and choice[1:].isdigit():
            raw_idx = choice[1:]
            if raw_idx in idx_map: arg_to_edit_key = idx_map[raw_idx]; is_toggle = True
        elif choice.isdigit():
            if choice in idx_map: arg_to_edit_key = idx_map[choice]

        if arg_to_edit_key:
            arg_type_info = arg_info_cli[arg_to_edit_key]['type']
            if is_toggle:
                if arg_type_info == 'bool':
                    current_effective_val = effective_display_args.get(arg_to_edit_key, False) # Default to False if not set for toggle
                    current_session_args_dict[arg_to_edit_key] = not current_effective_val
                    print_success(f"Toggled {arg_to_edit_key} for session to {'ENABLED' if current_session_args_dict[arg_to_edit_key] else 'DISABLED'}")
                else: print_error(f"Cannot toggle {arg_to_edit_key}. Not boolean.")
                continue

            current_val_edit_display = effective_display_args.get(arg_to_edit_key)
            prompt_msg = f"New value for {arg_to_edit_key} (Current: {current_val_edit_display if current_val_edit_display is not None else 'Default'} | Enter 'unset' to remove session override):"
            new_val_str = prompt(prompt_msg).strip()

            if new_val_str.lower() == 'unset':
                if arg_to_edit_key in current_session_args_dict:
                    del current_session_args_dict[arg_to_edit_key]
                print_success(f"Removed session override for {arg_to_edit_key}. It will use model/global default.")
                continue
            try:
                if arg_type_info == 'bool':
                    if new_val_str.lower() in ['true', 'yes', '1', 'on']: current_session_args_dict[arg_to_edit_key] = True
                    elif new_val_str.lower() in ['false', 'no', '0', 'off']: current_session_args_dict[arg_to_edit_key] = False
                    else: print_error(f"Invalid boolean for {arg_to_edit_key}. Use 'true' or 'false'."); continue
                elif arg_type_info == 'int_str': # For args that can be int or specific strings like "auto", "off"
                    try: 
                        # Test if it's an integer, but store as string to preserve "auto", "off"
                        int(new_val_str) 
                        current_session_args_dict[arg_to_edit_key] = new_val_str
                    except ValueError: # Not a simple integer
                         if new_val_str.lower() in ['auto', 'off'] and arg_to_edit_key in ["--gpulayers", "--quantkv", "--blasbatchsize", "--threads", "--nblas"]:
                             current_session_args_dict[arg_to_edit_key] = new_val_str.lower()
                         else: # Could be a specific string arg for KCPP, or just a typo. Store as is.
                             current_session_args_dict[arg_to_edit_key] = new_val_str
                else: # 'str'
                    current_session_args_dict[arg_to_edit_key] = new_val_str
                print_success(f"Set session override for {arg_to_edit_key} to {current_session_args_dict[arg_to_edit_key]}")
            except ValueError: print_error(f"Invalid value for {arg_to_edit_key}.")
        else: print_error("Invalid choice.")

def _log_to_cli_live_output(text_line: str, live_obj: Optional[Live] = None, progress_obj: Optional[Progress] = None):
    print(text_line.strip())

def monitor_kcpp_output_thread_target_cli(process: subprocess.Popen,
                                         success_event: threading.Event, oom_event: threading.Event,
                                         output_lines_list: List[str],
                                         success_regex_str: str, oom_keywords_list: List[str],
                                         target_port: str,
                                         live_disp: Optional[Live] = None, progress_disp: Optional[Progress] = None):
    try:
        for line_bytes in iter(process.stdout.readline, b''):
            if not line_bytes: break
            try: line_decoded = line_bytes.decode('utf-8', errors='replace')
            except UnicodeDecodeError: line_decoded = line_bytes.decode('latin-1', errors='replace')
            _log_to_cli_live_output(line_decoded, live_disp, progress_disp)
            line_strip = line_decoded.strip()
            if line_strip:
                output_lines_list.append(line_strip)
                line_lower = line_strip.lower()
                if not success_event.is_set() and not oom_event.is_set():
                    success_match = re.search(success_regex_str, line_strip, re.IGNORECASE)
                    if success_match:
                        launched_port_str = target_port
                        try: launched_port_str = success_match.group(1)
                        except IndexError: pass
                        if str(launched_port_str) == str(target_port):
                            success_event.set()
                    if not success_event.is_set():
                        for keyword in oom_keywords_list:
                            if keyword.lower() in line_lower:
                                oom_event.set(); break
            if success_event.is_set() or oom_event.is_set(): break
    except Exception as e:
        err_msg = f"\nError in KCPP monitor thread: {type(e).__name__}: {e}\n"
        _log_to_cli_live_output(err_msg, live_disp, progress_disp)
    finally:
        if process.stdout and not process.stdout.closed:
            try: process.stdout.close()
            except: pass
        _log_to_cli_live_output("\nKCPP output monitoring thread finished.\n", live_disp, progress_disp)

def launch_and_monitor_for_tuning_cli():
    global kcpp_process_obj, kcpp_monitor_thread, monitor_start_time
    global last_proposed_command_list_for_db, vram_at_decision_for_db, last_approx_vram_used_kcpp_mb
    global level_of_last_monitored_run # Ensure this is global to be set here

    if kcpp_process_obj and kcpp_process_obj.poll() is None:
        print_warning("A KoboldCpp process is already being monitored. Please stop it first or wait.")
        return

    print_info(f"Tuning: Launching & Monitoring for OT Level {current_tuning_attempt_level}")
    kcpp_success_event.clear(); kcpp_oom_event.clear(); kcpp_output_lines_shared.clear()
    last_approx_vram_used_kcpp_mb = None
    level_of_last_monitored_run = current_tuning_attempt_level  # Capture the level for *this* specific run

    ot_string = koboldcpp_core.generate_overridetensors(current_tuning_model_analysis_local, current_tuning_attempt_level)
    args_for_kcpp_list = koboldcpp_core.build_command(
        current_tuning_model_path_local, ot_string,
        current_tuning_model_analysis_local, current_tuning_session_base_args
    )
    last_proposed_command_list_for_db = koboldcpp_core.get_command_to_run(KOBOLDCPP_EXECUTABLE, args_for_kcpp_list)
    vram_at_decision_for_db, _, _, _ = koboldcpp_core.get_available_vram_mb()

    kcpp_process_obj, error_msg = koboldcpp_core.launch_process(
        last_proposed_command_list_for_db, capture_output=True, new_console=False, use_text_mode=False
    )

    if error_msg or not kcpp_process_obj:
        print_error(f"Failed to launch KCPP for monitoring: {error_msg or 'Unknown error'}")
        koboldcpp_core.save_config_to_db(
            DB_FILE, current_tuning_model_path_local, current_tuning_model_analysis_local, 
            vram_at_decision_for_db, last_proposed_command_list_for_db,
            level_of_last_monitored_run, # Use the captured level
            "LAUNCH_FAILED_SETUP_CLI", None
        )
        return

    print_info(f"KoboldCpp process started (PID: {kcpp_process_obj.pid}). Monitoring output...")
    effective_args_for_port = get_effective_session_args(current_tuning_model_path_local, current_tuning_session_base_args)
    target_port_for_success = effective_args_for_port.get("--port", "5000")
    
    live_display = None; progress_bar = None
    
    kcpp_monitor_thread = threading.Thread(
        target=monitor_kcpp_output_thread_target_cli,
        args=(kcpp_process_obj, kcpp_success_event, kcpp_oom_event, kcpp_output_lines_shared,
              KOBOLD_SUCCESS_PATTERN, OOM_ERROR_KEYWORDS, target_port_for_success,
              live_display, progress_bar),
        daemon=True
    )
    kcpp_monitor_thread.start()
    monitor_start_time = time.monotonic()
    
    final_outcome_key = None
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed:.0f}s / {task.total:.0f}s"), console=console, transient=True) if dependencies['rich']['module'] else nullcontext() as rich_progress_live:
        if rich_progress_live:
            loading_task = rich_progress_live.add_task("KCPP Loading...", total=float(LOADING_TIMEOUT_SECONDS))
        
        while True:
            elapsed_time = time.monotonic() - monitor_start_time
            if rich_progress_live:
                rich_progress_live.update(loading_task, completed=min(elapsed_time, float(LOADING_TIMEOUT_SECONDS)))

            process_exited = kcpp_process_obj.poll() is not None
            if kcpp_success_event.is_set(): final_outcome_key = "SUCCESS_LOAD_DETECTED_CLI"; break
            if kcpp_oom_event.is_set(): final_outcome_key = "OOM_CRASH_DETECTED_CLI"; break
            if process_exited: final_outcome_key = "PREMATURE_EXIT_CLI"; break
            if elapsed_time > LOADING_TIMEOUT_SECONDS: final_outcome_key = "TIMEOUT_NO_SIGNAL_CLI"; break
            time.sleep(0.25)

    print_info(f"Monitoring completed. Initial Outcome: {final_outcome_key}")
    if final_outcome_key in ["TIMEOUT_NO_SIGNAL_CLI", "OOM_CRASH_DETECTED_CLI", "PREMATURE_EXIT_CLI"]:
        if kcpp_process_obj and kcpp_process_obj.poll() is None:
            print_info("Terminating KCPP process due to unfavorable outcome...")
            koboldcpp_core.kill_process(kcpp_process_obj.pid, force=True)
        kcpp_process_obj = None
    
    db_outcome_to_save = final_outcome_key
    if final_outcome_key == "SUCCESS_LOAD_DETECTED_CLI":
        print_info("API detected. Waiting a few seconds for VRAM to stabilize...")
        time.sleep(max(2.0, float(CONFIG.get("vram_stabilization_wait_s", 3.0))))
        current_free_vram, total_vram, _, _ = koboldcpp_core.get_available_vram_mb()
        if total_vram > 0 and vram_at_decision_for_db is not None:
            used_vram = vram_at_decision_for_db - current_free_vram
            last_approx_vram_used_kcpp_mb = max(0, min(used_vram, total_vram))
            print_info(f"VRAM after load: {current_free_vram:.0f}MB free. Approx KCPP usage: {last_approx_vram_used_kcpp_mb:.0f}MB")
            if current_free_vram < MIN_VRAM_FREE_AFTER_LOAD_MB:
                print_warning(f"VRAM tight! {current_free_vram:.0f}MB < {MIN_VRAM_FREE_AFTER_LOAD_MB}MB target.")
                db_outcome_to_save = "SUCCESS_LOAD_VRAM_TIGHT_CLI"
            else:
                print_success("VRAM usage OK.")
                db_outcome_to_save = "SUCCESS_LOAD_VRAM_OK_CLI"
        else:
            db_outcome_to_save = "SUCCESS_LOAD_NO_VRAM_CHECK_CLI"

    koboldcpp_core.save_config_to_db(DB_FILE, current_tuning_model_path_local, current_tuning_model_analysis_local,
                                   vram_at_decision_for_db, last_proposed_command_list_for_db,
                                   level_of_last_monitored_run, # Use the captured level for DB
                                   db_outcome_to_save, last_approx_vram_used_kcpp_mb)
    handle_post_monitoring_choices_cli(db_outcome_to_save)


def handle_post_monitoring_choices_cli(outcome: str):
    global current_tuning_attempt_level, tuning_in_progress, kcpp_process_obj, last_launched_process_info
    global level_of_last_monitored_run # Ensure access if needed for DB updates

    choices = {}
    default_choice = ""
    print_title("Post-Monitoring Options")
    print_info(f"Outcome: {outcome}")
    if last_approx_vram_used_kcpp_mb is not None:
        print_info(f"Approx. KCPP VRAM Used: {last_approx_vram_used_kcpp_mb:.0f} MB")

    kcpp_is_running = kcpp_process_obj and kcpp_process_obj.poll() is None

    if "SUCCESS_LOAD_VRAM_OK" in outcome:
        print_success("KCPP loaded successfully (VRAM OK).")
        choices = {
            "u": "✅ Accept & Use this KCPP instance",
            "s": "💾 Save as Good, Stop KCPP & Continue Tuning (Try More GPU)",
            "g": "⚙️ Stop KCPP & Try More GPU (This Session)",
            "c": "⚙️ Stop KCPP & Try More CPU (This Session)",
            "q": "↩️ Stop KCPP, Save & Return to Tuning Menu"
        }
        default_choice = "u"
    elif "SUCCESS_LOAD_VRAM_TIGHT" in outcome:
        print_warning("KCPP loaded, but VRAM is tight!")
        choices = {
            "a": "⚠️ Auto-Adjust (Stop KCPP, More CPU & Continue Tuning)",
            "l": "🚀 Launch Anyway / Keep KCPP Running (Risky)",
            "q": "↩️ Stop KCPP, Save & Return to Tuning Menu"
        }
        default_choice = "a"
    elif "OOM" in outcome or "CRASH" in outcome or "PREMATURE_EXIT" in outcome:
        print_error("KCPP failed to load properly (OOM/Crash/Premature Exit).")
        choices = { "c": "⚙️ Try More CPU & Continue Tuning", "q": "↩️ Save & Return to Tuning Menu"}
        default_choice = "c"
    elif "TIMEOUT" in outcome:
        print_warning("KCPP launch timed out.")
        choices = { "c": "⚙️ Try More CPU (Assume OOM & Continue Tuning)", "q": "↩️ Save & Return to Tuning Menu"}
        default_choice = "c"
    else: 
        if kcpp_is_running:
            print_info("KCPP loaded (unknown VRAM status or generic success).")
            choices = {"u": "✅ Keep KCPP Running for Use", "q": "↩️ Stop KCPP, Save & Return to Tuning Menu"}
            default_choice = "u"
        else:
            print_warning("KCPP status unclear, or it has already exited.")
            choices = {"q": "↩️ Save & Return to Tuning Menu"}
            default_choice = "q"

    for key, desc in choices.items(): print(f"  ({key.upper()}) {desc}")
    user_action = prompt("Your choice?", choices=list(choices.keys()), default=default_choice).lower()
    
    new_db_outcome_suffix = ""
    action_requires_kcpp_stop = True

    if user_action == 'u': 
        if kcpp_is_running:
            print_info("Keeping current KoboldCpp instance running for use.")
            new_db_outcome_suffix = "_USER_ACCEPTED_AND_USED"
            last_launched_process_info["pid"] = kcpp_process_obj.pid
            last_launched_process_info["process_obj"] = kcpp_process_obj
            last_launched_process_info["command_list"] = last_proposed_command_list_for_db
            
            effective_args_for_port = get_effective_session_args(current_tuning_model_path_local, current_tuning_session_base_args)
            port_to_use = effective_args_for_port.get("--port", "5000")
            if AUTO_OPEN_WEBUI: webbrowser.open(f"http://localhost:{port_to_use}")
            
            final_outcome_for_db_accepted = outcome + new_db_outcome_suffix
            koboldcpp_core.save_config_to_db(DB_FILE, current_tuning_model_path_local, current_tuning_model_analysis_local,
                                   vram_at_decision_for_db, last_proposed_command_list_for_db,
                                   level_of_last_monitored_run, final_outcome_for_db_accepted, last_approx_vram_used_kcpp_mb)
            session_ctrl_res = kcpp_control_loop_cli(port_to_use, is_monitored_instance=True)
            tuning_in_progress = False
            if session_ctrl_res == "quit_script_leave_running": return "quit_script_leave_running"
            return "new_gguf" # Default to new model after controlling instance
        else:
            print_warning("KCPP is not running, cannot accept and use. Returning to tuning menu.")
            new_db_outcome_suffix = "_ATTEMPTED_USE_BUT_NOT_RUNNING"

    elif user_action == 'l': 
        if kcpp_is_running:
            print_info("Keeping current (risky) KoboldCpp instance running.")
            new_db_outcome_suffix = "_USER_KEPT_RISKY_RUNNING"
            last_launched_process_info["pid"] = kcpp_process_obj.pid
            last_launched_process_info["process_obj"] = kcpp_process_obj
            last_launched_process_info["command_list"] = last_proposed_command_list_for_db
            effective_args_for_port = get_effective_session_args(current_tuning_model_path_local, current_tuning_session_base_args)
            port_to_use = effective_args_for_port.get("--port", "5000")
            if AUTO_OPEN_WEBUI: webbrowser.open(f"http://localhost:{port_to_use}")

            final_outcome_for_db_risky = outcome + new_db_outcome_suffix
            koboldcpp_core.save_config_to_db(DB_FILE, current_tuning_model_path_local, current_tuning_model_analysis_local,
                                   vram_at_decision_for_db, last_proposed_command_list_for_db,
                                   level_of_last_monitored_run, final_outcome_for_db_risky, last_approx_vram_used_kcpp_mb)
            session_ctrl_res = kcpp_control_loop_cli(port_to_use, is_monitored_instance=True)
            tuning_in_progress = False
            if session_ctrl_res == "quit_script_leave_running": return "quit_script_leave_running"
            return "new_gguf"
        else: 
            print_info("KCPP not running. Re-launching this configuration (risky)...")
            new_db_outcome_suffix = "_USER_RELAUNCHED_RISKY"
            final_outcome_for_db_relaunch_risky = outcome + new_db_outcome_suffix
            # The command to relaunch is `last_proposed_command_list_for_db`
            launched_proc = launch_kobold_for_use_cli(last_proposed_command_list_for_db, final_outcome_for_db_relaunch_risky)
            if launched_proc:
                effective_args = get_effective_session_args(current_tuning_model_path_local, current_tuning_session_base_args)
                port = effective_args.get("--port", "5000")
                session_ctrl_res = kcpp_control_loop_cli(port)
                tuning_in_progress = False
                if session_ctrl_res == "quit_script_leave_running": return "quit_script_leave_running"
                return "new_gguf"
            else:
                print_error("Risky re-launch failed. Returning to tuning menu.")
                # DB update for the re-launch attempt failure is handled by launch_kobold_for_use_cli if it fails to start

    if kcpp_is_running and action_requires_kcpp_stop:
        print_info("Stopping current KoboldCpp instance...")
        koboldcpp_core.kill_process(kcpp_process_obj.pid, force=True)
    kcpp_process_obj = None

    if user_action == 's': 
        new_db_outcome_suffix = "_USER_SAVED_GOOD_MORE_GPU"
        if current_tuning_attempt_level > current_tuning_min_level: current_tuning_attempt_level -= 1
        else: print_warning("Already at Max GPU, cannot go further.")
    elif user_action == 'g': 
        new_db_outcome_suffix = "_USER_WANTS_MORE_GPU_NOW"
        if current_tuning_attempt_level > current_tuning_min_level: current_tuning_attempt_level -= 1
        else: print_warning("Already at Max GPU, cannot go further.")
    elif user_action == 'c' or user_action == 'a': 
        if user_action == 'a': new_db_outcome_suffix = "_USER_ACCEPTED_AUTO_ADJUST_CPU"
        else: new_db_outcome_suffix = "_USER_TRIED_MORE_CPU_AFTER_FAIL" if "FAIL" in outcome.upper() or "OOM" in outcome.upper() or "TIMEOUT" in outcome.upper() else "_USER_WANTS_MORE_CPU_NOW"
        if current_tuning_attempt_level < current_tuning_max_level: current_tuning_attempt_level += 1
        else: print_warning("Already at Max CPU, cannot go further.")
    elif user_action == 'q': 
        new_db_outcome_suffix = "_USER_SAVED_RETURN_TUNING_MENU"
    
    final_outcome_for_db = outcome + new_db_outcome_suffix
    koboldcpp_core.save_config_to_db(DB_FILE, current_tuning_model_path_local, current_tuning_model_analysis_local,
                                   vram_at_decision_for_db, last_proposed_command_list_for_db,
                                   level_of_last_monitored_run, # Save with the level that was *tested*
                                   final_outcome_for_db, last_approx_vram_used_kcpp_mb)
    # Implicitly continues tuning loop by returning from this function

def launch_kobold_for_use_cli(command_list_to_run: List[str], db_outcome_on_success: str, level_for_db_record: Optional[int] = None):
    global last_launched_process_info, gguf_file_global, current_model_analysis_global

    if last_launched_process_info["process_obj"] and last_launched_process_info["process_obj"].poll() is None:
        print_info(f"Stopping previously launched KCPP (PID: {last_launched_process_info['pid']})...")
        koboldcpp_core.kill_process(last_launched_process_info["pid"])
    last_launched_process_info = {"pid": None, "process_obj": None, "command_list": []}

    print_info(f"Launching KoboldCpp for use...")
    vram_before_launch, _, _, _ = koboldcpp_core.get_available_vram_mb()
    
    # Determine level to log. If provided, use it. Otherwise, use placeholder.
    final_level_for_db = level_for_db_record if level_for_db_record is not None else \
                         (current_tuning_attempt_level if tuning_in_progress else 0)

    koboldcpp_core.save_config_to_db(DB_FILE, gguf_file_global, current_model_analysis_global,
                                   vram_before_launch, command_list_to_run,
                                   final_level_for_db, 
                                   db_outcome_on_success, None) # VRAM used unknown before launch

    process, error_msg = koboldcpp_core.launch_process(command_list_to_run, capture_output=False, new_console=True)

    if error_msg or not process:
        print_error(f"Failed to launch KoboldCPP: {error_msg or 'Unknown error'}")
        # Log failure if it couldn't even start
        koboldcpp_core.save_config_to_db(DB_FILE, gguf_file_global, current_model_analysis_global,
                                   vram_before_launch, command_list_to_run,
                                   final_level_for_db, 
                                   "LAUNCH_FOR_USE_FAILED_CLI", None)
        return None
    else:
        print_success(f"KoboldCpp launched in new console (PID: {process.pid}).")
        last_launched_process_info["pid"] = process.pid
        last_launched_process_info["process_obj"] = process
        last_launched_process_info["command_list"] = command_list_to_run
        
        args_dict = koboldcpp_core.args_list_to_dict(command_list_to_run)
        port = args_dict.get("--port", CONFIG.get("default_args", {}).get("--port", "5000"))

        if AUTO_OPEN_WEBUI:
            print_info(f"Attempting to open Web UI at http://localhost:{port} in a few seconds...")
            threading.Timer(3.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
        return process

def kcpp_control_loop_cli(port_to_use: str, is_monitored_instance: bool = False) -> str:
    global last_launched_process_info, kcpp_process_obj

    process_to_control = kcpp_process_obj if is_monitored_instance else last_launched_process_info.get("process_obj")
    pid_to_control = process_to_control.pid if process_to_control and hasattr(process_to_control, 'pid') else last_launched_process_info.get("pid")

    while True:
        if process_to_control and process_to_control.poll() is not None:
            print_warning(f"KoboldCpp process (PID: {pid_to_control}) seems to have exited.")
            if is_monitored_instance: kcpp_process_obj = None
            else: last_launched_process_info = {"pid": None, "process_obj": None, "command_list": []}
            return "new_gguf"

        print_title("KCPP Control Options")
        active_pid_display = f" (PID: {pid_to_control})" if pid_to_control else ""
        print(f"  (S)top KCPP{active_pid_display} & Select New Model")
        print(f"  (Q)uit Launcher (Stops KCPP{active_pid_display} & Exits)")
        print(f"  (E)xit Launcher (Leave KCPP{active_pid_display} Running)")
        print(f"  (W)Open WebUI (http://localhost:{port_to_use})")
        
        choice_run = prompt("KCPP Control", choices=['s', 'q', 'e', 'w'], default='s').lower().strip()

        if choice_run == 's':
            if process_to_control and pid_to_control:
                print_info(f"Stopping KCPP (PID: {pid_to_control})...")
                koboldcpp_core.kill_process(pid_to_control)
            if is_monitored_instance: kcpp_process_obj = None
            else: last_launched_process_info = {"pid": None, "process_obj": None, "command_list": []}
            return "new_gguf"
        elif choice_run == 'q':
            if process_to_control and pid_to_control:
                print_info(f"Stopping KCPP (PID: {pid_to_control}) and quitting launcher...")
                koboldcpp_core.kill_process(pid_to_control)
            return "quit_script"
        elif choice_run == 'e':
            print_info(f"Exiting launcher. KCPP{active_pid_display} will be left running.")
            return "quit_script_leave_running"
        elif choice_run == 'w':
            print_info(f"Opening Web UI at http://localhost:{port_to_use}")
            webbrowser.open(f"http://localhost:{port_to_use}")
        else:
            print_error("Invalid control choice.")

def run_model_tuning_session_cli():
    global tuning_in_progress, current_tuning_attempt_level, current_tuning_min_level, current_tuning_max_level
    global current_tuning_session_base_args, current_tuning_model_path_local, current_tuning_model_analysis_local
    global gguf_file_global, current_model_analysis_global, level_of_last_monitored_run

    if not gguf_file_global or not current_model_analysis_global.get('filepath'):
        print_error("No model selected or analyzed. Please select a model first.")
        return "new_gguf"

    tuning_in_progress = True
    current_tuning_model_path_local = gguf_file_global
    current_tuning_model_analysis_local = current_model_analysis_global.copy()
    # Initialize session args from effective global/model-specific defaults
    current_tuning_session_base_args = get_effective_session_args(current_tuning_model_path_local, {})

    print_title(f"Starting Auto-Tuning Session for: {os.path.basename(current_tuning_model_path_local)}")
    print_info(f"Analysis: Size ~{current_tuning_model_analysis_local.get('size_b', 'N/A')}B, "
               f"Quant ~{current_tuning_model_analysis_local.get('quant', 'N/A')}, "
               f"MoE: {'Yes' if current_tuning_model_analysis_local.get('is_moe') else 'No'}")

    if current_tuning_model_analysis_local.get('is_moe'):
        current_tuning_min_level, current_tuning_max_level, initial_heuristic_level = -25, 10, -10
    else:
        current_tuning_min_level, current_tuning_max_level = -17, 9
        size_b = current_tuning_model_analysis_local.get('size_b', 0)
        if isinstance(size_b, (int, float)): 
            initial_heuristic_level = -3 if size_b >=30 else (-5 if size_b >=20 else -7)
        else: 
            initial_heuristic_level = -5

    vram_now, _, _, _ = koboldcpp_core.get_available_vram_mb()
    best_hist_config = koboldcpp_core.find_best_historical_config(DB_FILE, current_tuning_model_analysis_local, vram_now)
    
    if best_hist_config and "attempt_level" in best_hist_config:
        print_info(f"Found historical config. Level: {best_hist_config['attempt_level']}, Outcome: {best_hist_config['outcome']}")
        hist_level, hist_outcome = best_hist_config['attempt_level'], best_hist_config.get('outcome', "")
        # Adjust initial level based on historical outcome
        if hist_outcome.startswith("SUCCESS_LOAD_VRAM_OK") or hist_outcome.endswith("_USER_SAVED_GOOD_MORE_GPU"):
            initial_heuristic_level = max(current_tuning_min_level, hist_level - 1)
        elif hist_outcome.endswith("_USER_ACCEPTED_AUTO_ADJUST_CPU") or hist_outcome.endswith("_USER_TRIED_MORE_CPU_AFTER_FAIL"):
             initial_heuristic_level = min(current_tuning_max_level, hist_level + 1)
        else: initial_heuristic_level = hist_level
        
        remembered_args_list = best_hist_config.get("args_list", [])
        if remembered_args_list:
            remembered_args_dict = koboldcpp_core.args_list_to_dict(remembered_args_list)
            remembered_args_dict.pop("--model", None); remembered_args_dict.pop("--overridetensors", None)
            current_tuning_session_base_args.update(remembered_args_dict) # Apply remembered args
            print_info(f"Applied remembered arguments to current session base. OT Level adjusted to: {initial_heuristic_level}")
    else:
        print_info(f"No suitable historical config. Starting with heuristic OT Level: {initial_heuristic_level}")
    
    current_tuning_attempt_level = max(current_tuning_min_level, min(initial_heuristic_level, current_tuning_max_level))
    level_of_last_monitored_run = current_tuning_attempt_level # Initialize

    while tuning_in_progress:
        print("\n" + "=" * 70)
        current_tuning_attempt_level = max(current_tuning_min_level, min(current_tuning_attempt_level, current_tuning_max_level))
        ot_string = koboldcpp_core.generate_overridetensors(current_tuning_model_analysis_local, current_tuning_attempt_level)
        description = koboldcpp_core.get_offload_description(current_tuning_model_analysis_local, current_tuning_attempt_level, ot_string)
        gpu_layers = koboldcpp_core.get_gpu_layers_for_level(current_tuning_model_analysis_local, current_tuning_attempt_level)
        total_layers = current_tuning_model_analysis_local.get('num_layers', 32)
        
        if dependencies['rich']['module']:
            table = Table(title="Current Tensor Offload Strategy")
            table.add_column("Setting", style="cyan"); table.add_column("Value", style="yellow", overflow="fold")
            table.add_row("Level", f"{current_tuning_attempt_level}")
            range_desc = f"{current_tuning_min_level}=MaxGPU ... {current_tuning_max_level}={'SuperMaxCPU' if current_tuning_model_analysis_local.get('is_moe') else 'MaxCPU'}"
            table.add_row("Range", range_desc); table.add_row("Strategy", description)
            table.add_row("Regex", ot_string or "None (Max GPU layers if --gpulayers is high)")
            table.add_row("GPU Layers", f"{gpu_layers}/{total_layers}")
            console.print(table)
        else:
            print(f"🛠️ OT Level: {current_tuning_attempt_level}\n   Range: {current_tuning_min_level}=MaxGPU to {current_tuning_max_level}={'SuperMaxCPU' if current_tuning_model_analysis_local.get('is_moe') else 'MaxCPU'}\n   Strategy: {description}\n   Regex: {(ot_string or 'None')}\n   GPU Layers: {gpu_layers}/{total_layers}")

        args_for_kcpp_list = koboldcpp_core.build_command(
            current_tuning_model_path_local, ot_string,
            current_tuning_model_analysis_local, current_tuning_session_base_args
        )
        display_command_list = koboldcpp_core.get_command_to_run(KOBOLDCPP_EXECUTABLE, args_for_kcpp_list)
        _, _, vram_info_str, _ = koboldcpp_core.get_available_vram_mb()
        if dependencies['rich']['module']: console.print(Panel(f"{vram_info_str}", title="GPU Info", style="green" if "error" not in vram_info_str.lower() else "red"))
        else: print(f"    {vram_info_str}")
        print_title("Proposed Command for This OT Level"); print_command(koboldcpp_core.format_command_for_display(display_command_list))

        menu_options_str = "(L)aunch & Monitor | (S)kip Tune & Launch Config | (G)PU More | (C)PU More | (E)dit Base Args | (N)ew GGUF | (H)istory (This Model) | (Q)uit Tuning"
        print_title("Tuning Actions"); print(menu_options_str)
        user_choice = prompt("Your choice", choices=['l','s','g','c','e','n','h','q'], default='l').lower().strip()

        if user_choice == 'l':
            session_action = launch_and_monitor_for_tuning_cli() # This function now calls handle_post_monitoring_choices_cli
            if session_action == "quit_script_leave_running": # Propagate quit signal if KCPP left running
                tuning_in_progress = False; return "quit_script_leave_running"
            if not tuning_in_progress: # Tuning might have ended
                return "new_gguf" 
        elif user_choice == 's':
            print_info("Skipping further tuning, launching current configuration directly...")
            # Pass current_tuning_attempt_level for DB logging
            launched_proc = launch_kobold_for_use_cli(display_command_list, "SUCCESS_USER_DIRECT_LAUNCH_CLI", level_for_db_record=current_tuning_attempt_level)
            if launched_proc:
                effective_args = get_effective_session_args(current_tuning_model_path_local, current_tuning_session_base_args)
                port = effective_args.get("--port", "5000")
                session_ctrl_res = kcpp_control_loop_cli(port)
                tuning_in_progress = False; return session_ctrl_res
            else: print_error("Direct launch failed."); continue
        elif user_choice == 'g':
            if current_tuning_attempt_level > current_tuning_min_level: current_tuning_attempt_level -=1; print_success(f"Shifted towards GPU. New OT Level: {current_tuning_attempt_level}")
            else: print_warning(f"Already at Max GPU (Level {current_tuning_min_level}).")
        elif user_choice == 'c':
            if current_tuning_attempt_level < current_tuning_max_level: current_tuning_attempt_level +=1; print_success(f"Shifted towards CPU. New OT Level: {current_tuning_attempt_level}")
            else: print_warning(f"Already at Max CPU (Level {current_tuning_max_level}).")
        elif user_choice == 'e':
            updated_session_args, perm_changed = edit_current_args_interactive_cli(current_tuning_model_path_local, current_tuning_session_base_args)
            if updated_session_args is not None: current_tuning_session_base_args = updated_session_args
            if perm_changed :
                 current_tuning_session_base_args = get_effective_session_args(current_tuning_model_path_local, {})
                 print_info("Permanent arguments changed. Session overrides reset for this model.")
        elif user_choice == 'n':
            tuning_in_progress = False; return "new_gguf"
        elif user_choice == 'h': view_db_history_cli(current_tuning_model_path_local)
        elif user_choice == 'q':
            tuning_in_progress = False; return "new_gguf"
        else: print_error("Invalid input.")
    tuning_in_progress = False
    return "new_gguf"

def main_cli():
    global CONFIG, KOBOLDCPP_EXECUTABLE, DB_FILE, DEFAULT_GGUF_DIR, AUTO_OPEN_WEBUI
    global VRAM_SAFETY_BUFFER_MB, MIN_VRAM_FREE_AFTER_LOAD_MB, LOADING_TIMEOUT_SECONDS
    global KOBOLD_SUCCESS_PATTERN, OOM_ERROR_KEYWORDS
    global gguf_file_global, current_model_analysis_global, last_gguf_directory, last_launched_process_info

    core_init_results = koboldcpp_core.initialize_launcher()
    CONFIG = core_init_results["config"]
    KOBOLDCPP_EXECUTABLE = CONFIG["koboldcpp_executable"]
    DB_FILE = os.path.abspath(CONFIG["db_file"])
    DEFAULT_GGUF_DIR = CONFIG.get("default_gguf_dir", "")
    AUTO_OPEN_WEBUI = CONFIG.get("auto_open_webui", True)
    VRAM_SAFETY_BUFFER_MB = CONFIG.get("vram_safety_buffer_mb", 768)
    MIN_VRAM_FREE_AFTER_LOAD_MB = CONFIG.get("min_vram_free_after_load_success_mb", 512)
    LOADING_TIMEOUT_SECONDS = CONFIG.get("loading_timeout_seconds", 60)
    KOBOLD_SUCCESS_PATTERN = CONFIG.get("kobold_success_pattern", r"Starting Kobold API on port (\d+)")
    OOM_ERROR_KEYWORDS = [k.lower() for k in CONFIG.get("oom_error_keywords", [])]
    last_gguf_directory = CONFIG.get("last_used_gguf_dir", "")

    if not core_init_results["initialized"]:
        if not core_init_results["config_loaded"]: print_warning(f"Config issue: {core_init_results['config_message']}")
        if not core_init_results["db_success"]: print_warning(f"DB issue: {core_init_results['db_message']}")
    if not CONFIG.get("first_run_completed", False):
        if not handle_first_run_prompts_cli(CONFIG):
            print_error("Initial setup failed. Exiting."); sys.exit(1)
        KOBOLDCPP_EXECUTABLE = CONFIG["koboldcpp_executable"]; DEFAULT_GGUF_DIR = CONFIG.get("default_gguf_dir", ""); AUTO_OPEN_WEBUI = CONFIG.get("auto_open_webui", True)

    print_title(f"KoboldCpp Smart Launcher v{LAUNCHER_CLI_VERSION} (Core: {koboldcpp_core.DEFAULT_CONFIG_TEMPLATE.get('version', 'N/A')})")
    sys_info = core_init_results.get("system_info", {}); gpu_info = core_init_results.get("gpu_info", {})
    print_info(f"OS: {sys_info.get('os_name','N/A')} {sys_info.get('os_version','N/A')} | Python: {sys_info.get('python_version','N/A').split()[0]}")
    if gpu_info.get("success"): print_info(f"GPU: {gpu_info.get('name', 'N/A')} | VRAM: {gpu_info.get('free_mb',0):.0f}/{gpu_info.get('total_mb',0):.0f}MB free")
    else: print_warning(f"GPU Info: {gpu_info.get('message', 'Could not detect GPU details.')}")
    
    if not os.path.exists(KOBOLDCPP_EXECUTABLE):
        import shutil
        resolved_path = shutil.which(KOBOLDCPP_EXECUTABLE) or shutil.which(os.path.basename(KOBOLDCPP_EXECUTABLE))
        if resolved_path:
            print_info(f"KoboldCpp executable '{KOBOLDCPP_EXECUTABLE}' updated via PATH to: {resolved_path}")
            KOBOLDCPP_EXECUTABLE = os.path.abspath(resolved_path); CONFIG["koboldcpp_executable"] = KOBOLDCPP_EXECUTABLE
            koboldcpp_core.save_launcher_config(CONFIG)
        elif KOBOLDCPP_EXECUTABLE.lower().endswith(".py") and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), KOBOLDCPP_EXECUTABLE)):
             KOBOLDCPP_EXECUTABLE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), KOBOLDCPP_EXECUTABLE))
             print_info(f"Using KoboldCpp Python script: {KOBOLDCPP_EXECUTABLE}")
        else:
            print_error(f"FATAL: KoboldCpp target '{KOBOLDCPP_EXECUTABLE}' not found and not in PATH."); sys.exit(1)
    
    while True:
        select_result = select_gguf_file_cli()
        if select_result is None: break 
        if select_result == "main_menu": continue

        gguf_file_global = select_result
        current_model_analysis_global = koboldcpp_core.analyze_filename(gguf_file_global)
        session_outcome = run_model_tuning_session_cli()

        if session_outcome == "quit_script": break
        if session_outcome == "quit_script_leave_running":
            print_info("Exiting launcher. KoboldCpp may still be running."); return
    print_title("KoboldCpp Smart Launcher Finished")

class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): pass

if __name__ == "__main__":
    pynvml_globally_initialized_cli = False
    if koboldcpp_core.pynvml_available:
        try:
            koboldcpp_core.pynvml.nvmlInit()
            pynvml_globally_initialized_cli = True; print_info("PyNVML initialized.")
        except Exception as e_nvml_init: # Broader exception for nvmlInit
            print_warning(f"PyNVML init failed: {e_nvml_init}. GPU VRAM info might be unavailable for NVIDIA.")
            koboldcpp_core.pynvml_available = False # Mark as unavailable if init fails
    try:
        main_cli()
    except KeyboardInterrupt: print_warning("\nLauncher terminated by user (Ctrl+C).")
    except Exception as e_fatal:
        print_error(f"\nFATAL SCRIPT ERROR: {e_fatal}"); import traceback; traceback.print_exc()
    finally:
        print_info("Exiting. Cleaning up any lingering KoboldCpp process launched by this session...")
        if last_launched_process_info.get("process_obj") and last_launched_process_info["process_obj"].poll() is None:
            if last_launched_process_info.get("pid"): # Check if pid exists
                print_info(f"Stopping last launched KCPP process (PID: {last_launched_process_info['pid']})...")
                koboldcpp_core.kill_process(last_launched_process_info["pid"], force=True)
        
        if KOBOLDCPP_EXECUTABLE: 
            exe_base = os.path.basename(KOBOLDCPP_EXECUTABLE)
            print_info(f"Performing cleanup sweep for processes like '{exe_base}'...")
            koboldcpp_core.kill_processes_by_name(exe_base)
            if exe_base.lower().endswith(".py"): # Also check for python running this script
                 koboldcpp_core.kill_processes_by_name("python", cmdline_substr_filter=exe_base)

        if pynvml_globally_initialized_cli and koboldcpp_core.pynvml_available: # Check availability again before shutdown
            try: koboldcpp_core.pynvml.nvmlShutdown(); print_info("PyNVML shut down.")
            except Exception as e_nvml_shutdown: print_warning(f"Error during PyNVML shutdown: {e_nvml_shutdown}")
        print_info("Launcher exited.")
