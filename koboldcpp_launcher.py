#!/usr/bin/env python3
# KoboldCpp Smart Launcher - VRAM Auto-Tuning Edition
# Version 11.2.1 (CLI - Aligned with GUI features)

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
import pathlib # Keep for Path object usage
import webbrowser
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import platform
from pathlib import Path # Explicitly import Path

# Import the core script
import koboldcpp_core

# Handle dependency imports with helpful error messages
dependencies = {
    'rich': {'required': False, 'module': None, 'purpose': 'improved terminal UI'},
    'psutil': {'required': False, 'module': None, 'purpose': 'system resource monitoring'},
    'pynvml': {'required': False, 'module': None, 'purpose': 'NVIDIA GPU VRAM detection (used by core)'},
    'tkinter': {'required': False, 'module': None, 'purpose': 'file open dialog (optional)'}
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

# --- Global Configuration Variables (populated from koboldcpp_core) ---
CONFIG: Dict[str, Any] = {}
KOBOLDCPP_EXECUTABLE = ""
DB_FILE = "" # Will be an absolute path from core's CONFIG
DEFAULT_GGUF_DIR = ""
AUTO_OPEN_WEBUI = True
VRAM_SAFETY_BUFFER_MB = 0
MIN_VRAM_FREE_AFTER_LOAD_MB = 0
LOADING_TIMEOUT_SECONDS = 60
KOBOLD_SUCCESS_PATTERN = ""
OOM_ERROR_KEYWORDS: List[str] = []
LAUNCHER_CLI_VERSION = "11.2.1"

# --- Runtime State Variables ---
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
    detected_exe_candidate = None
    default_exe_name_platform = "koboldcpp.exe" if platform.system() == "Windows" else "./koboldcpp"
    if os.path.exists(default_exe_name_platform):
        detected_exe_candidate = os.path.abspath(default_exe_name_platform)
    elif shutil.which("koboldcpp"):
        detected_exe_candidate = shutil.which("koboldcpp")

    current_exe_in_config = config_dict.get("koboldcpp_executable", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["koboldcpp_executable"])
    exe_prompt_default = detected_exe_candidate or current_exe_in_config

    if detected_exe_candidate: print_success(f"Auto-detected KoboldCpp candidate: {detected_exe_candidate}")

    user_exe_path_input = prompt(f"Enter path to KoboldCpp executable/script", default=exe_prompt_default).strip()
    user_exe_path_abs = ""
    while True:
        if os.path.isabs(user_exe_path_input):
            potential_abs_path = user_exe_path_input
        else:
            potential_abs_path = os.path.abspath(user_exe_path_input)

        if os.path.exists(potential_abs_path) and (os.access(potential_abs_path, os.X_OK) or potential_abs_path.lower().endswith(".py")):
            user_exe_path_abs = potential_abs_path
            break
        
        found_in_path_shutil = shutil.which(user_exe_path_input)
        if found_in_path_shutil:
            user_exe_path_abs = os.path.abspath(found_in_path_shutil)
            print_success(f"Found '{os.path.basename(user_exe_path_input)}' in PATH: {user_exe_path_abs}")
            break
        
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
    else:
        config_dict["default_gguf_dir"] = ""

    db_file_location_absolute = config_dict["db_file"]
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

    config_dict['first_run_completed'] = True
    save_success, save_msg = koboldcpp_core.save_launcher_config(config_dict)
    if save_success: print_success("\nInitial setup complete! Configuration saved.")
    else: print_error(f"\nFailed to save initial configuration: {save_msg}")
    return save_success


def select_gguf_file_cli() -> Optional[str]:
    global last_gguf_directory, CONFIG, DEFAULT_GGUF_DIR

    print_title("Select GGUF Model / Main Menu")

    main_menu_actions = { "s": "Select GGUF Model File", "h": "View Global Launch History", "q": "Quit Launcher" }
    print("Main Menu Options:"); [print(f"  ({k.upper()}) {d}") for k, d in main_menu_actions.items()]

    while True:
        action_choice = prompt("Your choice", choices=list(main_menu_actions.keys()), default="s").lower()
        if action_choice == 'q': return None
        if action_choice == 'h':
            view_db_history_cli()
            print_title("Select GGUF Model / Main Menu"); [print(f"  ({k.upper()}) {d}") for k, d in main_menu_actions.items()]
            continue
        if action_choice == 's': break

    if dependencies['tkinter']['module']:
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
                koboldcpp_core.save_launcher_config(CONFIG)
                print_success(f"Selected via dialog: {os.path.basename(abs_filepath)}")
                return abs_filepath
            print_info("File selection cancelled via dialog.")
            return "main_menu"
        except Exception as e_tk:
            print_warning(f"Tkinter file dialog failed: {e_tk}. Falling back to manual path input.")

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
            koboldcpp_core.save_launcher_config(CONFIG)
            print_success(f"Selected via manual input: {os.path.basename(abs_path_manual)}")
            return abs_path_manual
        print_error(f"Path '{potential_full_path}' is not a valid .gguf file. Please try again or press Enter to cancel.")


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
            history_table.add_column(col_name, style=style, justify=justify_opt, overflow="fold")

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
    arg_definitions_for_cli_editor = [
        {"param": "--threads", "help": "CPU threads. 'auto' or number.", "type": "str"},
        {"param": "--nblas", "help": "BLAS threads. 'auto' or number.", "type": "str"},
        {"param": "--contextsize", "help": "Context tokens (e.g., 4096, 16384).", "type": "int_str"},
        {"param": "--promptlimit", "help": "Max prompt tokens (e.g. 4000).", "type": "int_str"},
        {"param": "--usecublas", "help": "Use CUBLAS (NVIDIA). (true/false)", "type": "bool"},
        {"param": "--flashattention", "help": "Use FlashAttention. (true/false)", "type": "bool"},
        {"param": "--nommap", "help": "Disable memory mapping. (true/false)", "type": "bool"},
        {"param": "--lowvram", "help": "Enable low VRAM mode. (true/false)", "type": "bool"},
        {"param": "--port", "help": "Network port (e.g., 5000).", "type": "int_str"},
        {"param": "--defaultgenamt", "help": "Default tokens to generate (e.g., 2048).", "type": "int_str"},
        {"param": "--gpulayers", "help": "GPU layers. 'auto', 'off', or number (e.g. 35, 999 for max).", "type": "str"},
        {"param": "--quantkv", "help": "K/V cache quant. 'auto', 'off', or number (0=f32, 1=Q8_0).", "type": "str"},
        {"param": "--blasbatchsize", "help": "BLAS batch size. 'auto', 'off', or number (e.g. 128, 512).", "type": "str"}
    ]
    editable_arg_keys_ordered = [
        defn["param"] for defn in arg_definitions_for_cli_editor
        if defn["param"] in koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"]
    ]
    temp_session_overrides = current_session_args_overrides.copy()

    while True:
        print_title("Edit Base Startup Arguments")
        effective_args_for_display = get_effective_session_args(model_path_for_specifics, temp_session_overrides)
        idx_to_param_map = {}

        if dependencies['rich']['module']:
            args_table = Table(title="Effective Arguments for This Session (Core -> Global -> Model -> Session)")
            args_table.add_column("#", style="cyan"); args_table.add_column("Argument", style="green");
            args_table.add_column("Current Value", style="yellow"); args_table.add_column("Description");
            for i, arg_key_name in enumerate(editable_arg_keys_ordered):
                idx_str_display = str(i + 1); idx_to_param_map[idx_str_display] = arg_key_name
                value_to_display = effective_args_for_display.get(arg_key_name)
                is_bool_type_param = any(d['param'] == arg_key_name and d['type'] == 'bool' for d in arg_definitions_for_cli_editor)
                value_str_display = ("ENABLED" if value_to_display is True else ("DISABLED" if value_to_display is False else "NOT SET (KCPP default)")) if is_bool_type_param else (str(value_to_display) if value_to_display is not None else "NOT SET (KCPP default)")
                help_text_for_arg = next((d['help'] for d in arg_definitions_for_cli_editor if d['param'] == arg_key_name), "")
                args_table.add_row(idx_str_display, arg_key_name, value_str_display, help_text_for_arg)
            console.print(args_table)
        else:
            print("Effective Arguments for This Session (Core -> Global -> Model -> Session):")
            for i, arg_key_name in enumerate(editable_arg_keys_ordered):
                idx_str_display = str(i + 1); idx_to_param_map[idx_str_display] = arg_key_name
                value_to_display = effective_args_for_display.get(arg_key_name)
                is_bool_type_param = any(d['param'] == arg_key_name and d['type'] == 'bool' for d in arg_definitions_for_cli_editor)
                value_str_display = ("ENABLED" if value_to_display is True else ("DISABLED" if value_to_display is False else str(value_to_display))) if is_bool_type_param else (str(value_to_display) if value_to_display is not None else "NOT SET")
                help_text_for_arg = next((d['help'] for d in arg_definitions_for_cli_editor if d['param'] == arg_key_name), "")
                print(f"  ({idx_str_display}) {arg_key_name:<18}: {value_str_display:<15} - {help_text_for_arg}")

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
                for arg_key_to_check in editable_arg_keys_ordered:
                    value_from_effective_display = effective_args_for_display.get(arg_key_to_check)
                    global_baseline_for_this_arg = global_baseline_args_for_perm.get(arg_key_to_check)
                    if value_from_effective_display is not None:
                        if value_from_effective_display != global_baseline_for_this_arg:
                            model_specifics_to_set[arg_key_to_check] = value_from_effective_display
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
            arg_type_info_str = next((d['type'] for d in arg_definitions_for_cli_editor if d['param'] == arg_to_edit_name), "str")
            if is_toggle_action:
                if arg_type_info_str == 'bool':
                    current_effective_val_for_toggle = effective_args_for_display.get(arg_to_edit_name, False)
                    temp_session_overrides[arg_to_edit_name] = not current_effective_val_for_toggle
                    print_success(f"Toggled session override for {arg_to_edit_name} to {'ENABLED' if temp_session_overrides[arg_to_edit_name] else 'DISABLED'}")
                else: print_error(f"Cannot toggle '{arg_to_edit_name}'. It is not a boolean type argument.")
                continue
            current_val_for_edit_prompt = effective_args_for_display.get(arg_to_edit_name)
            prompt_message = f"New value for {arg_to_edit_name} (Current: {current_val_for_edit_prompt if current_val_for_edit_prompt is not None else 'Default'} | Enter 'unset' to remove session override):"
            new_val_str_input = prompt(prompt_message).strip()
            if new_val_str_input.lower() == 'unset':
                if arg_to_edit_name in temp_session_overrides:
                    del temp_session_overrides[arg_to_edit_name]
                print_success(f"Removed session override for {arg_to_edit_name}. It will use model/global default.")
            elif arg_type_info_str == 'bool':
                if new_val_str_input.lower() in ['true', 'yes', '1', 'on']: temp_session_overrides[arg_to_edit_name] = True
                elif new_val_str_input.lower() in ['false', 'no', '0', 'off']: temp_session_overrides[arg_to_edit_name] = False
                else: print_error(f"Invalid boolean value for {arg_to_edit_name}. Use 'true' or 'false'."); continue
            elif arg_type_info_str == 'int_str':
                temp_session_overrides[arg_to_edit_name] = new_val_str_input
            else:
                temp_session_overrides[arg_to_edit_name] = new_val_str_input
            if new_val_str_input.lower() != 'unset':
                 print_success(f"Set session override for {arg_to_edit_name} to {temp_session_overrides.get(arg_to_edit_name)}")
        else: print_error("Invalid choice. Please enter a number from the list or a valid action.")


def _log_to_cli_live_output(text_line: str, live_obj: Optional[Live] = None, progress_obj: Optional[Progress] = None):
    print(text_line.strip())


def monitor_kcpp_output_thread_target_cli(
    process: subprocess.Popen,
    success_event_thread: threading.Event,
    oom_event_thread: threading.Event,
    output_lines_list_shared: List[str],
    success_regex_str_config: str,
    oom_keywords_list_config: List[str],
    target_port_for_success_check: str,
    live_display_obj: Optional[Live] = None,
    progress_bar_obj: Optional[Progress] = None
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
                        for oom_keyword in oom_keywords_list_config:
                            if oom_keyword.lower() in line_strip_lower_case:
                                oom_event_thread.set(); break
            if success_event_thread.is_set() or oom_event_thread.is_set(): break
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
    global level_of_last_monitored_run

    if kcpp_process_obj and kcpp_process_obj.poll() is None:
        print_warning("A KoboldCpp process is already being monitored. Please stop it first or wait.")
        return

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
    vram_at_decision_for_db, _, _, _ = koboldcpp_core.get_available_vram_mb()

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
        return

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
        args=(kcpp_process_obj, kcpp_success_event, kcpp_oom_event, kcpp_output_lines_shared, KOBOLD_SUCCESS_PATTERN, OOM_ERROR_KEYWORDS, target_port_str_for_success),
        daemon=True
    )
    kcpp_monitor_thread.start()
    monitor_start_time = time.monotonic()
    final_outcome_key_from_monitor = None
    try:
        while True:
            elapsed_monitor_time = time.monotonic() - monitor_start_time
            if rich_progress_live_obj and loading_task_id is not None:
                rich_progress_live_obj.update(loading_task_id, completed=min(elapsed_monitor_time, float(LOADING_TIMEOUT_SECONDS)))
            process_has_exited = kcpp_process_obj.poll() is not None
            if kcpp_success_event.is_set(): final_outcome_key_from_monitor = "SUCCESS_LOAD_DETECTED_CLI"; break
            if kcpp_oom_event.is_set(): final_outcome_key_from_monitor = "OOM_CRASH_DETECTED_CLI"; break
            if process_has_exited and not kcpp_success_event.is_set() and not kcpp_oom_event.is_set():
                final_outcome_key_from_monitor = "PREMATURE_EXIT_CLI"; break
            if elapsed_monitor_time > LOADING_TIMEOUT_SECONDS: final_outcome_key_from_monitor = "TIMEOUT_NO_SIGNAL_CLI"; break
            time.sleep(0.25)
    finally:
        if rich_progress_live_obj: rich_progress_live_obj.stop()

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
        current_free_vram_after_load, total_vram_after_load, _, _ = koboldcpp_core.get_available_vram_mb()
        if total_vram_after_load > 0 and vram_at_decision_for_db is not None:
            vram_used_by_kcpp = vram_at_decision_for_db - current_free_vram_after_load
            last_approx_vram_used_kcpp_mb = max(0, min(vram_used_by_kcpp, total_vram_after_load))
            print_info(f"VRAM after load: {current_free_vram_after_load:.0f}MB free. Approx KCPP usage: {last_approx_vram_used_kcpp_mb:.0f}MB")
            if current_free_vram_after_load < MIN_VRAM_FREE_AFTER_LOAD_MB:
                print_warning(f"VRAM tight! {current_free_vram_after_load:.0f}MB < {MIN_VRAM_FREE_AFTER_LOAD_MB}MB target.")
                db_outcome_to_save_str = "SUCCESS_LOAD_VRAM_TIGHT_CLI"
            else:
                print_success("VRAM usage OK.")
                db_outcome_to_save_str = "SUCCESS_LOAD_VRAM_OK_CLI"
        else:
            db_outcome_to_save_str = "SUCCESS_LOAD_NO_VRAM_CHECK_CLI"

    koboldcpp_core.save_config_to_db(
        DB_FILE, current_tuning_model_path_local, current_tuning_model_analysis_local,
        vram_at_decision_for_db, last_proposed_command_list_for_db,
        level_of_last_monitored_run, db_outcome_to_save_str, last_approx_vram_used_kcpp_mb
    )
    # Call handle_post_monitoring_choices_cli which returns the next action string
    return handle_post_monitoring_choices_cli(db_outcome_to_save_str, kcpp_process_obj)


def handle_post_monitoring_choices_cli(outcome_from_monitor: str, monitored_kcpp_instance: Optional[subprocess.Popen]):
    global current_tuning_attempt_level, tuning_in_progress, kcpp_process_obj
    global last_launched_process_info, level_of_last_monitored_run

    kcpp_process_obj = monitored_kcpp_instance
    choices_dict = {}; default_choice_key = ""
    print_title("Post-Monitoring Options")
    print_info(f"Outcome of monitored launch: {outcome_from_monitor}")
    if last_approx_vram_used_kcpp_mb is not None:
        print_info(f"Approx. KCPP VRAM Used: {last_approx_vram_used_kcpp_mb:.0f} MB")
    kcpp_is_still_running = kcpp_process_obj and kcpp_process_obj.poll() is None

    if "SUCCESS_LOAD_VRAM_OK" in outcome_from_monitor:
        print_success("KCPP loaded successfully (VRAM OK).")
        choices_dict = { "u": "✅ Accept & Use this KCPP instance", "s": "💾 Save as Good, Stop KCPP & Continue Tuning (Try More GPU)", "g": "⚙️ Stop KCPP & Try More GPU (This Session)", "c": "⚙️ Stop KCPP & Try More CPU (This Session)", "q": "↩️ Stop KCPP, Save & Return to Tuning Menu" }
        default_choice_key = "u" if kcpp_is_still_running else "s"
    elif "SUCCESS_LOAD_VRAM_TIGHT" in outcome_from_monitor:
        print_warning("KCPP loaded, but VRAM is tight!")
        choices_dict = { "a": "⚠️ Auto-Adjust (Stop KCPP, More CPU & Continue Tuning)", "l": "🚀 Launch Anyway / Keep KCPP Running (Risky)", "q": "↩️ Stop KCPP, Save & Return to Tuning Menu" }
        default_choice_key = "a"
    elif "OOM" in outcome_from_monitor or "CRASH" in outcome_from_monitor or "PREMATURE_EXIT" in outcome_from_monitor:
        print_error("KCPP failed to load properly (OOM/Crash/Premature Exit).")
        choices_dict = { "c": "⚙️ Try More CPU & Continue Tuning", "q": "↩️ Save & Return to Tuning Menu"}
        default_choice_key = "c"
    elif "TIMEOUT" in outcome_from_monitor:
        print_warning("KCPP launch timed out (no success/OOM signal).")
        choices_dict = { "c": "⚙️ Try More CPU (Assume OOM & Continue Tuning)", "q": "↩️ Save & Return to Tuning Menu"}
        default_choice_key = "c"
    else:
        if kcpp_is_still_running:
            print_info("KCPP loaded (status: unknown VRAM or generic success).")
            choices_dict = {"u": "✅ Keep KCPP Running for Use", "q": "↩️ Stop KCPP, Save & Return to Tuning Menu"}
            default_choice_key = "u"
        else:
            print_warning("KCPP status unclear or it has already exited without specific error.")
            choices_dict = {"q": "↩️ Save & Return to Tuning Menu"}
            default_choice_key = "q"

    for key, desc in choices_dict.items(): print(f"  ({key.upper()}) {desc}")
    user_action_choice = prompt("Your choice?", choices=list(choices_dict.keys()), default=default_choice_key).lower()
    db_outcome_suffix_for_action = ""
    should_stop_monitored_kcpp = True
    if user_action_choice in ['u', 'l'] and kcpp_is_still_running: should_stop_monitored_kcpp = False

    if kcpp_is_still_running and should_stop_monitored_kcpp:
        print_info(f"Stopping monitored KCPP instance (PID: {kcpp_process_obj.pid})...")
        koboldcpp_core.kill_process(kcpp_process_obj.pid, force=True)
        kcpp_process_obj = None

    if user_action_choice == 'u':
        if kcpp_is_still_running:
            print_info("Keeping current KoboldCpp instance running for use.")
            db_outcome_suffix_for_action = "_USER_ACCEPTED_AND_USED"
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
            return session_control_result # Propagate this
        else:
            print_warning("KCPP is not running (or was stopped). Cannot accept and use. Returning to tuning menu.")
            db_outcome_suffix_for_action = "_ATTEMPTED_USE_BUT_NOT_RUNNING"
    elif user_action_choice == 'l':
        if kcpp_is_still_running:
            print_info("Keeping current (potentially risky) KoboldCpp instance running.")
            db_outcome_suffix_for_action = "_USER_KEPT_RISKY_RUNNING"
            last_launched_process_info["pid"] = kcpp_process_obj.pid
            last_launched_process_info["process_obj"] = kcpp_process_obj
            last_launched_process_info["command_list"] = last_proposed_command_list_for_db
            kcpp_process_obj = None
            effective_args_for_webui_risky = get_effective_session_args(current_tuning_model_path_local, current_tuning_session_base_args)
            port_for_webui_risky = effective_args_for_webui_risky.get("--port", "5000")
            if AUTO_OPEN_WEBUI: webbrowser.open(f"http://localhost:{port_for_webui_risky}")
            final_outcome_db_risky = outcome_from_monitor + db_outcome_suffix_for_action
            koboldcpp_core.save_config_to_db(DB_FILE, current_tuning_model_path_local, current_tuning_model_analysis_local, vram_at_decision_for_db, last_proposed_command_list_for_db, level_of_last_monitored_run, final_outcome_db_risky, last_approx_vram_used_kcpp_mb)
            tuning_in_progress = False
            session_ctrl_res_risky = kcpp_control_loop_cli(port_for_webui_risky, is_monitored_instance_being_controlled=False)
            return session_ctrl_res_risky
        else:
            print_info("KCPP not running. Re-launching this (potentially risky) configuration...")
            db_outcome_suffix_for_action = "_USER_RELAUNCHED_RISKY"
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
        db_outcome_suffix_for_action = "_USER_SAVED_GOOD_MORE_GPU"
        if current_tuning_attempt_level > current_tuning_min_level: current_tuning_attempt_level -= 1
        else: print_warning("Already at Max GPU, cannot go further.")
    elif user_action_choice == 'g':
        db_outcome_suffix_for_action = "_USER_WANTS_MORE_GPU_NOW"
        if current_tuning_attempt_level > current_tuning_min_level: current_tuning_attempt_level -= 1
        else: print_warning("Already at Max GPU, cannot go further.")
    elif user_action_choice == 'c' or user_action_choice == 'a':
        if user_action_choice == 'a': db_outcome_suffix_for_action = "_USER_ACCEPTED_AUTO_ADJUST_CPU"
        else: db_outcome_suffix_for_action = "_USER_TRIED_MORE_CPU_AFTER_FAIL" if "FAIL" in outcome_from_monitor.upper() or "OOM" in outcome_from_monitor.upper() or "TIMEOUT" in outcome_from_monitor.upper() else "_USER_WANTS_MORE_CPU_NOW"
        if current_tuning_attempt_level < current_tuning_max_level: current_tuning_attempt_level += 1
        else: print_warning("Already at Max CPU, cannot go further.")
    elif user_action_choice == 'q':
        db_outcome_suffix_for_action = "_USER_SAVED_RETURN_TUNING_MENU"

    if db_outcome_suffix_for_action:
        final_outcome_for_db_update_choice = outcome_from_monitor + db_outcome_suffix_for_action
        koboldcpp_core.save_config_to_db(DB_FILE, current_tuning_model_path_local, current_tuning_model_analysis_local, vram_at_decision_for_db, last_proposed_command_list_for_db, level_of_last_monitored_run, final_outcome_for_db_update_choice, last_approx_vram_used_kcpp_mb)
    
    return "continue_tuning" # Signal to continue the tuning loop


def launch_kobold_for_use_cli(command_list_to_run: List[str], db_outcome_on_success: str, level_for_db_record: Optional[int] = None):
    global last_launched_process_info, gguf_file_global, current_model_analysis_global

    if last_launched_process_info["process_obj"] and last_launched_process_info["process_obj"].poll() is None:
        print_info(f"Stopping previously launched KCPP (PID: {last_launched_process_info['pid']})...")
        koboldcpp_core.kill_process(last_launched_process_info["pid"])
    last_launched_process_info = {"pid": None, "process_obj": None, "command_list": []}

    print_info(f"Launching KoboldCpp for use...")
    vram_before_this_launch, _, _, _ = koboldcpp_core.get_available_vram_mb()
    final_level_for_db_log = level_for_db_record if level_for_db_record is not None else (current_tuning_attempt_level if tuning_in_progress else 0)
    model_p = current_tuning_model_path_local if tuning_in_progress else gguf_file_global
    model_a = current_tuning_model_analysis_local if tuning_in_progress else current_model_analysis_global

    koboldcpp_core.save_config_to_db(DB_FILE, model_p, model_a, vram_before_this_launch, command_list_to_run, final_level_for_db_log, db_outcome_on_success, None)
    launched_kcpp_process, launch_err_msg = koboldcpp_core.launch_process(command_list_to_run, capture_output=False, new_console=True)

    if launch_err_msg or not launched_kcpp_process:
        print_error(f"Failed to launch KoboldCPP: {launch_err_msg or 'Unknown error'}")
        koboldcpp_core.save_config_to_db(DB_FILE, model_p, model_a, vram_before_this_launch, command_list_to_run, final_level_for_db_log, "LAUNCH_FOR_USE_FAILED_CLI", None)
        return None
    else:
        print_success(f"KoboldCpp launched in new console (PID: {launched_kcpp_process.pid}).")
        last_launched_process_info["pid"] = launched_kcpp_process.pid
        last_launched_process_info["process_obj"] = launched_kcpp_process
        last_launched_process_info["command_list"] = command_list_to_run
        if AUTO_OPEN_WEBUI:
            args_dict_from_cmd = koboldcpp_core.args_list_to_dict(command_list_to_run)
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
    """Manages the interactive tuning session for a selected model."""
    global tuning_in_progress, current_tuning_attempt_level, current_tuning_min_level, current_tuning_max_level
    global current_tuning_session_base_args, current_tuning_model_path_local, current_tuning_model_analysis_local
    global gguf_file_global, current_model_analysis_global, level_of_last_monitored_run

    if not gguf_file_global or not current_model_analysis_global.get('filepath'):
        print_error("No model selected or analyzed. Please select a model first.")
        return "new_gguf"

    tuning_in_progress = True
    current_tuning_model_path_local = gguf_file_global
    current_tuning_model_analysis_local = current_model_analysis_global.copy()
    current_tuning_session_base_args = get_effective_session_args(current_tuning_model_path_local, {})

    print_title(f"Starting Auto-Tuning Session for: {os.path.basename(current_tuning_model_path_local)}")
    print_info(f"Analysis: Size ~{current_tuning_model_analysis_local.get('size_b', 'N/A')}B, "
               f"Quant ~{current_tuning_model_analysis_local.get('quant', 'N/A')}, "
               f"MoE: {'Yes' if current_tuning_model_analysis_local.get('is_moe') else 'No'}, "
               f"Layers: {current_tuning_model_analysis_local.get('num_layers', 'N/A')}")

    if current_tuning_model_analysis_local.get('is_moe'):
        current_tuning_min_level, current_tuning_max_level, initial_heuristic_level = -25, 10, -10
    else:
        current_tuning_min_level, current_tuning_max_level = -17, 9
        size_b_param = current_tuning_model_analysis_local.get('size_b', 0)
        initial_heuristic_level = -3 if isinstance(size_b_param, (int, float)) and size_b_param >=30 else (-5 if isinstance(size_b_param, (int, float)) and size_b_param >=20 else -7) if isinstance(size_b_param, (int, float)) else -5

    vram_available_now, _, _, _ = koboldcpp_core.get_available_vram_mb()
    best_historical_config = koboldcpp_core.find_best_historical_config(DB_FILE, current_tuning_model_analysis_local, vram_available_now)

    if best_historical_config and "attempt_level" in best_historical_config:
        print_info(f"Found historical config. Level: {best_historical_config['attempt_level']}, Outcome: {best_historical_config['outcome']}")
        hist_lvl, hist_outcome_str = best_historical_config['attempt_level'], best_historical_config.get('outcome', "")
        if hist_outcome_str.startswith("SUCCESS_LOAD_VRAM_OK") or hist_outcome_str.endswith("_USER_SAVED_GOOD_MORE_GPU"):
            initial_heuristic_level = max(current_tuning_min_level, hist_lvl - 1)
        elif hist_outcome_str.endswith("_USER_ACCEPTED_AUTO_ADJUST_CPU") or hist_outcome_str.endswith("_USER_TRIED_MORE_CPU_AFTER_FAIL") or \
             "OOM" in hist_outcome_str.upper() or "TIGHT" in hist_outcome_str.upper():
             initial_heuristic_level = min(current_tuning_max_level, hist_lvl + 1)
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

        if dependencies['rich']['module']:
            strategy_table = Table(title="Current Tensor Offload Strategy")
            strategy_table.add_column("Setting", style="cyan"); strategy_table.add_column("Value", style="yellow", overflow="fold")
            strategy_table.add_row("OT Level", f"{current_tuning_attempt_level}")
            range_desc_str = f"{current_tuning_min_level}=MaxGPU ... {current_tuning_max_level}={'SuperMaxCPU' if current_tuning_model_analysis_local.get('is_moe') else 'MaxCPU'}"
            strategy_table.add_row("Level Range", range_desc_str); strategy_table.add_row("Strategy Desc", strategy_description)
            strategy_table.add_row("OT Regex", ot_string_generated or "None (Max GPU layers if --gpulayers is high)")
            strategy_table.add_row("Effective GPU Layers", f"{gpu_layers_for_level}/{total_model_layers}")
            console.print(strategy_table)
        else:
            print(f"🛠️ OT Level: {current_tuning_attempt_level}\n   Range: {current_tuning_min_level}=MaxGPU to {current_tuning_max_level}={'SuperMaxCPU' if current_tuning_model_analysis_local.get('is_moe') else 'MaxCPU'}\n   Strategy: {strategy_description}\n   Regex: {(ot_string_generated or 'None')}\n   GPU Layers: {gpu_layers_for_level}/{total_model_layers}")

        args_for_kcpp_display_list = koboldcpp_core.build_command(current_tuning_model_path_local, ot_string_generated, current_tuning_model_analysis_local, current_tuning_session_base_args)
        display_full_command_list = koboldcpp_core.get_command_to_run(KOBOLDCPP_EXECUTABLE, args_for_kcpp_display_list)
        _, _, vram_info_message_str, _ = koboldcpp_core.get_available_vram_mb()
        if dependencies['rich']['module']:
            console.print(Panel(f"{vram_info_message_str}", title="Current GPU Info", style="green" if "error" not in vram_info_message_str.lower() else "red", expand=False))
        else: print(f"    GPU Status: {vram_info_message_str}")
        print_title("Proposed Command for This OT Level"); print_command(koboldcpp_core.format_command_for_display(display_full_command_list))
        menu_options_text = "(L)aunch & Monitor | (S)kip Tune & Launch Now | (G)PU More (↓Lvl) | (C)PU More (↑Lvl) | (E)dit Session Args | (P)ermanent Model Args | (H)istory (This Model) | (N)ew GGUF | (Q)uit Tuning"
        print_title("Tuning Actions"); print(menu_options_text)
        user_tuning_choice = prompt("Your choice", choices=['l','s','g','c','e', 'p', 'h','n','q'], default='l').lower().strip()

        if user_tuning_choice == 'l':
            post_monitoring_action_result = launch_and_monitor_for_tuning_cli() # This now returns string for next action
            if post_monitoring_action_result == "quit_script_leave_running":
                tuning_in_progress = False; return "quit_script_leave_running"
            elif post_monitoring_action_result == "new_gguf": # User chose to use instance, then new gguf
                tuning_in_progress = False; return "new_gguf"
            elif post_monitoring_action_result == "quit_script": # User chose to use instance, then quit
                tuning_in_progress = False; return "quit_script"
            # If "continue_tuning", loop continues
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
        KOBOLDCPP_EXECUTABLE = CONFIG["koboldcpp_executable"]
        DEFAULT_GGUF_DIR = CONFIG.get("default_gguf_dir", "")
        AUTO_OPEN_WEBUI = CONFIG.get("auto_open_webui", True)

    core_version_display = "N/A"
    if hasattr(koboldcpp_core, 'DEFAULT_CONFIG_TEMPLATE') and \
       isinstance(koboldcpp_core.DEFAULT_CONFIG_TEMPLATE, dict):
        core_version_display = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE.get('launcher_core_version', 'N/A')

    print_title(f"KoboldCpp Smart Launcher v{LAUNCHER_CLI_VERSION} (Core: {core_version_display})")
    sys_info_data = core_init_data.get("system_info", {})
    gpu_info_data = core_init_data.get("gpu_info", {})

    print_info(f"OS: {sys_info_data.get('os_name','N/A')} {sys_info_data.get('os_version','N/A')} | Python: {sys_info_data.get('python_version','N/A').split()[0]}")
    gpu_message_to_display = gpu_info_data.get('message', 'Could not detect GPU details.')
    if gpu_info_data.get("success"): print_info(f"GPU Info: {gpu_message_to_display}")
    else: print_warning(f"GPU Info: {gpu_message_to_display}")

    if not os.path.exists(KOBOLDCPP_EXECUTABLE):
        import shutil
        resolved_exe_path = shutil.which(KOBOLDCPP_EXECUTABLE) or shutil.which(os.path.basename(KOBOLDCPP_EXECUTABLE))
        if resolved_exe_path:
            print_info(f"KoboldCpp executable '{KOBOLDCPP_EXECUTABLE}' updated via PATH to: {resolved_exe_path}")
            KOBOLDCPP_EXECUTABLE = os.path.abspath(resolved_exe_path)
            CONFIG["koboldcpp_executable"] = KOBOLDCPP_EXECUTABLE
            koboldcpp_core.save_launcher_config(CONFIG)
        elif KOBOLDCPP_EXECUTABLE.lower().endswith(".py") and \
             os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), KOBOLDCPP_EXECUTABLE)):
             KOBOLDCPP_EXECUTABLE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), KOBOLDCPP_EXECUTABLE))
             print_info(f"Using KoboldCpp Python script found relative to launcher: {KOBOLDCPP_EXECUTABLE}")
        else:
            print_error(f"FATAL: KoboldCpp target '{KOBOLDCPP_EXECUTABLE}' not found and not in PATH. Please check the path in the config or during first setup."); sys.exit(1)

    while True:
        gguf_selection_result = select_gguf_file_cli()
        if gguf_selection_result is None: break
        if gguf_selection_result == "main_menu": continue

        gguf_file_global = gguf_selection_result
        current_model_analysis_global = koboldcpp_core.analyze_filename(gguf_file_global)
        tuning_session_outcome = run_model_tuning_session_cli() # This line was causing the error

        if tuning_session_outcome == "quit_script": break
        if tuning_session_outcome == "quit_script_leave_running":
            print_info("Exiting launcher. KoboldCpp may still be running as per user choice."); return
    print_title("KoboldCpp Smart Launcher Finished")


class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): pass

if __name__ == "__main__":
    if dependencies['pynvml']['module'] is None and platform.system() == "Windows":
        print_warning("PyNVML (for NVIDIA GPU monitoring) is not installed. NVIDIA GPU VRAM info might be limited.")
        print_warning("You can install it with: pip install pynvml")

    try:
        main_cli()
    except KeyboardInterrupt:
        print_warning("\nLauncher terminated by user (Ctrl+C).")
    except Exception as e_fatal_script:
        print_error(f"\nFATAL SCRIPT ERROR: {e_fatal_script}")
        import traceback
        traceback.print_exc()
    finally:
        print_info("Exiting. Cleaning up any lingering KoboldCpp process launched by this session...")
        if last_launched_process_info.get("process_obj") and last_launched_process_info["process_obj"].poll() is None:
            if last_launched_process_info.get("pid"):
                print_info(f"Stopping last directly launched KCPP process (PID: {last_launched_process_info['pid']})...")
                koboldcpp_core.kill_process(last_launched_process_info["pid"], force=True)

        if KOBOLDCPP_EXECUTABLE:
            kcpp_exe_basename = os.path.basename(KOBOLDCPP_EXECUTABLE)
            print_info(f"Performing cleanup sweep for processes like '{kcpp_exe_basename}'...")
            koboldcpp_core.kill_processes_by_name(kcpp_exe_basename)
            if kcpp_exe_basename.lower().endswith(".py"):
                 koboldcpp_core.kill_processes_by_name("python", cmdline_substr_filter=kcpp_exe_basename)
        print_info("Launcher exited.")
