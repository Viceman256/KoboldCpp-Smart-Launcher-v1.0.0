import os
import sys
import subprocess
import re
import json
import time
import threading
import signal
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, List, Any
import pathlib # Should be imported directly, not as pathlib.Path
import shutil
import platform
from pathlib import Path # Specific import for Path object

# --- Appdirs Integration ---
APP_NAME = "TensorTune"
APP_AUTHOR = "Viceman256"

appdirs_available = False
try:
    import appdirs
    appdirs_available = True
except ImportError:
    appdirs_available = False

def _get_user_app_config_dir():
    """Gets the user-specific configuration directory for the application."""
    if appdirs_available:
        path = appdirs.user_config_dir(APP_NAME, APP_AUTHOR)
    else:
        if sys.platform == "win32":
            path = os.path.join(Path.home(), "AppData", "Roaming", APP_NAME, "Config")
        elif sys.platform == "darwin":
            path = os.path.join(Path.home(), "Library", "Application Support", APP_NAME, "Config")
        else: # Linux and other XDG-like systems
            path = os.path.join(Path.home(), ".config", APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path

def _get_user_app_data_dir():
    """Gets the user-specific data directory for the application."""
    if appdirs_available:
        path = appdirs.user_data_dir(APP_NAME, APP_AUTHOR)
    else:
        if sys.platform == "win32":
            path = os.path.join(Path.home(), "AppData", "Local", APP_NAME, "Data")
        elif sys.platform == "darwin":
            path = os.path.join(Path.home(), "Library", "Application Support", APP_NAME, "Data")
        else: # Linux and other XDG-like systems
            path = os.path.join(Path.home(), ".local", "share", APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path

# --- Constants and Configuration ---
_CONFIG_FILE_BASENAME = "tensortune_config.json"
CONFIG_FILE = os.path.join(_get_user_app_config_dir(), _CONFIG_FILE_BASENAME)

_DB_FILE_BASENAME_DEFAULT = "tensortune_history.db"

DEFAULT_CONFIG_TEMPLATE = {
    "koboldcpp_executable": "koboldcpp.exe" if sys.platform == "win32" else "./koboldcpp",
    "default_gguf_dir": "",
    "last_used_gguf_dir": "",
    "db_file": _DB_FILE_BASENAME_DEFAULT,
    "first_run_completed": False,
    "first_run_intro_shown": False,
    "cli_use_tkinter_dialog": False,
    "color_mode": "auto",
    "auto_open_webui": True,
    "gpu_detection": {
        "nvidia": True,
        "amd": True,
        "intel": True,
        "apple": True
    },
    "model_specific_args": {},
    "default_args": {
        "--threads": "auto",
        "--usecublas": True,
        "--usehipblas": False,
        "--contextsize": "16384",
        "--promptlimit": "16000",
        "--flashattention": True,
        "--port": "5000",
        "--defaultgenamt": "2048",
        "--gpulayers": "auto",
        "--quantkv": "auto",
        "--blasbatchsize": "auto",
        "--nommap": False,
        "--lowvram": False,
        "--nblas": "auto"
    },
    "vram_safety_buffer_mb": 768,
    "min_vram_free_after_load_success_mb": 512,
    "use_psutil": True,
    "loading_timeout_seconds": 60,
    "vram_stabilization_wait_s": 3.0,
    "kobold_success_pattern": r"Starting Kobold API on port (\d+)",
    "oom_error_keywords": [
        "cuda out of memory", "outofmemory", "out of vram", "cublasstatusallocfailed",
        "ggml_cuda_host_malloc: failed to allocate", "ggml-cuda backend: failed to allocate",
        "failed to allocate memory on gpu", "vram allocation failed",
        "llama_new_context_with_model: failed to load model", "unable to initialize backend",
        "failed to load model", "model load failed", "segmentation fault", "aborted",
        "illegal instruction", "clblast error", "opencl error", "rocm error", "hip error",
        "cl_out_of_resources"
    ],
    "gpu_selection_mode": "auto", 
    "selected_gpu_index": 0,
    "override_vram_budget": False,
    "manual_vram_total_mb": 8192,
    "launcher_core_version": "1.0.0-TT" # Current core version
}

KOBOLDCPP_ARG_DEFINITIONS = [
    {"key": "--model", "name": "Model Path", "help": "Path to the GGUF model file.", "type_hint": "path", "category": "core"},
    {"key": "--port", "name": "Port", "help": "Network port for the KoboldCpp API (e.g., 5000).", "type_hint": "int", "category": "network"},
    {"key": "--threads", "name": "CPU Threads", "help": "Number of CPU threads to use. 'auto' for detection.", "type_hint": "str_auto_num", "category": "performance"},
    {"key": "--nblas", "name": "BLAS Threads", "help": "Number of CPU threads for BLAS. 'auto' for detection.", "type_hint": "str_auto_num", "category": "performance"},
    {"key": "--contextsize", "name": "Context Size", "help": "Max context window size in tokens (e.g., 4096, 16384).", "type_hint": "int", "category": "model_params"},
    {"key": "--promptlimit", "name": "Prompt Limit", "help": "Maximum length of prompt that can be processed (<= contextsize).", "type_hint": "int", "category": "model_params"},
    {"key": "--defaultgenamt", "name": "Default Generation Amount", "help": "Default number of tokens to generate (e.g., 1024, 2048).", "type_hint": "int", "category": "model_params"},
    {"key": "--gpulayers", "name": "GPU Layers", "help": "Number of model layers to offload to GPU. 'auto', 'off', or number (e.g., 35, 999 for max).", "type_hint": "str_auto_num", "category": "gpu"},
    {"key": "--nogpulayers", "name": "No GPU Layers", "help": "Explicitly disable GPU layer offloading (alternative to --gpulayers 0/off).", "type_hint": "bool_flag", "category": "gpu"},
    {"key": "--usecublas", "name": "Use CUBLAS (NVIDIA)", "help": "Enable CUBLAS for NVIDIA GPU acceleration.", "type_hint": "bool", "category": "gpu_nvidia"},
    {"key": "--usehipblas", "name": "Use hipBLAS (AMD ROCm)", "help": "Enable hipBLAS for AMD GPU acceleration (ROCm).", "type_hint": "bool", "category": "gpu_amd"},
    {"key": "--flashattention", "name": "Flash Attention", "help": "Enable FlashAttention optimization (if supported by model and GPU).", "type_hint": "bool", "category": "gpu"},
    {"key": "--nommap", "name": "No Memory Map", "help": "Disable memory mapping of the model file.", "type_hint": "bool", "category": "memory"},
    {"key": "--lowvram", "name": "Low VRAM Mode", "help": "Enable optimizations for systems with low VRAM.", "type_hint": "bool", "category": "memory"},
    {"key": "--quantkv", "name": "Quantize K/V Cache", "help": "Quantization for K/V cache. 'auto', 'off', or number (0=F32, 1=Q8_0, etc.).", "type_hint": "str_auto_num", "category": "gpu"},
    {"key": "--blasbatchsize", "name": "BLAS Batch Size", "help": "Batch size for BLAS operations. 'auto', 'off', or number (e.g., 128, 512).", "type_hint": "str_auto_num", "category": "performance"},
    {"key": "--overridetensors", "name": "Override Tensors", "help": "Advanced: Specify tensor offload patterns to CPU (regex).", "type_hint": "str_regex", "category": "gpu_advanced"},
]

pynvml_available = False
psutil_available = False
pyadlx_available = False
wmi_available = False
pyze_available = False
metal_available = False

try:
    import pynvml
    pynvml.nvmlInit()
    pynvml_available = True
except ImportError:
    pynvml_available = False
except pynvml.NVMLError: 
    pynvml_available = False

try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False

if sys.platform == "win32":
    try:
        import pyadlx
        if hasattr(pyadlx, 'ADLXHelper'):
             helper = pyadlx.ADLXHelper() 
             pyadlx_available = True
             del helper 
        else:
             pyadlx_available = False
    except ImportError:
        pyadlx_available = False
    except Exception: 
        pyadlx_available = False
    try:
        import wmi
        wmi_available = True
    except ImportError:
        wmi_available = False

try:
    import pyze.api as pyze_api
    if hasattr(pyze_api, 'zeInit') and hasattr(pyze_api, 'zeDriverGet'):
        pyze_available = True 
except ImportError:
    pyze_available = False
except Exception: 
    pyze_available = False

if sys.platform == "darwin":
    try:
        import metal
        if metal.MTLCopyAllDevices(): 
             metal_available = True
    except ImportError:
        metal_available = False
    except Exception: 
        metal_available = False

def adapt_datetime_iso(val: datetime) -> str:
    if val.tzinfo is None:
        val = val.astimezone(timezone.utc)
    else:
        val = val.astimezone(timezone.utc)
    return val.isoformat().replace("+00:00", "Z")

def convert_datetime(val: bytes) -> Optional[datetime]:
    try:
        dt_str = val.decode()
        if dt_str.endswith('Z'):
            dt_obj = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        else:
            dt_obj = datetime.fromisoformat(dt_str)
        if dt_obj.tzinfo is None:
            return dt_obj.replace(tzinfo=timezone.utc)
        return dt_obj.astimezone(timezone.utc)
    except ValueError:
        try:
            dt_obj = datetime.strptime(val.decode(), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            return dt_obj
        except ValueError:
            return None

sqlite3.register_adapter(datetime, adapt_datetime_iso)
sqlite3.register_converter("DATETIME", convert_datetime)
sqlite3.register_converter("TIMESTAMP", convert_datetime)

def save_launcher_config(config_to_save: dict) -> Tuple[bool, str]:
    try:
        config_copy_for_saving = json.loads(json.dumps(config_to_save))
        default_data_dir = _get_user_app_data_dir()
        if "db_file" in config_copy_for_saving:
            db_file_abs = os.path.abspath(config_copy_for_saving["db_file"])
            try:
                common_path = os.path.commonpath([db_file_abs, default_data_dir])
                if common_path == default_data_dir:
                    config_copy_for_saving["db_file"] = os.path.basename(db_file_abs)
            except ValueError: 
                pass
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_copy_for_saving, f, indent=4)
        return True, f"Configuration saved to {CONFIG_FILE}"
    except Exception as e:
        return False, f"Error saving configuration: {e}"

def load_config() -> Tuple[dict, bool, str]: # Return type changed: dict, bool (loaded_ok), message
    config_data = DEFAULT_CONFIG_TEMPLATE.copy()
    for key, value in DEFAULT_CONFIG_TEMPLATE.items():
        if isinstance(value, dict):
            config_data[key] = value.copy()

    default_db_basename = DEFAULT_CONFIG_TEMPLATE["db_file"]
    config_data["db_file"] = os.path.join(_get_user_app_data_dir(), default_db_basename)
    os.makedirs(os.path.dirname(config_data["db_file"]), exist_ok=True)

    current_template_version_str = DEFAULT_CONFIG_TEMPLATE.get("launcher_core_version", "0.0.0")
    try:
        current_version_tuple = tuple(map(int, current_template_version_str.split('.')))
    except ValueError:
        current_version_tuple = (0,0,0)
    
    config_was_migrated_or_keys_added = False 
    loaded_ok = False
    config_message = ""

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                user_config_loaded = json.load(f)
            loaded_ok = True
            config_message = f"Successfully loaded configuration from {CONFIG_FILE}"

            loaded_version_str = user_config_loaded.get("launcher_core_version", "0.0.0")
            try:
                loaded_version_tuple = tuple(map(int, loaded_version_str.split('.')))
            except ValueError:
                loaded_version_tuple = (0,0,0)

            if loaded_version_tuple < current_version_tuple:
                print(f"INFO: Launcher configuration version mismatch (Loaded: {loaded_version_str}, Current Template: {current_template_version_str}). Migrating and updating version.")
                user_config_loaded["launcher_core_version"] = current_template_version_str 
                config_was_migrated_or_keys_added = True

            # Merge loaded config into the template-based config_data
            for key, default_value_from_template in DEFAULT_CONFIG_TEMPLATE.items():
                if key in user_config_loaded:
                    if isinstance(default_value_from_template, dict) and isinstance(user_config_loaded[key], dict):
                        # For specific dicts like default_args, merge, don't just overwrite
                        if key in ["default_args", "gpu_detection", "model_specific_args"]:
                            merged_sub_dict = default_value_from_template.copy()
                            merged_sub_dict.update(user_config_loaded[key])
                            config_data[key] = merged_sub_dict
                        else: # For other dicts, user's value takes precedence if present
                            config_data[key] = user_config_loaded[key]
                    else: # Not a dict or types don't match, user's value takes precedence
                        config_data[key] = user_config_loaded[key]
                # If key not in user_config_loaded, it's already set from DEFAULT_CONFIG_TEMPLATE copy

            # Ensure all sub-keys within critical dicts are present, add from template if missing
            for dict_key_to_check in ["default_args", "gpu_detection", "model_specific_args"]: 
                if dict_key_to_check not in config_data: # Should not happen if copied from template initially
                     config_data[dict_key_to_check] = DEFAULT_CONFIG_TEMPLATE[dict_key_to_check].copy()
                     config_was_migrated_or_keys_added = True
                
                for sub_key, sub_default_val in DEFAULT_CONFIG_TEMPLATE[dict_key_to_check].items():
                    if sub_key not in config_data[dict_key_to_check]:
                        config_data[dict_key_to_check][sub_key] = sub_default_val
                        config_was_migrated_or_keys_added = True
            
            # Ensure all top-level keys from template exist in config_data, add if missing
            for key, default_value in DEFAULT_CONFIG_TEMPLATE.items():
                if key not in config_data:
                    config_data[key] = default_value.copy() if isinstance(default_value, dict) else default_value
                    config_was_migrated_or_keys_added = True


            db_file_val_from_loaded = user_config_loaded.get("db_file", default_db_basename)
            new_db_path = os.path.join(_get_user_app_data_dir(), db_file_val_from_loaded) if not os.path.isabs(db_file_val_from_loaded) else db_file_val_from_loaded
            if config_data["db_file"] != new_db_path:
                config_data["db_file"] = new_db_path
                config_was_migrated_or_keys_added = True # Path changed, consider it a migration for saving
            os.makedirs(os.path.dirname(config_data["db_file"]), exist_ok=True)
            
            if config_was_migrated_or_keys_added:
                save_ok, save_msg = save_launcher_config(config_data)
                if save_ok:
                    config_message += " (Config updated/migrated and saved)"
                else:
                    config_message += f" (Config updated/migrated but FAILED to save: {save_msg})"

        except Exception as e_load:
            loaded_ok = False
            config_message = f"Error loading or processing {CONFIG_FILE}: {e_load}. Falling back to defaults."
            # config_data is already a copy of DEFAULT_CONFIG_TEMPLATE.
            # db_file path has also been set correctly. Ensure version is current if we fallback.
            config_data["launcher_core_version"] = current_template_version_str
            # Attempt to save this default config if one didn't exist or was corrupt
            if not os.path.exists(CONFIG_FILE) or loaded_ok is False : # loaded_ok might be True if JSON loaded but processing failed
                save_ok, save_msg = save_launcher_config(config_data)
                if save_ok:
                    config_message += " (Saved default config)"
                else:
                    config_message += f" (Failed to save default config: {save_msg})"
    else:
        loaded_ok = False
        config_message = f"No config file found at {CONFIG_FILE}. Using default settings and saving. DB path: {config_data['db_file']}"
        config_data["launcher_core_version"] = current_template_version_str # Ensure version is current for new file
        save_ok, save_msg = save_launcher_config(config_data)
        if not save_ok:
             config_message += f" (Failed to save default config: {save_msg})"

    return config_data, loaded_ok, config_message

def export_config_to_file(config_data_to_export: dict, export_filepath: str) -> Tuple[bool, str]:
    try:
        config_copy_for_saving = json.loads(json.dumps(config_data_to_export))
        default_data_dir = _get_user_app_data_dir()
        if "db_file" in config_copy_for_saving:
            db_file_abs = os.path.abspath(config_copy_for_saving["db_file"])
            try:
                common_path = os.path.commonpath([db_file_abs, default_data_dir])
                if common_path == default_data_dir: 
                    config_copy_for_saving["db_file"] = os.path.basename(db_file_abs) 
            except ValueError:
                pass 
        with open(export_filepath, 'w', encoding='utf-8') as f:
            json.dump(config_copy_for_saving, f, indent=4)
        return True, f"Configuration exported successfully to {export_filepath}"
    except Exception as e:
        return False, f"Error during configuration export: {e}"

def import_config_from_file(import_filepath: str) -> Tuple[Optional[dict], str]:
    try:
        with open(import_filepath, 'r', encoding='utf-8') as f:
            imported_data = json.load(f)
        
        essential_keys = ["koboldcpp_executable", "default_args", "launcher_core_version"]
        missing_keys = [k for k in essential_keys if k not in imported_data]
        if missing_keys:
            return None, f"Imported file is missing essential configuration keys: {', '.join(missing_keys)}."
        if not isinstance(imported_data.get("default_args"), dict):
            return None, "Imported 'default_args' is not a valid dictionary structure."
        
        for key in ["model_specific_args", "gpu_detection"]:
            if key not in imported_data or not isinstance(imported_data[key], dict):
                imported_data[key] = DEFAULT_CONFIG_TEMPLATE[key].copy()
        
        return imported_data, "Configuration data imported successfully from file."
    except json.JSONDecodeError:
        return None, "Error: Imported file is not valid JSON."
    except FileNotFoundError:
        return None, f"Error: Import file not found at {import_filepath}."
    except Exception as e:
        return None, f"Error importing configuration data: {e}"

def init_db(db_file):
    conn = None
    try:
        db_dir = os.path.dirname(db_file)
        if db_dir: 
            os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS launch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_filepath TEXT NOT NULL,
                model_size_b REAL, model_quant_type TEXT, is_moe BOOLEAN,
                vram_at_launch_decision_mb INTEGER, kobold_args_json TEXT,
                attempt_level_used INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                launch_outcome TEXT, approx_vram_used_kcpp_mb INTEGER,
                UNIQUE(model_filepath, vram_at_launch_decision_mb, kobold_args_json, attempt_level_used)
            )
        ''')
        cols_to_check = {"launch_outcome": "TEXT", "approx_vram_used_kcpp_mb": "INTEGER"}
        table_info = cursor.execute("PRAGMA table_info(launch_history)").fetchall()
        existing_cols = [col_info[1] for col_info in table_info]
        for col, col_type in cols_to_check.items():
            if col not in existing_cols:
                cursor.execute(f"ALTER TABLE launch_history ADD COLUMN {col} {col_type}")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lh_model_details ON launch_history (model_filepath, model_quant_type, is_moe);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lh_timestamp ON launch_history (timestamp DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lh_model_filepath ON launch_history (model_filepath);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lh_launch_outcome ON launch_history (launch_outcome);")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_lh_lookup_for_best ON launch_history (
                model_filepath, model_quant_type, is_moe,
                launch_outcome, vram_at_launch_decision_mb, attempt_level_used, timestamp DESC
            );
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_lh_failures ON launch_history (
                model_filepath, model_quant_type, is_moe, launch_outcome, 
                vram_at_launch_decision_mb, attempt_level_used
            );
        """)
        conn.commit()
        return True, f"Database initialized successfully at {db_file}"
    except sqlite3.Error as e:
        return False, f"Database initialization error: {e}"
    finally:
        if conn:
            conn.close()

def save_config_to_db(db_file, model_filepath, model_analysis, vram_at_decision_mb, command_args_list_with_exe, attempt_level, outcome, approx_vram_used_kcpp_mb=None):
    conn = None
    try:
        conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        cursor = conn.cursor()
        num_prefix_items_to_skip = 1 
        if command_args_list_with_exe and command_args_list_with_exe[0].lower() == sys.executable.lower() and \
           len(command_args_list_with_exe) > 1 and \
           (command_args_list_with_exe[1].lower().endswith(".py") or os.path.basename(command_args_list_with_exe[1].lower()) == os.path.basename(DEFAULT_CONFIG_TEMPLATE["koboldcpp_executable"].lower())):
            num_prefix_items_to_skip = 2 

        args_to_save_list = command_args_list_with_exe[num_prefix_items_to_skip:] if command_args_list_with_exe else []
        args_json_str = json.dumps(args_to_save_list) 

        vram_at_decision_mb_int = int(vram_at_decision_mb) if vram_at_decision_mb is not None else None
        approx_vram_used_kcpp_mb_int = int(approx_vram_used_kcpp_mb) if approx_vram_used_kcpp_mb is not None else None
        current_timestamp = datetime.now(timezone.utc)

        model_size_to_db = model_analysis.get('size_b')
        if isinstance(model_size_to_db, str): model_size_to_db = None 
        elif model_size_to_db is not None: model_size_to_db = float(model_size_to_db)

        try:
            cursor.execute('''
                INSERT INTO launch_history
                (model_filepath, model_size_b, model_quant_type, is_moe, vram_at_launch_decision_mb,
                 kobold_args_json, attempt_level_used, launch_outcome, approx_vram_used_kcpp_mb, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (model_filepath, model_size_to_db, model_analysis.get('quant'), 
                  model_analysis.get('is_moe', False), vram_at_decision_mb_int, 
                  args_json_str, attempt_level, outcome, approx_vram_used_kcpp_mb_int, current_timestamp))
            success_msg = f"Saved new launch record to database (Outcome: {outcome})."
        except sqlite3.IntegrityError: 
            cursor.execute('''
                UPDATE launch_history SET launch_outcome = ?, approx_vram_used_kcpp_mb = ?, timestamp = ?
                WHERE model_filepath = ? 
                  AND (vram_at_launch_decision_mb = ? OR (vram_at_launch_decision_mb IS NULL AND ? IS NULL))
                  AND kobold_args_json = ? 
                  AND attempt_level_used = ?
            ''', (outcome, approx_vram_used_kcpp_mb_int, current_timestamp, model_filepath,
                  vram_at_decision_mb_int, vram_at_decision_mb_int, 
                  args_json_str, attempt_level))
            if cursor.rowcount == 0: 
                return False, f"Failed to update existing DB entry (IntegrityError but no row updated for outcome {outcome})."
            else:
                success_msg = f"Updated existing launch record in database (Outcome: {outcome})."
        
        conn.commit()
        return True, success_msg
    except Exception as e:
        return False, f"Could not save/update launch record to DB: {type(e).__name__}: {e}"
    finally:
        if conn: 
            conn.close()

def find_best_historical_config(db_file, current_model_analysis, current_available_dedicated_vram_mb, config_snapshot):
    conn = None
    try:
        conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        cursor = conn.cursor()
        vram_tolerance_percent_oom_avoid = float(config_snapshot.get("vram_historical_oom_avoid_tolerance_percent", 0.10))
        vram_tolerance_percent_success_match = float(config_snapshot.get("vram_historical_success_match_tolerance_percent", 0.25))
        model_size_tolerance_b = float(config_snapshot.get("model_size_historical_match_tolerance_b", 0.5))
        vram_safety_buffer_mb = int(config_snapshot.get("vram_safety_buffer_mb", 768))
        current_vram_for_query = float(current_available_dedicated_vram_mb) if current_available_dedicated_vram_mb is not None else 0.0
        target_vram_for_sort = current_vram_for_query if current_vram_for_query > 0 else 8192.0
        model_size_query_val = current_model_analysis.get('size_b')
        model_size_query_for_db = None
        if model_size_query_val is not None and not isinstance(model_size_query_val, str):
            try: model_size_query_for_db = float(model_size_query_val)
            except (ValueError, TypeError): pass
        problematic_levels_query = """
            SELECT DISTINCT attempt_level_used
            FROM launch_history
            WHERE model_filepath = ? AND model_quant_type = ? AND is_moe = ?
              AND (launch_outcome LIKE '%OOM%' OR launch_outcome LIKE '%CRASH%' OR launch_outcome LIKE '%TIGHT%')
              AND vram_at_launch_decision_mb IS NOT NULL
              AND vram_at_launch_decision_mb >= (? * (1 - ?))
        """
        cursor.execute(problematic_levels_query, (
            current_model_analysis['filepath'], current_model_analysis.get('quant'),
            current_model_analysis.get('is_moe', False),
            current_vram_for_query,
            vram_tolerance_percent_oom_avoid
        ))
        failed_levels_list = [row[0] for row in cursor.fetchall()]
        where_clause_failed_levels_filter = ""
        params_for_failed_levels_filter = []
        if failed_levels_list:
            placeholders = ','.join(['?'] * len(failed_levels_list))
            where_clause_failed_levels_filter = f"""
              AND NOT (
                    h.attempt_level_used IN ({placeholders})
                    AND ? <= COALESCE((SELECT MAX(sub_h.vram_at_launch_decision_mb)
                                      FROM launch_history sub_h
                                      WHERE sub_h.model_filepath = h.model_filepath
                                        AND sub_h.model_quant_type = h.model_quant_type
                                        AND sub_h.is_moe = h.is_moe
                                        AND sub_h.attempt_level_used = h.attempt_level_used
                                        AND (sub_h.launch_outcome LIKE '%OOM%' OR sub_h.launch_outcome LIKE '%CRASH%')
                                     ), 0) * (1 + ?)
                  )
            """
            params_for_failed_levels_filter.extend(failed_levels_list)
            params_for_failed_levels_filter.append(current_vram_for_query)
            params_for_failed_levels_filter.append(vram_tolerance_percent_oom_avoid)
        query = f"""
            SELECT
                h.kobold_args_json, h.attempt_level_used, h.vram_at_launch_decision_mb,
                h.launch_outcome, h.approx_vram_used_kcpp_mb
            FROM launch_history h
            WHERE h.model_filepath = ? AND h.model_quant_type = ? AND h.is_moe = ?
              AND (? IS NULL OR h.model_size_b IS NULL OR ABS(h.model_size_b - ?) < ?)
              AND (
                  h.vram_at_launch_decision_mb IS NULL OR
                  h.vram_at_launch_decision_mb BETWEEN
                     (? * (1 - ?)) AND (? * (1 + ?))
                  )
              {where_clause_failed_levels_filter}
            ORDER BY
              CASE WHEN h.launch_outcome LIKE 'SUCCESS_USER_CONFIRMED%' THEN 0
                   WHEN h.launch_outcome LIKE '%_USER_SAVED_GOOD_GPU_%' THEN 1
                   ELSE 10 END ASC,
              CASE WHEN h.launch_outcome LIKE 'SUCCESS_LOAD_VRAM_OK%' AND
                        h.approx_vram_used_kcpp_mb IS NOT NULL AND
                        (h.approx_vram_used_kcpp_mb + ?) < ? THEN 2
                   ELSE 10 END ASC,
              CASE WHEN h.launch_outcome LIKE 'SUCCESS%' AND
                        h.approx_vram_used_kcpp_mb IS NOT NULL AND
                        (h.approx_vram_used_kcpp_mb + ?) < ? THEN 3
                   ELSE 10 END ASC,
              CASE WHEN h.launch_outcome LIKE 'SUCCESS_USER_DIRECT_LAUNCH%' THEN 4
                   WHEN h.launch_outcome LIKE '%_USER_ACCEPTED_TUNED_%' THEN 5
                   ELSE 10 END ASC,
              CASE WHEN h.launch_outcome LIKE 'SUCCESS%' THEN 6 ELSE 10 END ASC,
              ABS(COALESCE(h.vram_at_launch_decision_mb, ?) - ?) ASC,
              h.attempt_level_used ASC,
              h.timestamp DESC
            LIMIT 1
        """
        base_params = [
            current_model_analysis['filepath'], current_model_analysis.get('quant'),
            current_model_analysis.get('is_moe', False),
            model_size_query_for_db, model_size_query_for_db, model_size_tolerance_b,
            current_vram_for_query, vram_tolerance_percent_success_match,
            current_vram_for_query, vram_tolerance_percent_success_match,
        ]
        base_params.extend(params_for_failed_levels_filter)
        base_params.extend([
            vram_safety_buffer_mb, current_vram_for_query,
            vram_safety_buffer_mb, current_vram_for_query,
            target_vram_for_sort, target_vram_for_sort
        ])
        cursor.execute(query, tuple(base_params))
        row = cursor.fetchone()
        if row:
            try:
                loaded_args_list = json.loads(row[0])
                return {"args_list": [str(item) for item in loaded_args_list],
                        "attempt_level": row[1],
                        "historical_vram_mb": row[2],
                        "outcome": row[3],
                        "approx_vram_used_kcpp_mb": row[4]}
            except json.JSONDecodeError:
                return None
        return None
    except sqlite3.Error as e_sql:
        print(f"Database error in find_best_historical_config: {e_sql}")
        return None
    except Exception as e_general:
        print(f"Unexpected error in find_best_historical_config: {e_general}")
        return None
    finally:
        if conn:
            conn.close()

def get_history_entries(db_file, limit=50):
    conn = None
    try:
        conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT model_filepath, model_size_b, model_quant_type, is_moe,
                   vram_at_launch_decision_mb, attempt_level_used, launch_outcome,
                   approx_vram_used_kcpp_mb, timestamp
            FROM launch_history ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"DB Error fetching history: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_system_info():
    info = {"cpu_model": "Unknown", "cpu_cores_physical": "N/A", "cpu_cores_logical": "N/A",
            "ram_total_gb": 0, "ram_free_gb": 0, "ram_used_percent": 0,
            "os_name": sys.platform, "os_version": "Unknown", "python_version": sys.version}
    try:
        if sys.platform == "win32": info["os_name"] = "Windows"; info["os_version"] = platform.version()
        elif sys.platform == "linux":
            info["os_name"] = "Linux"
            try:
                with open('/etc/os-release') as f:
                    info["os_version"] = next((l.split('=')[1].strip().strip('"')
                                               for l in f if l.startswith('PRETTY_NAME=')), "Unknown")
            except Exception: info["os_version"] = "Unknown (os-release error)"
        elif sys.platform == "darwin": info["os_name"] = "macOS"; info["os_version"] = platform.mac_ver()[0]

        if sys.platform == "win32": info["cpu_model"] = platform.processor()
        elif sys.platform == "linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    info["cpu_model"] = next((l.split(':')[1].strip()
                                              for l in f if l.startswith('model name')), "Unknown")
            except Exception: info["cpu_model"] = "Unknown (cpuinfo error)"
        elif sys.platform == "darwin":
            try:
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                        capture_output=True, text=True, check=False, timeout=2)
                if result.returncode == 0 and result.stdout.strip():
                    info["cpu_model"] = result.stdout.strip()
                else: info["cpu_model"] = "Unknown (sysctl error)"
            except Exception: info["cpu_model"] = "Unknown (sysctl exception)"
    except Exception: pass

    if psutil_available:
        try:
            info["cpu_cores_physical"] = psutil.cpu_count(logical=False) or "N/A"
            info["cpu_cores_logical"] = psutil.cpu_count(logical=True) or "N/A"
            mem = psutil.virtual_memory()
            info["ram_total_gb"] = round(mem.total / (1024**3), 1)
            info["ram_free_gb"] = round(mem.available / (1024**3), 1)
            info["ram_used_percent"] = round(mem.percent, 1)
        except Exception: pass
    return info

def list_nvidia_gpus() -> List[Dict[str, Any]]:
    if not pynvml_available: return []
    gpus = []
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name_raw = pynvml.nvmlDeviceGetName(handle)
            name = name_raw.decode('utf-8') if isinstance(name_raw, bytes) else str(name_raw)
            gpus.append({"id": i, "name": name})
    except pynvml.NVMLError:
        pass 
    return gpus

def list_amd_gpus_windows() -> List[Dict[str, Any]]:
    gpus = []
    if pyadlx_available:
        try:
            helper = pyadlx.ADLXHelper()
            adlx_gpus_list = helper.get_gpus() 
            amd_gpu_idx_counter = 0
            for gpu_obj in adlx_gpus_list: 
                if hasattr(gpu_obj, 'name') and ("amd" in gpu_obj.name.lower() or "radeon" in gpu_obj.name.lower()):
                    gpus.append({"id": amd_gpu_idx_counter, "name": gpu_obj.name, "source": "pyadlx"})
                    amd_gpu_idx_counter += 1
            if gpus: return gpus 
        except Exception:
            pass 

    if wmi_available:
        try:
            c = wmi.WMI()
            wmi_gpu_idx = 0
            for gpu_wmi_item in c.Win32_VideoController():
                if hasattr(gpu_wmi_item, 'Name') and ("amd" in gpu_wmi_item.Name.lower() or "radeon" in gpu_wmi_item.Name.lower()):
                    if not any(g['name'] == gpu_wmi_item.Name and g.get('source') == 'pyadlx' for g in gpus):
                        gpus.append({"id": wmi_gpu_idx, "name": gpu_wmi_item.Name, "source": "wmi"})
                    wmi_gpu_idx += 1
        except Exception:
            pass 
    return gpus

def list_amd_gpus_linux() -> List[Dict[str, Any]]:
    gpus = []
    try:
        if subprocess.run(["which", "rocm-smi"], capture_output=True, text=True, check=False, timeout=2).returncode != 0:
            return [] 
        json_proc = subprocess.run(["rocm-smi", "--showproductname", "--json"], capture_output=True, text=True, check=True, timeout=5, errors='replace')
        data = json.loads(json_proc.stdout)
        gpu_idx_counter = 0
        sorted_card_keys = sorted([key for key in data.keys() if key.startswith("card")], key=lambda x: int(x.replace("card","")))

        for card_key in sorted_card_keys:
            gpu_name = data[card_key].get("Card SKU", data[card_key].get("Card series", f"AMD GPU {gpu_idx_counter}"))
            gpus.append({"id": gpu_idx_counter, "name": gpu_name})
            gpu_idx_counter +=1
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass 
    except Exception: 
        pass
    return gpus

def list_intel_gpus() -> List[Dict[str, Any]]:
    if not pyze_available: return []
    gpus = []
    idx_counter = 0
    try:
        if pyze_api.zeInit(0) != pyze_api.ZE_RESULT_SUCCESS: return [] 
        num_drivers_ptr = pyze_api.new_uint32_tp(); pyze_api.zeDriverGet(num_drivers_ptr, None)
        num_drivers = pyze_api.uint32_tp_value(num_drivers_ptr); pyze_api.delete_uint32_tp(num_drivers_ptr)
        if num_drivers > 0:
            drivers_array = pyze_api.new_ze_driver_handle_t_array(num_drivers)
            pyze_api.zeDriverGet(pyze_api.new_uint32_tp_assign(num_drivers), drivers_array)
            for i in range(num_drivers):
                driver = pyze_api.ze_driver_handle_t_array_getitem(drivers_array, i)
                num_devices_ptr = pyze_api.new_uint32_tp(); pyze_api.zeDeviceGet(driver, num_devices_ptr, None)
                num_devices = pyze_api.uint32_tp_value(num_devices_ptr); pyze_api.delete_uint32_tp(num_devices_ptr)
                if num_devices > 0:
                    devices_array = pyze_api.new_ze_device_handle_t_array(num_devices)
                    pyze_api.zeDeviceGet(driver, pyze_api.new_uint32_tp_assign(num_devices), devices_array)
                    for j in range(num_devices):
                        device = pyze_api.ze_device_handle_t_array_getitem(devices_array, j)
                        props = pyze_api.ze_device_properties_t(); props.stype = pyze_api.ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
                        pyze_api.zeDeviceGetProperties(device, props)
                        if props.type == pyze_api.ZE_DEVICE_TYPE_GPU:
                            gpu_name = bytes(props.name).decode('utf-8', errors='ignore').rstrip('\x00')
                            gpus.append({"id": idx_counter, "name": gpu_name}) 
                            idx_counter += 1
                    pyze_api.delete_ze_device_handle_t_array(devices_array)
            pyze_api.delete_ze_driver_handle_t_array(drivers_array)
    except Exception: pass 
    return gpus

def list_apple_gpus() -> List[Dict[str, Any]]:
    if not (sys.platform == "darwin" and metal_available): return []
    gpus = []
    try:
        devices = metal.MTLCopyAllDevices()
        for i, device in enumerate(devices):
            name_suffix = " (Low Power)" if hasattr(device, 'isLowPower') and device.isLowPower() else ""
            gpus.append({"id": i, "name": device.name() + name_suffix})
    except Exception: pass
    return gpus

def get_gpu_info_nvidia(device_index: int = 0) -> Optional[Dict[str, Any]]:
    if not pynvml_available: return None
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0 or device_index >= device_count or device_index < 0:
            return {"success": False, "type": "NVIDIA_NONE_FOUND" if device_count == 0 else "NVIDIA_BAD_INDEX",
                    "message": f"NVML: {'No NVIDIA GPUs detected' if device_count == 0 else f'Index {device_index} out of range ({device_count} GPUs found)'}."}
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mb, total_mb, used_mb = mem_info.free / (1024**2), mem_info.total / (1024**2), mem_info.used / (1024**2)
        name_raw = pynvml.nvmlDeviceGetName(handle)
        name = name_raw.decode('utf-8') if isinstance(name_raw, bytes) else str(name_raw)
        used_percent = (used_mb / total_mb * 100) if total_mb > 0 else 0.0
        return {"success": True, "type": "NVIDIA", "name": f"{name} (ID {device_index})",
                "free_mb": round(free_mb,1), "total_mb": round(total_mb,1), "used_percent": round(used_percent,1),
                "message": f"NVIDIA {name} (ID {device_index}): {free_mb:.0f}/{total_mb:.0f}MB free ({used_percent:.1f}% used) [NVML]"}
    except pynvml.NVMLError as e_nvml:
        if e_nvml.value in [pynvml.NVML_ERROR_FUNCTION_NOT_FOUND, pynvml.NVML_ERROR_LIBRARY_NOT_FOUND, pynvml.NVML_ERROR_DRIVER_NOT_LOADED]:
            return {"success": False, "type": "NVIDIA_DRIVER_ISSUE", "message": f"NVML Error (Driver/Lib): {e_nvml}"}
        return {"success": False, "type": "NVIDIA_ERROR", "message": f"NVML Error (ID {device_index}): {e_nvml}"}
    except Exception as e_generic:
        return {"success": False, "type": "NVIDIA_ERROR", "message": f"NVIDIA Generic Error (ID {device_index}): {e_generic}"}

def _format_amd_win_message(method: str, status: str, detail: str = "") -> str:
    return f"AMD Win ({method}): {status}. {detail}".strip()

def get_gpu_info_amd(device_index: int = 0) -> Optional[Dict[str, Any]]:
    if sys.platform == "linux":
        try:
            if subprocess.run(["which", "rocm-smi"], capture_output=True, text=True, check=False, timeout=2).returncode != 0:
                return {"success": False, "type": "AMD_ROCM_SMI_NOT_FOUND", "message": "rocm-smi not found."}

            target_card_key_for_query = f"card{device_index}"
            json_proc_all = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json"], capture_output=True, text=True, check=True, timeout=5, errors='replace')
            data_all = json.loads(json_proc_all.stdout)

            actual_card_key_to_use = None
            if target_card_key_for_query in data_all:
                actual_card_key_to_use = target_card_key_for_query
            elif device_index == 0 and len(data_all) > 0: 
                sorted_card_keys_from_data = sorted([k for k in data_all if k.startswith("card")], key=lambda x: int(x.replace("card","")))
                if sorted_card_keys_from_data:
                    actual_card_key_to_use = sorted_card_keys_from_data[0]
                    print(f"Warning: AMD GPU index {device_index} ('{target_card_key_for_query}') not directly found, using first available key: {actual_card_key_to_use}")
            
            if not actual_card_key_to_use:
                 return {"success": False, "type": "AMD_ROCM_SMI_BAD_INDEX", "message": f"rocm-smi: Card index {device_index} not found in JSON output keys: {list(data_all.keys())}."}

            gpu_data = data_all[actual_card_key_to_use]
            total_mem_bytes_str, used_mem_bytes_str = gpu_data.get("VRAM Total Memory (B)"), gpu_data.get("VRAM Used Memory (B)")
            if not total_mem_bytes_str or not used_mem_bytes_str:
                return {"success": False, "type": "AMD_ROCM_SMI_PARSE_ERROR", "message": f"VRAM info missing for {actual_card_key_to_use} in rocm-smi JSON."}

            total_mb_val, used_mb_val = int(total_mem_bytes_str) / (1024**2), int(used_mem_bytes_str) / (1024**2)
            free_mb_val = total_mb_val - used_mb_val
            gpu_name_val = gpu_data.get("Card SKU", gpu_data.get("Card series", f"AMD GPU {actual_card_key_to_use}"))
            used_percent_val = (used_mb_val / total_mb_val * 100) if total_mb_val > 0 else 0.0
            return {"success": True, "type": "AMD", "name": f"{gpu_name_val} (ID {device_index} -> {actual_card_key_to_use})",
                    "free_mb": round(free_mb_val,1), "total_mb": round(total_mb_val,1), "used_percent": round(used_percent_val,1),
                    "message": f"AMD {gpu_name_val} (ID {device_index} -> {actual_card_key_to_use}): {free_mb_val:.0f}/{total_mb_val:.0f}MB free ({used_percent_val:.1f}% used) [rocm-smi]"}
        except Exception as e_rocm:
            return {"success": False, "type": "AMD_LINUX_ERROR", "message": f"AMD Linux rocm-smi error (ID {device_index}): {e_rocm}"}

    elif sys.platform == "win32":
        if pyadlx_available:
            try:
                helper = pyadlx.ADLXHelper()
                all_gpus_adlx_objs = helper.get_gpus()
                amd_gpu_candidates_adlx = [gpu for gpu in all_gpus_adlx_objs if hasattr(gpu, 'name') and ("amd" in gpu.name.lower() or "radeon" in gpu.name.lower())]

                if amd_gpu_candidates_adlx and 0 <= device_index < len(amd_gpu_candidates_adlx):
                    amd_gpu_obj_selected = amd_gpu_candidates_adlx[device_index]
                    vram_info_adlx = amd_gpu_obj_selected.get_vram_usage()
                    total_mb_adlx = float(vram_info_adlx.get('vram_total_mb', 0))
                    used_mb_adlx = float(vram_info_adlx.get('vram_used_mb', 0))
                    if total_mb_adlx > 0:
                        free_mb_adlx = total_mb_adlx - used_mb_adlx
                        used_percent_adlx = (used_mb_adlx / total_mb_adlx * 100) if total_mb_adlx > 0 else 0.0
                        return {"success": True, "type": "AMD", "name": f"{amd_gpu_obj_selected.name} (ID {device_index})",
                                "free_mb": round(free_mb_adlx,1), "total_mb": round(total_mb_adlx,1), "used_percent": round(used_percent_adlx,1),
                                "message": _format_amd_win_message("pyADLX", "Success", f"{amd_gpu_obj_selected.name} (ID {device_index}): {free_mb_adlx:.0f}/{total_mb_adlx:.0f}MB free ({used_percent_adlx:.1f}% used)")}
                elif amd_gpu_candidates_adlx and device_index >= len(amd_gpu_candidates_adlx):
                     return {"success": False, "type": "AMD_ADLX_BAD_INDEX", "message": _format_amd_win_message("pyADLX", "Index out of range", f"Index {device_index} for {len(amd_gpu_candidates_adlx)} AMD GPUs found.")}
            except Exception: 
                pass 

        if wmi_available:
            try:
                c = wmi.WMI()
                video_controllers_wmi = c.Win32_VideoController()
                amd_gpu_candidates_wmi = [gpu for gpu in video_controllers_wmi if hasattr(gpu, 'Name') and ("amd" in gpu.Name.lower() or "radeon" in gpu.Name.lower())]

                if amd_gpu_candidates_wmi and 0 <= device_index < len(amd_gpu_candidates_wmi):
                    gpu_wmi_item_selected = amd_gpu_candidates_wmi[device_index]
                    total_bytes_wmi = 0
                    if hasattr(gpu_wmi_item_selected, 'AdapterRAM') and gpu_wmi_item_selected.AdapterRAM is not None:
                        try: total_bytes_wmi = int(gpu_wmi_item_selected.AdapterRAM)
                        except ValueError: total_bytes_wmi = 0
                    total_mb_wmi = total_bytes_wmi / (1024**2)
                    if total_mb_wmi > 0 :
                        return {"success": True, "type": "AMD", "name": f"{gpu_wmi_item_selected.Name} (ID {device_index})",
                                "free_mb": 0, "total_mb": round(total_mb_wmi,1), "used_percent": 0,
                                "message": _format_amd_win_message("WMI", "Success (limited info)", f"{gpu_wmi_item_selected.Name} (ID {device_index}): Total {total_mb_wmi:.0f}MB")}
                elif amd_gpu_candidates_wmi and device_index >= len(amd_gpu_candidates_wmi):
                     return {"success": False, "type": "AMD_WMI_BAD_INDEX", "message": _format_amd_win_message("WMI", "Index out of range", f"Index {device_index} for {len(amd_gpu_candidates_wmi)} AMD GPUs found.")}
                
                return {"success": False, "type": "AMD_WMI_NO_MATCH", "message": _format_amd_win_message("WMI", "No AMD/Radeon GPU with VRAM info found (or bad index).")}
            except Exception as e_wmi:
                return {"success": False, "type": "AMD_WMI_QUERY_ERROR", "message": _format_amd_win_message("WMI", "Error during WMI query", str(e_wmi))}
        
        final_message_parts = []
        if not pyadlx_available: final_message_parts.append("pyADLX unavailable")
        else: final_message_parts.append("pyADLX failed or found no AMD VRAM for index")
        if not wmi_available: final_message_parts.append("WMI unavailable")
        else: final_message_parts.append("WMI failed or found no AMD VRAM for index")
        return {"success": False, "type": "AMD_WIN_DETECTION_UNAVAILABLE", "message": _format_amd_win_message("All Methods", "Unavailable/Failed", ", ".join(final_message_parts))}
    return None 

def get_gpu_info_intel(device_index: int = 0) -> Optional[Dict[str, Any]]:
    if not pyze_available: return None
    try:
        if pyze_api.zeInit(0) != pyze_api.ZE_RESULT_SUCCESS:
            return {"success": False, "type": "INTEL_PYZE_INIT_ERROR", "message": "Intel pyze: zeInit failed"}
        
        num_drivers_ptr = pyze_api.new_uint32_tp()
        pyze_api.zeDriverGet(num_drivers_ptr, None)
        num_drivers = pyze_api.uint32_tp_value(num_drivers_ptr); pyze_api.delete_uint32_tp(num_drivers_ptr)
        if num_drivers == 0: return None

        drivers_array = pyze_api.new_ze_driver_handle_t_array(num_drivers)
        pyze_api.zeDriverGet(pyze_api.new_uint32_tp_assign(num_drivers), drivers_array)
        
        target_device_obj = None
        current_gpu_device_idx = 0 

        for i in range(num_drivers): 
            driver = pyze_api.ze_driver_handle_t_array_getitem(drivers_array, i)
            num_devices_ptr_drv = pyze_api.new_uint32_tp()
            pyze_api.zeDeviceGet(driver, num_devices_ptr_drv, None)
            num_devices_drv = pyze_api.uint32_tp_value(num_devices_ptr_drv); pyze_api.delete_uint32_tp(num_devices_ptr_drv)
            if num_devices_drv == 0: continue

            devices_array_drv = pyze_api.new_ze_device_handle_t_array(num_devices_drv)
            pyze_api.zeDeviceGet(driver, pyze_api.new_uint32_tp_assign(num_devices_drv), devices_array_drv)
            for j in range(num_devices_drv):
                device_candidate = pyze_api.ze_device_handle_t_array_getitem(devices_array_drv, j)
                props_candidate = pyze_api.ze_device_properties_t(); props_candidate.stype = pyze_api.ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
                pyze_api.zeDeviceGetProperties(device_candidate, props_candidate)
                if props_candidate.type == pyze_api.ZE_DEVICE_TYPE_GPU:
                    if current_gpu_device_idx == device_index:
                        target_device_obj = device_candidate
                        break 
                    current_gpu_device_idx += 1
            pyze_api.delete_ze_device_handle_t_array(devices_array_drv)
            if target_device_obj: break 
        pyze_api.delete_ze_driver_handle_t_array(drivers_array)

        if not target_device_obj:
             return {"success": False, "type": "INTEL_PYZE_BAD_INDEX", "message": f"Intel pyze: GPU Index {device_index} not found among {current_gpu_device_idx} GPU(s)."}

        props = pyze_api.ze_device_properties_t(); props.stype = pyze_api.ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
        pyze_api.zeDeviceGetProperties(target_device_obj, props)
        gpu_name = bytes(props.name).decode('utf-8', errors='ignore').rstrip('\x00')
        
        total_vram_bytes, mem_props_count_ptr = 0, pyze_api.new_uint32_tp()
        pyze_api.zeDeviceGetMemoryProperties(target_device_obj, mem_props_count_ptr, None)
        mem_props_count = pyze_api.uint32_tp_value(mem_props_count_ptr); pyze_api.delete_uint32_tp(mem_props_count_ptr)
        if mem_props_count > 0:
            mem_props_array_ptr = pyze_api.new_ze_device_memory_properties_t_array(mem_props_count)
            for k in range(mem_props_count): pyze_api.ze_device_memory_properties_t_stype_set(pyze_api.ze_device_memory_properties_t_array_getitem_ptr(mem_props_array_ptr, k), pyze_api.ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES)
            pyze_api.zeDeviceGetMemoryProperties(target_device_obj, pyze_api.new_uint32_tp_assign(mem_props_count), mem_props_array_ptr)
            for k in range(mem_props_count):
                if pyze_api.ze_device_memory_properties_t_array_getitem(mem_props_array_ptr, k).flags & pyze_api.ZE_DEVICE_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL:
                    total_vram_bytes += pyze_api.ze_device_memory_properties_t_array_getitem(mem_props_array_ptr, k).totalSize
            pyze_api.delete_ze_device_memory_properties_t_array(mem_props_array_ptr)
        total_mb = total_vram_bytes / (1024**2)
        return {"success": True, "type": "Intel", "name": f"{gpu_name} (ID {device_index})",
                "free_mb": 0, "total_mb": round(total_mb,1), "used_percent": 0, 
                "message": f"Intel {gpu_name} (ID {device_index}): Total {total_mb:.0f}MB (Free VRAM not via pyze)"}
    except Exception as e_pyze:
        return {"success": False, "type": "INTEL_PYZE_ERROR", "message": f"Intel pyze error (ID {device_index}): {e_pyze}"}

def get_gpu_info_apple_metal(device_index: int = 0) -> Optional[Dict[str, Any]]:
    if not (sys.platform == "darwin" and metal_available): return None
    try:
        devices = metal.MTLCopyAllDevices()
        if not devices or device_index >= len(devices) or device_index < 0:
            return {"success": False, "type": "APPLE_METAL_NO_DEVICE" if not devices else "APPLE_METAL_BAD_INDEX",
                    "message": f"Metal: {'No devices found' if not devices else f'Index {device_index} out of range ({len(devices)} found)'}."}
        
        selected_device = devices[device_index]
        gpu_name_raw = selected_device.name()
        gpu_name = str(gpu_name_raw) if gpu_name_raw else "Unknown Apple GPU"
        
        total_mb_approx = 0
        if hasattr(selected_device, 'recommendedMaxWorkingSetSize') and selected_device.recommendedMaxWorkingSetSize():
            total_mb_approx = selected_device.recommendedMaxWorkingSetSize() / (1024**2)
        
        current_allocated_size_mb = 0
        if hasattr(selected_device, 'currentAllocatedSize'):
            current_allocated_size_mb = selected_device.currentAllocatedSize() / (1024**2)
        
        free_mb_approx, used_percent_approx = 0, 0
        if total_mb_approx > 0:
            free_mb_approx = total_mb_approx - current_allocated_size_mb
            used_percent_approx = (current_allocated_size_mb / total_mb_approx * 100)

        return {"success": True, "type": "APPLE_METAL", "name": f"{gpu_name} (ID {device_index})",
                "free_mb": round(free_mb_approx,1), "total_mb": round(total_mb_approx,1), "used_percent": round(used_percent_approx,1),
                "message": f"Metal {gpu_name} (ID {device_index}): Approx {free_mb_approx:.0f}/{total_mb_approx:.0f}MB free ({used_percent_approx:.1f}% used) [Unified]"}
    except Exception as e_metal:
        return {"success": False, "type": "APPLE_METAL_ERROR", "message": f"Apple Metal error (ID {device_index}): {e_metal}"}

def get_available_vram_mb(current_config: Optional[Dict] = None,
                          target_gpu_type: Optional[str] = None,
                          target_gpu_index: Optional[int] = None
                         ) -> Tuple[float, float, str, dict]: # Returns: (budgeted_free, budgeted_total, message, rich_gpu_info_dict)
    effective_config = current_config if current_config else DEFAULT_CONFIG_TEMPLATE.copy()
    gpu_detection_prefs = effective_config.get("gpu_detection", DEFAULT_CONFIG_TEMPLATE["gpu_detection"].copy())
    
    cfg_target_type = effective_config.get("gpu_selection_mode", "auto")
    cfg_target_idx = effective_config.get("selected_gpu_index", 0)

    final_target_type = target_gpu_type if target_gpu_type and target_gpu_type != "auto" else \
                       (cfg_target_type if cfg_target_type != "auto" else None)
    final_target_idx = target_gpu_index if target_gpu_index is not None else cfg_target_idx

    raw_gpu_info_dict = None # This will store the info from the specific GPU function

    if final_target_type: 
        if final_target_type == "nvidia" and gpu_detection_prefs.get("nvidia", True):
            raw_gpu_info_dict = get_gpu_info_nvidia(final_target_idx)
        elif final_target_type == "amd" and gpu_detection_prefs.get("amd", True):
            raw_gpu_info_dict = get_gpu_info_amd(final_target_idx)
        elif final_target_type == "intel" and gpu_detection_prefs.get("intel", True):
            raw_gpu_info_dict = get_gpu_info_intel(final_target_idx)
        elif final_target_type == "apple" and gpu_detection_prefs.get("apple", True) and sys.platform == "darwin":
            raw_gpu_info_dict = get_gpu_info_apple_metal(final_target_idx)
        
        if not raw_gpu_info_dict : # Detection for this type failed or not enabled
             msg = f"Targeted GPU type '{final_target_type}' not enabled in detection preferences or detection failed for ID {final_target_idx}."
             raw_gpu_info_dict = {"type": "INVALID_TARGET_PREFS", "name":"N/A", "free_mb":0.0, "total_mb":0.0, "success": False, "message": msg}
    else: # Auto-detection logic
        all_detection_results = []
        if gpu_detection_prefs.get("nvidia", True):
            nvidia_info = get_gpu_info_nvidia(0) 
            if nvidia_info:
                all_detection_results.append(nvidia_info)
                if nvidia_info.get("success") and (nvidia_info.get("free_mb",0) > 0 or nvidia_info.get("total_mb",0) > 0):
                    raw_gpu_info_dict = nvidia_info
        
        if not raw_gpu_info_dict and gpu_detection_prefs.get("amd", True):
            amd_info = get_gpu_info_amd(0)
            if amd_info:
                all_detection_results.append(amd_info)
                if amd_info.get("success") and (amd_info.get("free_mb",0) > 0 or amd_info.get("total_mb",0) > 0):
                    raw_gpu_info_dict = amd_info

        if not raw_gpu_info_dict and gpu_detection_prefs.get("intel", True):
            intel_info = get_gpu_info_intel(0)
            if intel_info:
                all_detection_results.append(intel_info)
                if intel_info.get("success") and intel_info.get("total_mb",0) > 1000 and (intel_info.get("free_mb",0) > 0 or intel_info.get("total_mb",0) > 0) : 
                    raw_gpu_info_dict = intel_info

        if not raw_gpu_info_dict and sys.platform == "darwin" and gpu_detection_prefs.get("apple", True):
            metal_info = get_gpu_info_apple_metal(0)
            if metal_info:
                all_detection_results.append(metal_info)
                if metal_info.get("success") and metal_info.get("total_mb", 0) > 0:
                    raw_gpu_info_dict = metal_info
        
        if not raw_gpu_info_dict: # No successful detection with VRAM > 0
            # Prioritize showing an error from an attempted detection if one occurred
            raw_gpu_info_dict = {"type": "Unknown/None", "name": "N/A", "free_mb": 0.0, "total_mb": 0.0, "success": False, "message": "No dedicated GPU VRAM info detected/supported (auto-detect)."}
            for res_dict_iter in all_detection_results:
                if res_dict_iter and res_dict_iter.get("success"): # A successful detection, even if 0 VRAM, is preferred over no detection
                    raw_gpu_info_dict = res_dict_iter
                    break
            if not raw_gpu_info_dict.get("success"): # If still no success, pick best error
                for res_type_preference in ["AMD", "NVIDIA", "Intel", "APPLE_METAL"]: 
                    errors_of_type = [res for res in all_detection_results if res and res_type_preference in res.get("type","") and not res.get("success")]
                    if errors_of_type:
                        raw_gpu_info_dict = errors_of_type[0]
                        break
                if raw_gpu_info_dict.get("type") == "Unknown/None" or "No dedicated GPU" in raw_gpu_info_dict.get("message","") :
                    sys_ram_info = get_system_info()
                    msg_suffix = f" System RAM: {sys_ram_info['ram_free_gb']:.1f}/{sys_ram_info['ram_total_gb']:.1f}GB free."
                    raw_gpu_info_dict["message"] = raw_gpu_info_dict.get("message","").rstrip('.') + msg_suffix
    
    # Now, raw_gpu_info_dict contains the best info we could get about the hardware.
    # Apply manual VRAM budget override if active.
    final_return_dict = raw_gpu_info_dict.copy() # Start with raw HW info
    final_return_dict["override_active"] = False
    final_return_dict["total_mb_budgeted"] = raw_gpu_info_dict.get("total_mb", 0.0)
    final_return_dict["free_mb_budgeted"] = raw_gpu_info_dict.get("free_mb", 0.0)
    final_return_dict["used_percent_budgeted"] = raw_gpu_info_dict.get("used_percent", 0.0)

    if effective_config.get("override_vram_budget", False):
        manual_total_mb = float(effective_config.get("manual_vram_total_mb", 0))
        final_return_dict["override_active"] = True
        final_return_dict["total_mb_budgeted"] = manual_total_mb
        
        actual_hw_used_mb = raw_gpu_info_dict.get("total_mb", 0.0) - raw_gpu_info_dict.get("free_mb", 0.0)
        
        if manual_total_mb > 0:
            final_return_dict["free_mb_budgeted"] = max(0.0, manual_total_mb - actual_hw_used_mb)
            final_return_dict["used_percent_budgeted"] = round((actual_hw_used_mb / manual_total_mb * 100) if manual_total_mb > 0 else 0.0, 1)
        else: # Manual total is 0, so budgeted free is 0, used is 100%
            final_return_dict["free_mb_budgeted"] = 0.0
            final_return_dict["used_percent_budgeted"] = 100.0 if actual_hw_used_mb > 0 else 0.0
            
        # Prepend override info to the message
        override_msg_prefix = f"Manual Budget: {manual_total_mb:.0f}MB. "
        original_message = final_return_dict.get("message", "")
        if original_message.startswith("NVIDIA") or original_message.startswith("AMD") or original_message.startswith("Intel") or original_message.startswith("Metal"):
            # Insert after the GPU type/name part
            parts = original_message.split(":", 1)
            if len(parts) > 1:
                final_return_dict["message"] = f"{parts[0]}: {override_msg_prefix} ({parts[1].strip()})"
            else:
                final_return_dict["message"] = f"{override_msg_prefix} ({original_message})"
        else: # Generic message
            final_return_dict["message"] = f"{override_msg_prefix} ({original_message})"


    return (final_return_dict.get("free_mb_budgeted", 0.0), 
            final_return_dict.get("total_mb_budgeted", 0.0), 
            final_return_dict.get("message", "N/A"), 
            final_return_dict)


def analyze_filename(filepath: str) -> dict:
    filename_lower = os.path.basename(filepath).lower()
    analysis = {'filepath': filepath, 'is_moe': False, 'quant': 'unknown', 'size_b': 0, 'details': {}, 'num_layers': 32, 'estimated_vram_gb_full_gpu': 0.0}
    
    if 'moe' in filename_lower or 'mixtral' in filename_lower or 'grok' in filename_lower or re.search(r'-a\d+(\.\d+)?[bB]', filename_lower):
        analysis['is_moe'] = True
    
    quant_match = re.search(r'(q[2-8](?:_[0ksmKSML]{1,2})?|iq[1-4](?:_[smlxSMLX]{1,2})?|bpw\d+|bf16|fp16|f16|f32|ggml|exl\d|awq|gptq|q_k_l|q_k_m|q_k_s|q_k_xl)', filename_lower, re.I)
    if quant_match:
        analysis['quant'] = quant_match.group(1).upper().replace("FP16", "F16")
        
    size_match = re.search(r'(?<![a-zA-Z0-9_])(\d{1,3}(?:\.\d{1,2})?)[bB](?![piA-Z0-9_])', filename_lower) or \
                 re.search(r'(?<![a-zA-Z0-9_])(\d{1,3}(?:\.\d{1,2})?)[bB][-_]', filename_lower)
    if size_match:
        try:
            size_val = float(size_match.group(1))
            analysis['size_b'] = int(size_val) if size_val.is_integer() else size_val
        except ValueError:
            analysis['size_b'] = "N/A_ParseErr"
    
    current_size_b_val = analysis.get('size_b', 0)
    file_size_gb = os.path.getsize(filepath) / (1024**3) if os.path.exists(filepath) else 0

    if (isinstance(current_size_b_val, (int, float)) and current_size_b_val == 0) or not isinstance(current_size_b_val, (int, float)):
        if file_size_gb > 0 and analysis['quant'] != 'unknown':
            gb_per_b_param_map = {'IQ1': 0.28, 'Q2_0': 0.28, 'Q2_K_S': 0.28, 'IQ2_XS': 0.30, 'IQ2_S': 0.30, 'Q2_K': 0.30,'IQ2': 0.35, 'Q3_0': 0.35, 'Q3_K_S': 0.35, 'IQ3_XS': 0.40, 'IQ3_S': 0.40, 'Q3_K': 0.40, 'Q3_K_M': 0.40, 'Q3_K_L': 0.42,'IQ3_M': 0.50, 'IQ3_L': 0.50, 'Q4_0': 0.50, 'Q4_K_S': 0.53, 'IQ4_XS': 0.55, 'IQ4_S': 0.55, 'Q4_K_M': 0.56,'Q5_0': 0.60, 'Q5_K_S': 0.63, 'Q5_K_M': 0.66, 'Q5_1': 0.65,'Q6_K': 0.75, 'Q8_0': 1.05,'F16': 2.05, 'BF16': 2.05, 'F32': 4.05}
            est_b_from_file = 0
            quant_upper = analysis['quant'].upper()
            if quant_upper in gb_per_b_param_map:
                est_b_from_file = file_size_gb / gb_per_b_param_map[quant_upper]
            else:
                for qk, fv in gb_per_b_param_map.items():
                    if qk in quant_upper:
                        est_b_from_file = file_size_gb / fv
                        break
            if est_b_from_file == 0 and file_size_gb > 0: 
                est_b_from_file = file_size_gb / 0.6 
            
            if est_b_from_file > 0.5:
                common_sizes = sorted([1, 1.5, 2.7, 3, 7, 8, 11, 13, 15, 20, 22, 27, 30, 32, 33, 34, 35, 40, 47, 65, 70, 120, 180, 235]) 
                if analysis['is_moe'] and ("mixtral" in filename_lower or "8x7" in filename_lower) and abs(est_b_from_file - 47) < 5 : 
                    analysis['size_b'] = 47
                    analysis['details']['size_is_moe_special'] = "Mixtral 8x7B type"
                
                if not isinstance(analysis['size_b'], (int,float)) or analysis['size_b'] == 0: 
                    closest_s = min((s for s in common_sizes if isinstance(s, (int,float))), key=lambda x: abs(x - est_b_from_file), default=None)
                    if closest_s is not None and abs(closest_s - est_b_from_file) < closest_s * 0.25: 
                        analysis['size_b'] = int(closest_s) if float(closest_s).is_integer() else closest_s
                        analysis['details']['size_is_estimated_from_filesize'] = True
                    else:
                        analysis['size_b'] = round(est_b_from_file, 1)
                        analysis['details']['size_is_estimated_from_filesize_raw'] = True
        elif file_size_gb == 0 and not isinstance(analysis['size_b'], (int,float)): 
            analysis['size_b'] = "N/A_NoInfo"

    layer_patterns = [r'(\d+)l', r'l(\d+)', r'-(\d+)layers', r'(\d+)layers']
    model_layer_defaults = {
        'gemma': {'2b': 18, '7b': 28, 'default': 28}, 
        'llama': {'7b': 32, '13b': 40, '30b': 60, '34b':48, '65b': 80, '70b': 80, 'default': 32}, 
        'mistral': {'7b': 32, 'default': 32}, 
        'mixtral': {'8x7b': 32, 'default': 32}, 
        'qwen': {'default': 32}, 
        'phi': {'default': 32}, 
        'gpt-j': {'6b': 28, 'default': 28}, 
        'gpt-neox': {'20b': 44, 'default': 44}, 
        'pythia': {'default': 32}, 
        'falcon': {'7b': 32, '40b': 60, 'default': 32}, 
        'mamba': {'default': 64}
    }
    num_layers_val = None
    for p in layer_patterns:
        m = re.search(p, filename_lower)
        if m:
            try:
                num_layers_val = int(m.group(1))
                break 
            except ValueError:
                pass
                
    if num_layers_val is None: 
        size_b_for_layers = analysis['size_b'] if isinstance(analysis['size_b'], (int,float)) else 0.0
        if isinstance(size_b_for_layers, int): size_b_for_layers = float(size_b_for_layers) 

        for model_key_iter, layer_config_iter in model_layer_defaults.items():
            if model_key_iter in filename_lower:
                if isinstance(layer_config_iter, dict):
                    s_match_found_in_model_type = False
                    for size_str_iter, layers_val_iter in layer_config_iter.items():
                        if size_str_iter == 'default': continue 
                        try:
                            config_size_float = float(size_str_iter.replace('b',''))
                            if abs(size_b_for_layers - config_size_float) < 0.1: 
                                num_layers_val = layers_val_iter
                                s_match_found_in_model_type = True
                                break
                        except ValueError: pass 
                    
                    if s_match_found_in_model_type: break 
                    
                    if 'default' in layer_config_iter: 
                        num_layers_val = layer_config_iter['default']
                        break 
                break 

    if num_layers_val is None and isinstance(analysis['size_b'], (int, float)): 
        s_b = analysis['size_b']
        if s_b <= 1.5: num_layers_val = 22
        elif s_b <= 3: num_layers_val = 26
        elif s_b <= 7: num_layers_val = 32
        elif s_b <= 13: num_layers_val = 40
        elif s_b <= 20: num_layers_val = 48
        elif s_b <= 35: num_layers_val = 60
        elif s_b <= 70: num_layers_val = 80
        else: num_layers_val = 96
            
    analysis['num_layers'] = num_layers_val if num_layers_val is not None else 32 

    vram_gb_per_b_param = { 
        'F32': 4.5, 'BF16': 2.5, 'F16': 2.5, 'Q8_0': 1.5, 
        'Q6_K': 1.1, 'Q5_K_M': 0.95, 'Q5_K_S': 0.90, 'Q5_0': 0.90, 'Q5_1':0.95, 
        'Q4_K_M': 0.8, 'Q4_K_S': 0.75, 'Q4_0': 0.70, 
        'Q3_K_L': 0.65, 'Q3_K_M': 0.60, 'Q3_K_S': 0.55, 
        'Q2_K': 0.50, 'Q2_0': 0.50, 
        'IQ4_XS': 0.75, 'IQ3_XS':0.60, 'IQ2_XS': 0.50, 'IQ1_S':0.40, 
        'BPW': 0.4 
    }
    est_vram_gb_val = 0.0
    model_quant_upper, model_size_final = analysis.get('quant', 'unknown').upper(), analysis.get('size_b', 0)
    
    if isinstance(model_size_final, (int, float)) and model_size_final > 0:
        vram_factor = vram_gb_per_b_param.get(model_quant_upper) 
        if vram_factor is None: 
            vram_factor = next((v for k,v in vram_gb_per_b_param.items() if k in model_quant_upper), None)
        if vram_factor is None: 
             vram_factor = vram_gb_per_b_param['BPW'] if "BPW" in model_quant_upper else 0.9 
        
        est_vram_gb_val = model_size_final * vram_factor
        if analysis['is_moe'] and not analysis['details'].get('size_is_moe_special'):
            est_vram_gb_val *= 1.20 
            
    analysis['estimated_vram_gb_full_gpu'] = round(est_vram_gb_val, 2)
    return analysis

def get_gpu_layers_for_level(model_analysis: dict, attempt_level: int) -> int:
    total_layers = model_analysis.get('num_layers', 32)
    is_moe = model_analysis.get('is_moe', False)
    
    if not isinstance(total_layers, int) or total_layers <= 0:
        total_layers = 32
        
    if is_moe:
        if attempt_level <= -23: return 999
        elif attempt_level <= -19: return int(total_layers * 0.95)
        elif attempt_level <= -16: return int(total_layers * 0.90)
        elif attempt_level <= -13: return int(total_layers * 0.85)
        elif attempt_level <= -10: return int(total_layers * 0.80)
        elif attempt_level <= -7: return int(total_layers * 0.75)
        elif attempt_level <= -4: return int(total_layers * 0.70)
        elif attempt_level <= -1: return int(total_layers * 0.60)
        elif attempt_level <= 2: return int(total_layers * 0.50)
        elif attempt_level <= 5: return int(total_layers * 0.40)
        elif attempt_level <= 8: return int(total_layers * 0.30)
        else: return max(0, int(total_layers * 0.20))
    else: 
        if attempt_level <= -15: return 999
        elif attempt_level <= -12: return int(total_layers * 0.95)
        elif attempt_level <= -10: return int(total_layers * 0.90)
        elif attempt_level <= -8: return int(total_layers * 0.85)
        elif attempt_level <= -6: return int(total_layers * 0.80)
        elif attempt_level <= -4: return int(total_layers * 0.70)
        elif attempt_level <= -2: return int(total_layers * 0.60)
        elif attempt_level <= 0: return int(total_layers * 0.50)
        elif attempt_level <= 2: return int(total_layers * 0.40)
        elif attempt_level <= 4: return int(total_layers * 0.30)
        elif attempt_level <= 7: return int(total_layers * 0.20)
        else: return max(0, int(total_layers * 0.10))

def get_level_from_overridetensors(override_tensor_str: Optional[str], model_analysis: dict) -> int:
    is_moe = model_analysis.get('is_moe', False)
    if not override_tensor_str: return -25 if is_moe else -17
    ot_lower = override_tensor_str.lower()
    if is_moe:
        if "attn\\.(q|k|v|o)\\.weight" in ot_lower and "ffn_down_exps" in ot_lower and "ffn_up_exps" in ot_lower: return 8
        elif "ffn_down_exps" in ot_lower and "ffn_up_exps" in ot_lower and "ffn_gate_exps" in ot_lower: return 1
        elif "ffn_down_exps" in ot_lower and "ffn_up_exps" in ot_lower: return -6
        elif "ffn_down_exps" in ot_lower and not ("ffn_up_exps" in ot_lower or "ffn_gate_exps" in ot_lower): return -10
        elif "blk\\.(0|4|8|12|16|20|24|28)\\.ffn_down_exps" in ot_lower: return -18
        return -15
    else: 
        if "attn\\.(q|k|v|o)\\.weight" in ot_lower and "ffn_down.weight" in ot_lower and "ffn_up.weight" in ot_lower: return 7
        elif "ffn_down.weight" in ot_lower and "ffn_up.weight" in ot_lower and "ffn_gate.weight" in ot_lower: return 0
        elif "ffn_up.weight" in ot_lower and "ffn_down.weight" not in ot_lower and "ffn_gate.weight" not in ot_lower: return -5
        elif "blk\\.(0|4|8|12|16|20|24|28)\\.ffn_up\\.weight" in ot_lower: return -11
        return -8

def get_offload_description(model_analysis: dict, attempt_level: int, current_ot_string: Optional[str]) -> str:
    if current_ot_string == "FAILURE_MAX_ATTEMPTS": return "MAX ATTEMPTS REACHED. No further CPU offload possible."
    is_moe, total_model_layers = model_analysis.get('is_moe', False), model_analysis.get('num_layers', 32)
    if not isinstance(total_model_layers, int) or total_model_layers <=0: total_model_layers = 32
    desc_map = {
        True: {10:"MoE ULTRA MAX CPU", 8:"MoE SUPER MAX CPU", 6:"MoE SUPER CPU++", 4:"MoE SUPER CPU+", 2:"MoE SUPER CPU", 1:"MoE SUPER MAX EXPERT CPU",0:"MoE MAX EXPERT CPU", -2:"MoE CPU++", -4:"MoE CPU+", -6:"MoE CPU/GPU Bal++", -8:"MoE CPU/GPU Bal+", -10:"MoE GPU Focus",-12:"MoE GPU+", -15:"MoE GPU++", -18:"MoE GPU+++", -21:"MoE GPU++++", -25:"MoE MAX GPU"},
        False: {9:"Dense ULTRA MAX CPU", 7:"Dense SUPER MAX CPU", 5:"Dense SUPER CPU", 3:"Dense SUPER CPU-", 0:"Dense MAX FFN CPU", -1:"Dense CPU++",-3:"Dense CPU+", -5:"Dense GPU Focus", -7:"Dense GPU+", -9:"Dense GPU++", -11:"Dense GPU+++", -14:"Dense GPU++++", -17:"Dense MAX GPU"}
    }[is_moe]
    applicable_keys = [k for k in desc_map if k <= attempt_level]
    closest_key = max(applicable_keys) if applicable_keys else min(desc_map.keys())
    base_desc_from_map = f"{'MoE' if is_moe else 'Dense'} Level {attempt_level} (near '{desc_map.get(closest_key, 'Custom')}')" if attempt_level not in desc_map else desc_map.get(attempt_level, f"{'MoE' if is_moe else 'Dense'} Custom (Lvl {attempt_level})")
    gpu_layers_count = get_gpu_layers_for_level(model_analysis, attempt_level)
    ot_str_preview = current_ot_string[:30] + '...' if current_ot_string and len(current_ot_string) > 30 else current_ot_string
    layers_info = f"Cmd uses '--gpulayers 999'. OT string ('{ot_str_preview}') handles CPU offload. (Equiv. layers on GPU if no OT: {gpu_layers_count}/{total_model_layers})" if current_ot_string \
                  else f"Cmd will use '--gpulayers {gpu_layers_count}'. (Based on level {attempt_level}, {total_model_layers} total layers). No OT string."
    return f"{base_desc_from_map}. {layers_info}"

def generate_overridetensors(model_analysis: dict, attempt_level: int) -> Optional[str]:
    moe_s, dense_s = "_exps\\.weight", "\\.weight"; d, u, g = "ffn_down", "ffn_up", "ffn_gate"
    all_l, even_l = "blk\\.\\d+\\.", "blk\\.\\d*[02468]\\."
    num_model_layers = model_analysis.get('num_layers', 32)
    if not isinstance(num_model_layers, int) or num_model_layers <= 0: num_model_layers = 32
    def get_nth_blocks_regex(total_layers, n_groups):
        if n_groups <= 0: return "blk\\.NONE\\." 
        step = max(1, total_layers // n_groups if n_groups > 0 else total_layers)
        selected_blocks = [i for i in range(0, total_layers, step if step > 0 else 1)][:n_groups]
        return f"blk\\.({'|'.join(map(str, selected_blocks))})\\." if selected_blocks else "blk\\.NONE\\."
    l0369, l048 = get_nth_blocks_regex(num_model_layers, num_model_layers//3 if num_model_layers//3 > 0 else 1), get_nth_blocks_regex(num_model_layers, num_model_layers//4 if num_model_layers//4 > 0 else 1)
    eighth_blocks, sixteenth_blocks = get_nth_blocks_regex(num_model_layers, 8), get_nth_blocks_regex(num_model_layers, 16)
    parts = []
    if model_analysis.get('is_moe'):
        if attempt_level>=10: parts.extend([f"{all_l}({d}|{u}|{g}){moe_s}",f"{all_l}({d}|{u}|{g}){dense_s}",f"{all_l}attn\\.(q|k|v|o)\\.weight",f"{all_l}attn_norm\\.weight",f"{all_l}ffn_norm\\.weight"])
        elif attempt_level>=8: parts.extend([f"{all_l}({d}|{u}|{g}){moe_s}",f"{all_l}({d}|{u}|{g}){dense_s}",f"{all_l}attn\\.(q|k|v|o)\\.weight"])
        elif attempt_level>=6: parts.extend([f"{all_l}({d}|{u}|{g}){moe_s}",f"{all_l}({d}|{u}|{g}){dense_s}",f"{even_l}attn\\.(q|k)\\.weight"])
        elif attempt_level>=4: parts.extend([f"{all_l}({d}|{u}|{g}){moe_s}",f"{all_l}({d}|{u}|{g}){dense_s}",f"{l0369}attn\\.q\\.weight"])
        elif attempt_level>=2: parts.extend([f"{all_l}({d}|{u}|{g}){moe_s}",f"{all_l}({d}|{u}|{g}){dense_s}",f"{l048}attn\\.q\\.weight"])
        elif attempt_level==1: parts.extend([f"{all_l}({d}|{u}|{g}){moe_s}",f"{all_l}({d}|{u}|{g}){dense_s}"])
        elif attempt_level==0: parts.extend([f"{all_l}({d}|{u}|{g}){moe_s}",f"{all_l}{d}{dense_s}"])
        elif attempt_level>=-2: parts.append(f"{all_l}({d}|{u}|{g}){moe_s}")
        elif attempt_level>=-4: parts.extend([f"{all_l}({d}|{u}){moe_s}",f"{even_l}{g}{moe_s}"])
        elif attempt_level>=-6: parts.append(f"{all_l}({d}|{u}){moe_s}")
        elif attempt_level>=-8: parts.extend([f"{all_l}{d}{moe_s}",f"{even_l}{u}{moe_s}"])
        elif attempt_level>=-10: parts.append(f"{all_l}{d}{moe_s}")
        elif attempt_level>=-12: parts.append(f"{even_l}{d}{moe_s}")
        elif attempt_level>=-15: parts.append(f"{l048}{d}{moe_s}")
        elif attempt_level>=-18: parts.append(f"{eighth_blocks}{d}{moe_s}")
        elif attempt_level>=-21: parts.append(f"{sixteenth_blocks}{d}{moe_s}")
    else: 
        if attempt_level>=9: parts.extend([f"{all_l}({d}|{u}|{g}){dense_s}",f"{all_l}attn\\.(q|k|v|o)\\.weight",f"{all_l}attn_norm\\.weight",f"{all_l}ffn_norm\\.weight","tok_embeddings\\.weight","output\\.weight"])
        elif attempt_level>=7: parts.extend([f"{all_l}({d}|{u}|{g}){dense_s}",f"{all_l}attn\\.(q|k|v|o)\\.weight"])
        elif attempt_level>=5: parts.extend([f"{all_l}({d}|{u}|{g}){dense_s}",f"{even_l}attn\\.(q|k)\\.weight"])
        elif attempt_level>=3: parts.extend([f"{all_l}({d}|{u}|{g}){dense_s}",f"{l0369}attn\\.q\\.weight"])
        elif attempt_level>=1: parts.extend([f"{all_l}({d}|{u}|{g}){dense_s}",f"{l048}attn\\.q\\.weight"])
        elif attempt_level==0: parts.append(f"{all_l}({d}|{u}|{g}){dense_s}")
        elif attempt_level>=-1: parts.append(f"{all_l}({d}|{u}){dense_s}")
        elif attempt_level>=-3: parts.extend([f"{all_l}{u}{dense_s}",f"{even_l}{d}{dense_s}"])
        elif attempt_level>=-5: parts.append(f"{all_l}{u}{dense_s}")
        elif attempt_level>=-7: parts.append(f"{l0369}{u}{dense_s}")
        elif attempt_level>=-9: parts.append(f"{l048}{u}{dense_s}")
        elif attempt_level>=-11: parts.append(f"{eighth_blocks}{u}{dense_s}")
        elif attempt_level>=-14: parts.append(f"{sixteenth_blocks}{u}{dense_s}")
    valid_parts = [p for p in parts if p and p != "blk\\.NONE\\."]
    if not valid_parts: return None
    return f"({'|'.join(valid_parts)})=CPU" if len(valid_parts) > 1 else f"{valid_parts[0]}=CPU"

def args_list_to_dict(args_list):
    args_dict, i = {}, 0
    while i < len(args_list):
        arg = args_list[i]
        if arg.startswith("--"):
            if i + 1 < len(args_list) and not args_list[i+1].startswith("--"):
                args_dict[arg] = args_list[i+1]; i += 2
            else: args_dict[arg] = True; i += 1
        else: i += 1 
    return args_dict

def args_dict_to_list(args_dict):
    cmd_list_part = []
    if "--model" in args_dict: cmd_list_part.extend(["--model", str(args_dict["--model"])])
    processed_keys = {"--model"}
    defined_keys_ordered = [arg_def["key"] for arg_def in KOBOLDCPP_ARG_DEFINITIONS if arg_def["key"] != "--model"]
    for key in defined_keys_ordered:
        if key in args_dict and key not in processed_keys:
            value = args_dict[key]
            arg_def_type = next((d.get("type_hint", "str") for d in KOBOLDCPP_ARG_DEFINITIONS if d["key"] == key), "str")
            if arg_def_type == "bool_flag": 
                if value is True: cmd_list_part.append(key)
            elif isinstance(value, bool): 
                if value is True: cmd_list_part.append(key)
            elif value is not None: 
                cmd_list_part.extend([key, str(value)])
            processed_keys.add(key)
    for key in sorted([k for k in args_dict if k not in processed_keys]):
        value = args_dict[key]
        if value is True: cmd_list_part.append(key) 
        elif value not in [False, None]: cmd_list_part.extend([key, str(value)])
    return cmd_list_part

def format_command_for_display(cmd_list):
    if not cmd_list: return ""
    exe_part = cmd_list[0]; args_start_index = 1
    if len(cmd_list) > 1 and cmd_list[0].lower() == sys.executable.lower() and cmd_list[1].lower().endswith(".py"):
        exe_part = f'"{cmd_list[0]}" "{cmd_list[1]}"'; args_start_index = 2
    elif " " in cmd_list[0] and not (cmd_list[0].startswith('"') and cmd_list[0].endswith('"')):
        exe_part = f'"{cmd_list[0]}"'
    
    formatted_cmd_parts = [exe_part]
    i = args_start_index
    while i < len(cmd_list):
        item = cmd_list[i]
        if item.startswith("--"):
            current_arg_part = f"\n  {item}" 
            if i + 1 < len(cmd_list) and not cmd_list[i+1].startswith("--"): 
                value_part = cmd_list[i+1]
                if (' ' in value_part or any(c in value_part for c in ['\\', '/', ':'])) and \
                   not value_part.startswith("blk.") and \
                   not value_part.isdigit() and \
                   not (value_part.startswith("(") and value_part.endswith("=CPU")) and \
                   not (value_part.startswith('"') and value_part.endswith('"')):
                    current_arg_part += f' "{value_part}"'
                else:
                    current_arg_part += f" {value_part}"
                i += 1 
            formatted_cmd_parts.append(current_arg_part)
        i += 1 
    return "".join(formatted_cmd_parts)

def get_command_to_run(executable_path, args_list):
    return [sys.executable, executable_path] + args_list if executable_path.lower().endswith(".py") else [executable_path] + args_list

def build_command(model_path, override_tensor_str_from_tuning, model_analysis, session_base_args_dict):
    current_cmd_args_dict = session_base_args_dict.copy()
    current_cmd_args_dict["--model"] = model_path
    model_analysis_dict = model_analysis if isinstance(model_analysis, dict) else {}

    if "--threads" in current_cmd_args_dict and str(current_cmd_args_dict["--threads"]).lower() == "auto":
        if psutil_available:
            try:
                phys_cores = psutil.cpu_count(logical=False)
                current_cmd_args_dict["--threads"] = str(max(1, phys_cores -1 if phys_cores and phys_cores > 1 else 1)) if phys_cores and phys_cores > 0 else str(max(1, (psutil.cpu_count(logical=True) or 2)//2))
            except Exception: current_cmd_args_dict["--threads"] = "4" 
        else: current_cmd_args_dict["--threads"] = "4" 

    if current_cmd_args_dict.get("--nblas") is None or (isinstance(current_cmd_args_dict.get("--nblas"), str) and current_cmd_args_dict.get("--nblas").lower() == 'auto'):
        if "--nblas" in current_cmd_args_dict: del current_cmd_args_dict["--nblas"]
    
    if override_tensor_str_from_tuning and override_tensor_str_from_tuning != "FAILURE_MAX_ATTEMPTS":
        current_cmd_args_dict["--overridetensors"] = override_tensor_str_from_tuning
        current_cmd_args_dict["--gpulayers"] = "999" 
    else: 
        if "--overridetensors" in current_cmd_args_dict: del current_cmd_args_dict["--overridetensors"]
        current_cmd_args_dict["--gpulayers"] = str(session_base_args_dict.get("--gpulayers", DEFAULT_CONFIG_TEMPLATE["default_args"].get("--gpulayers", "auto")))

    gpulayers_val = current_cmd_args_dict.get("--gpulayers")
    if isinstance(gpulayers_val, str) and gpulayers_val.lower() in ['off', '0']:
        if "--gpulayers" in current_cmd_args_dict: del current_cmd_args_dict["--gpulayers"] 
        current_cmd_args_dict["--nogpulayers"] = True 
    elif "--nogpulayers" in current_cmd_args_dict and not (isinstance(gpulayers_val, str) and gpulayers_val.lower() in ['off', '0']):
        del current_cmd_args_dict["--nogpulayers"] 

    quantkv_val = current_cmd_args_dict.get("--quantkv")
    if quantkv_val is None or (isinstance(quantkv_val, str) and quantkv_val.lower() == 'auto'):
        if "--quantkv" in current_cmd_args_dict: del current_cmd_args_dict["--quantkv"] 
        quant_upper, size_b_val = model_analysis_dict.get('quant', 'unknown').upper(), model_analysis_dict.get('size_b', 0)
        if any(q_str in quant_upper for q_str in ['Q5','Q6','Q8','F16','BF16','K_M','K_L','K_XL']) or \
           'XL' in quant_upper or \
           (isinstance(size_b_val, (int,float)) and size_b_val >=30):
            current_cmd_args_dict["--quantkv"] = "1" 
    elif isinstance(quantkv_val, str) and quantkv_val.lower() == 'off':
        if "--quantkv" in current_cmd_args_dict: del current_cmd_args_dict["--quantkv"] 

    bbs_val = current_cmd_args_dict.get("--blasbatchsize")
    if bbs_val is None or (isinstance(bbs_val, str) and bbs_val.lower() == 'auto'):
        if "--blasbatchsize" in current_cmd_args_dict: del current_cmd_args_dict["--blasbatchsize"] 
        size_b_bbs = model_analysis_dict.get('size_b', 0)
        current_cmd_args_dict["--blasbatchsize"] = "128" if model_analysis_dict.get('is_moe',False) else \
                                               ("256" if isinstance(size_b_bbs,(int,float)) and size_b_bbs > 20 else "512")
    elif isinstance(bbs_val, str) and bbs_val.lower() == 'off':
        if "--blasbatchsize" in current_cmd_args_dict: del current_cmd_args_dict["--blasbatchsize"]

    for flag_key in ["--usecublas", "--usehipblas", "--flashattention", "--nommap", "--lowvram"]:
        if flag_key in current_cmd_args_dict and current_cmd_args_dict[flag_key] is False:
            del current_cmd_args_dict[flag_key]
            
    return args_dict_to_list(current_cmd_args_dict)

def kill_process(pid, force=True):
    if not pid: return False, "No PID provided."
    try:
        if sys.platform == "win32":
            args = ["taskkill"]
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW 
            startupinfo.wShowWindow = subprocess.SW_HIDE 
            if force: args.extend(["/F", "/T"])
            args.extend(["/PID", str(pid)])
            process = subprocess.Popen(args, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
            stdout_bytes, stderr_bytes = process.communicate(timeout=5)
            
            if process.returncode == 0: return True, f"Process {pid} terminated."
            
            err_str = stderr_bytes.decode(errors='ignore').lower()
            if process.returncode == 128 or "could not find the process" in err_str or "no running instance" in err_str or "process with pid" in err_str and "not found" in err_str :
                return True, f"Process {pid} not found or already terminated."
            return False, f"Failed to kill process {pid}: RC={process.returncode}, Err={err_str.strip()}"
        else: 
            os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
            return True, f"Signal sent to process {pid}."
    except ProcessLookupError: 
        return True, f"Process {pid} not found (already terminated)."
    except subprocess.TimeoutExpired:
        return False, f"Killing process {pid} timed out."
    except Exception as e:
        return False, f"Error killing process {pid}: {e}"

def kill_processes_by_name(process_name_pattern, cmdline_substr_filter=None):
    if not psutil_available: return False, "psutil not available for process scan."
    killed_any, results_list = False, []
    try:
        for proc_psutil in psutil.process_iter(['pid', 'name', 'exe', 'cmdline']):
            try:
                p_info_psutil = proc_psutil.info
                proc_name_psutil = (p_info_psutil.get('name', '') or "").lower()
                proc_exe_psutil = (p_info_psutil.get('exe', '') or "").lower()
                proc_cmdline_list_psutil = p_info_psutil.get('cmdline', [])
                proc_cmdline_str_psutil = ' '.join(proc_cmdline_list_psutil).lower()
                
                target_name_lower = process_name_pattern.lower()
                match_found = False
                if target_name_lower in proc_name_psutil or \
                   target_name_lower in proc_exe_psutil or \
                   target_name_lower in proc_cmdline_str_psutil:
                    
                    if cmdline_substr_filter: 
                        match_found = any(cmdline_substr_filter.lower() in arg.lower() for arg in proc_cmdline_list_psutil)
                    else: 
                        match_found = True
                
                if match_found:
                    success_kill, message_kill = kill_process(proc_psutil.pid)
                    if success_kill: killed_any = True
                    results_list.append(f"PID {proc_psutil.pid} ('{p_info_psutil.get('name','')}'): {message_kill}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue 
    except Exception as e_scan:
        return False, f"Error scanning processes: {e_scan}"
    
    filter_str = f" (filter: '{cmdline_substr_filter}')" if cmdline_substr_filter else ""
    msg_prefix_scan = f"Sweep for '{process_name_pattern}'{filter_str}: "
    if not results_list:
        return False, msg_prefix_scan + "No matching processes found."
    return (True, msg_prefix_scan + "; ".join(results_list)) if killed_any else (False, msg_prefix_scan + "Matching processes found but none killed (or already terminated).")

def detect_koboldcpp_capabilities(executable_path: str) -> dict: 
    if not executable_path or not (os.path.exists(executable_path) or shutil.which(executable_path)):
        return {"error": "KoboldCpp executable path not provided or invalid.",
                "cuda": False, "rocm": False, "opencl": False, "vulkan": False,
                "flash_attn": False, "auto_quantkv": False, "overridetensors": False,
                "available_args": []}
    resolved_exe_path = shutil.which(executable_path) or executable_path
    try:
        cmd_to_run = [sys.executable, resolved_exe_path, "--help"] if resolved_exe_path.lower().endswith(".py") else [resolved_exe_path, "--help"]
        process = subprocess.run(cmd_to_run, capture_output=True, text=True,
                                 check=False, timeout=10, errors='replace')
        if process.returncode != 0 and not ("--help" in process.stdout or "--port" in process.stdout): 
            error_detail = process.stderr.strip() if process.stderr else process.stdout.strip() or "Unknown error"
            return {"error": f"Failed to run '{os.path.basename(resolved_exe_path)} --help'. RC={process.returncode}. Detail: {error_detail}",
                    "cuda": False, "rocm": False, "opencl": False, "vulkan": False,
                    "flash_attn": False, "auto_quantkv": False, "overridetensors": False,
                    "available_args": []}
        output = process.stdout + "\n" + process.stderr 
        features = {
            "cuda": "--usecublas" in output,
            "rocm": "--usehipblas" in output or "--userocmblas" in output, 
            "opencl": "--useclblast" in output,
            "vulkan": "--usevulkan" in output or "--vulkan" in output,
            "flash_attn": "--flashattention" in output,
            "auto_quantkv": ("--quantkv" in output and "auto" in output.lower()), 
            "overridetensors": "--overridetensors" in output,
            "available_args": []
        }
        arg_pattern = r"(?<!\w)(-{1,2}[\w-]+)"
        try:
            features["available_args"] = list(set(re.findall(arg_pattern, output)))
        except Exception: 
            features["available_args"] = ["Error parsing args"]
        return features
    except subprocess.TimeoutExpired:
        return {"error": f"Timeout running '{os.path.basename(resolved_exe_path)} --help'.", "cuda": False, "rocm": False, "opencl": False, "vulkan": False, "flash_attn": False, "auto_quantkv": False, "overridetensors": False, "available_args": []}
    except FileNotFoundError:
        return {"error": f"Executable '{resolved_exe_path}' not found.", "cuda": False, "rocm": False, "opencl": False, "vulkan": False, "flash_attn": False, "auto_quantkv": False, "overridetensors": False, "available_args": []}
    except Exception as e:
        return {"error": f"Error detecting KCPP capabilities: {type(e).__name__}: {e}", "cuda": False, "rocm": False, "opencl": False, "vulkan": False, "flash_attn": False, "auto_quantkv": False, "overridetensors": False, "available_args": []}

def launch_process(cmd, capture_output=True, new_console=False, use_text_mode=True):
    try:
        kwargs = {}
        if capture_output:
            kwargs.update({'stdout': subprocess.PIPE, 
                           'stderr': subprocess.STDOUT }) 
            if use_text_mode: 
                kwargs.update({'text': True, 'universal_newlines': True, 'bufsize': 1, 'errors': 'replace'})
        
        if sys.platform == "win32":
            creation_flags = 0
            if new_console:
                creation_flags = subprocess.CREATE_NEW_CONSOLE
            elif capture_output: 
                creation_flags = subprocess.CREATE_NO_WINDOW
            if creation_flags != 0:
                kwargs['creationflags'] = creation_flags
        
        process = subprocess.Popen(cmd, **kwargs)
        return process, None
    except FileNotFoundError:
        return None, f"Executable '{cmd[0]}' not found."
    except PermissionError:
        return None, f"Permission denied for '{cmd[0]}'."
    except Exception as e:
        return None, f"Launch error: {type(e).__name__}: {e}"


def initialize_launcher():
    config, config_loaded_ok, config_message = load_config()
    db_success, db_message = init_db(config.get("db_file"))

    if psutil_available and config["default_args"].get("--threads") == "auto":
        try:
            phys_cores = psutil.cpu_count(logical=False)
            config["default_args"]["--threads"] = str(max(1, phys_cores - 1 if phys_cores and phys_cores > 1 else 1)) if phys_cores and phys_cores > 0 else str(max(1, (psutil.cpu_count(logical=True) or 2) // 2))
        except Exception: config["default_args"]["--threads"] = "4" 
    elif config["default_args"].get("--threads") == "auto": 
        config["default_args"]["--threads"] = "4"

    _, _, _, gpu_details = get_available_vram_mb(current_config=config) 
    
    kcpp_caps = detect_koboldcpp_capabilities(config.get("koboldcpp_executable",""))

    config["default_args"]["--usecublas"] = False
    config["default_args"]["--usehipblas"] = False
    
    if gpu_details and gpu_details.get("success"): 
        if gpu_details.get("type") == "NVIDIA" and \
           config.get("gpu_detection",{}).get("nvidia",True) and \
           kcpp_caps.get("cuda"):
            config["default_args"]["--usecublas"] = True
        elif gpu_details.get("type") == "AMD" and \
             config.get("gpu_detection",{}).get("amd",True) and \
             kcpp_caps.get("rocm"): 
            config["default_args"]["--usehipblas"] = True
    
    return {"initialized": config_loaded_ok and db_success, "config": config,
            "config_loaded": config_loaded_ok, "config_message": config_message,
            "db_success": db_success, "db_message": db_message,
            "system_info": get_system_info(), "gpu_info": gpu_details, 
            "koboldcpp_capabilities": kcpp_caps}

def _cleanup_nvml():
    if pynvml_available:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError: 
            pass
        except Exception: 
            pass

if __name__ == "__main__":
    print("TensorTune Core Library Test - v" + DEFAULT_CONFIG_TEMPLATE["launcher_core_version"])
    init_results = initialize_launcher() 
    
    print(f"\nConfig File: {CONFIG_FILE}")
    print(f"Init OK: {init_results['initialized']}")
    print(f"Config Loaded: {init_results['config_loaded']} ({init_results['config_message']})")
    print(f"  DB Path: {init_results['config']['db_file']}")
    print(f"DB Status: {init_results['db_success']} ({init_results['db_message']})")

    print("\nSystem Info:")
    for k,v in init_results['system_info'].items(): print(f"  {k}: {v}")
    
    print("\nGPU Info (from initialize_launcher):")
    gpu = init_results['gpu_info']
    if gpu and isinstance(gpu, dict): 
        if gpu.get("success"):
            print(f"  Type: {gpu.get('type')}, Name: {gpu.get('name')}")
            print(f"  Actual HW Total: {gpu.get('total_mb'):.0f}MB, Actual HW Free: {gpu.get('free_mb'):.0f}MB")
            print(f"  Budgeted Total: {gpu.get('total_mb_budgeted'):.0f}MB, Budgeted Free: {gpu.get('free_mb_budgeted'):.0f}MB")
            print(f"  Override Active: {gpu.get('override_active')}")
            print(f"  Message: {gpu.get('message')}")
        else:
            print(f"  Detection Attempt Message: {gpu.get('message', 'GPU detection failed or no VRAM info.')}")
    else:
        print("  No GPU dictionary returned by initialization.")

    print("\nKCPP Capabilities:")
    caps = init_results.get("koboldcpp_capabilities", {})
    if "error" in caps: print(f"  Error: {caps['error']}")
    else: print(f"  CUDA: {caps.get('cuda')}, ROCm: {caps.get('rocm')}, FlashAttn: {caps.get('flash_attn')}")

    print("\nListing NVIDIA GPUs:")
    for g in list_nvidia_gpus(): print(f"  ID {g['id']}: {g['name']}")
    print("Listing AMD GPUs (Windows):")
    for g in list_amd_gpus_windows(): print(f"  ID {g['id']}: {g['name']} (Source: {g.get('source','N/A')})")
    print("Listing AMD GPUs (Linux):")
    for g in list_amd_gpus_linux(): print(f"  ID {g['id']}: {g['name']}")


    _cleanup_nvml()

import atexit
atexit.register(_cleanup_nvml)