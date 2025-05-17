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
appdirs_load_error_reason = None
try:
    import appdirs
    appdirs_available = True
except ImportError:
    appdirs_load_error_reason = "Appdirs library not found. Using fallback paths."
except Exception as e_appdirs:
    appdirs_load_error_reason = f"Appdirs library failed to load: {e_appdirs}. Using fallback paths."


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
CORE_VERSION = "1.0.1-TT" # Core version updated

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
    "launcher_core_version": CORE_VERSION, # Ensure CORE_VERSION is defined, e.g., "1.1.0-TT"
    "suppress_optional_lib_warnings": False # <-- NEW FLAG
}

KOBOLDCPP_ARG_DEFINITIONS = [
    {"key": "--model", "name": "Model Path", "help": "Path to the GGUF model file.", "type_hint": "path", "category": "Core"},
    {"key": "--port", "name": "Port", "help": "Network port for the KoboldCpp API (e.g., 5000).", "type_hint": "int", "category": "Network"},
    {"key": "--threads", "name": "CPU Threads", "help": "Number of CPU threads to use. 'auto' for detection.", "type_hint": "str_auto_num", "category": "Performance"},
    {"key": "--nblas", "name": "BLAS Threads", "help": "Number of CPU threads for BLAS. 'auto' for detection.", "type_hint": "str_auto_num", "category": "Performance"},
    {"key": "--contextsize", "name": "Context Size", "help": "Max context window size in tokens (e.g., 4096, 16384).", "type_hint": "int", "category": "Model Params"},
    {"key": "--promptlimit", "name": "Prompt Limit", "help": "Maximum length of prompt that can be processed (<= contextsize).", "type_hint": "int", "category": "Model Params"},
    {"key": "--defaultgenamt", "name": "Default Generation Amount", "help": "Default number of tokens to generate (e.g., 1024, 2048).", "type_hint": "int", "category": "Model Params"},
    {"key": "--gpulayers", "name": "GPU Layers", "help": "Number of model layers to offload to GPU. 'auto', 'off', or number (e.g., 35, 999 for max).", "type_hint": "str_auto_num", "category": "GPU Offload"},
    {"key": "--nogpulayers", "name": "No GPU Layers", "help": "Explicitly disable GPU layer offloading (alternative to --gpulayers 0/off).", "type_hint": "bool_flag", "category": "GPU Offload"},
    {"key": "--usecublas", "name": "Use CUBLAS (NVIDIA)", "help": "Enable CUBLAS for NVIDIA GPU acceleration.", "type_hint": "bool", "category": "GPU Backend"},
    {"key": "--usehipblas", "name": "Use hipBLAS (AMD ROCm)", "help": "Enable hipBLAS for AMD GPU acceleration (ROCm).", "type_hint": "bool", "category": "GPU Backend"},
    {"key": "--flashattention", "name": "Flash Attention", "help": "Enable FlashAttention optimization (if supported by model and GPU).", "type_hint": "bool", "category": "GPU Optimizations"},
    {"key": "--nommap", "name": "No Memory Map", "help": "Disable memory mapping of the model file.", "type_hint": "bool", "category": "Memory"},
    {"key": "--lowvram", "name": "Low VRAM Mode", "help": "Enable optimizations for systems with low VRAM.", "type_hint": "bool", "category": "Memory"},
    {"key": "--quantkv", "name": "Quantize K/V Cache", "help": "Quantization for K/V cache. 'auto', 'off', or number (0=F32, 1=Q8_0, etc.).", "type_hint": "str_auto_num", "category": "GPU Optimizations"},
    {"key": "--blasbatchsize", "name": "BLAS Batch Size", "help": "Batch size for BLAS operations. 'auto', 'off', or number (e.g., 128, 512).", "type_hint": "str_auto_num", "category": "Performance"},
    {"key": "--overridetensors", "name": "Override Tensors", "help": "Advanced: Specify tensor offload patterns to CPU (regex).", "type_hint": "str_regex", "category": "GPU Offload (Advanced)"},
]


# GPU / System Library Availability & Load Error Tracking
pynvml_available = False
pynvml_load_error_reason = None
psutil_available = False
psutil_load_error_reason = None
pyadlx_available = False
pyadlx_load_error_reason = None
wmi_available = False
wmi_load_error_reason = None
pyze_available = False
pyze_load_error_reason = None
metal_available = False
metal_load_error_reason = None

try:
    import pynvml
    try:
        pynvml.nvmlInit()
        pynvml_available = True
    except pynvml.NVMLError as e_init:
        pynvml_load_error_reason = f"PyNVML (NVIDIA) initialized failed: {e_init}. Ensure NVIDIA drivers are correctly installed and up-to-date."
    except Exception as e_generic_init:
        pynvml_load_error_reason = f"PyNVML (NVIDIA) generic initialization error: {e_generic_init}."
except ImportError:
    pynvml_load_error_reason = "PyNVML (NVIDIA) library not found. Install with: pip install pynvml"

try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_load_error_reason = "Psutil library not found. Install with: pip install psutil"
except Exception as e_psutil:
    psutil_load_error_reason = f"Psutil library failed to load: {e_psutil}"

if sys.platform == "win32":
    try:
        import pyadlx # Attempt to import the manually built PyADLX bindings
        if hasattr(pyadlx, 'ADLXHelper'):
            try:
                # This tests if the ADLX SDK is properly installed and accessible
                # via the PyADLX bindings.
                helper = pyadlx.ADLXHelper()
                pyadlx_available = True
                del helper
            except Exception as e_adlx_init: # Catches ADLXNotFoundException or other ADLX init issues
                pyadlx_available = False
                pyadlx_load_error_reason = (
                    f"PyADLX (AMD) bindings found, but ADLX system helper failed to initialize: {e_adlx_init}. "
                    "This usually means the AMD Adrenalin software (which includes ADLX SDK components) is not fully installed or there's a driver/SDK mismatch. "
                    "TensorTune will use WMI for AMD GPU info instead. "
                    "For advanced users wanting full PyADLX support, ensure Adrenalin is installed and see 'PYADLX_SETUP_GUIDE.md' (distributed with TensorTune or in its documentation) for PyADLX setup details."
                )
        else:
            pyadlx_available = False
            pyadlx_load_error_reason = (
                "PyADLX (AMD) Python bindings found, but 'ADLXHelper' is missing. "
                "This suggests an incomplete or corrupted PyADLX manual build. "
                "TensorTune will use WMI for AMD GPU info. "
                "See 'PYADLX_SETUP_GUIDE.md' (distributed with TensorTune or in its documentation) for PyADLX build instructions."
            )
    except ImportError:
        pyadlx_available = False
        pyadlx_load_error_reason = (
            "PyADLX (AMD) Python library not found. This optional library provides detailed AMD GPU information on Windows. "
            "It requires manual setup: AMD Adrenalin software must be installed, and then Python bindings for the ADLX SDK must be manually built. "
            "TensorTune will use WMI for AMD GPU info instead. "
            "For advanced users, see 'PYADLX_SETUP_GUIDE.md' (distributed with TensorTune or in its documentation) for instructions."
        )
    except Exception as e_adlx_load: # Catch any other unexpected error during pyadlx import/initialization
        pyadlx_available = False
        pyadlx_load_error_reason = (
            f"PyADLX (AMD) library failed to load unexpectedly: {type(e_adlx_load).__name__}: {e_adlx_load}. "
            "TensorTune will use WMI for AMD GPU info. "
            "Ensure AMD Adrenalin software is installed. For advanced PyADLX setup, see 'PYADLX_SETUP_GUIDE.md' (distributed with TensorTune or in its documentation)."
        )

    # WMI (as before)
    try:
        import wmi
        try:
            wmi.WMI() # Test basic WMI functionality
            wmi_available = True
        except Exception as e_wmi_init:
            wmi_load_error_reason = (f"WMI (Windows) initialization failed: {e_wmi_init}. "
                                     "WMI may be corrupted on your system. See 'WMI_SETUP_GUIDE.md' for troubleshooting tips.")
    except ImportError:
        wmi_load_error_reason = ("WMI (Windows) Python library not found. "
                                 "This library is used as a fallback for system/GPU info. Install with: pip install WMI. "
                                 "See 'WMI_SETUP_GUIDE.md' for details.")
    except Exception as e_wmi_load:
        wmi_load_error_reason = (f"WMI (Windows) Python library failed to load: {e_wmi_load}. "
                                 "See 'WMI_SETUP_GUIDE.md' for troubleshooting.")



try:
    import pyze.api as pyze_api
    if hasattr(pyze_api, 'zeInit') and hasattr(pyze_api, 'zeDriverGet'):
        try:
            if pyze_api.zeInit(0) == pyze_api.ZE_RESULT_SUCCESS:
                pyze_available = True
            else:
                pyze_load_error_reason = ("PyZE (Intel Level Zero) zeInit call failed. "
                                          "Ensure Intel drivers and Level Zero runtime are correctly installed. "
                                          "See 'PYZE_SETUP_GUIDE.md' for details.")
        except Exception as e_pyze_init:
            pyze_load_error_reason = (f"PyZE (Intel Level Zero) initialization error: {e_pyze_init}. "
                                      "See 'PYZE_SETUP_GUIDE.md' for troubleshooting.")
    else:
        pyze_load_error_reason = ("PyZE (Intel Level Zero) found, but essential functions are missing. "
                                  "This may indicate a corrupt or old install. Try reinstalling 'pyze-l0'. "
                                  "See 'PYZE_SETUP_GUIDE.md' for details.")
except ImportError:
    pyze_load_error_reason = ("PyZE (Intel Level Zero) library not found. "
                              "This optional library is required for detailed Intel GPU info. Install with: pip install pyze-l0. "
                              "See 'PYZE_SETUP_GUIDE.md' for more information.")
except Exception as e_pyze_load:
    pyze_load_error_reason = (f"PyZE (Intel Level Zero) library failed to load: {e_pyze_load}. "
                              "See 'PYZE_SETUP_GUIDE.md' for troubleshooting.")


if sys.platform == "darwin":
    try:
        import metal
        try:
            if metal.MTLCopyAllDevices(): # Test basic Metal functionality
                metal_available = True
            else:
                # This case might mean no Metal devices, which is different from library load/init error
                metal_load_error_reason = "Metal (Apple) MTLCopyAllDevices returned no devices."
        except Exception as e_metal_init:
            metal_load_error_reason = f"Metal (Apple) API call failed: {e_metal_init}. Ensure macOS and drivers are up-to-date."
    except ImportError:
        metal_load_error_reason = "Metal (Apple) library bindings not found. This is unexpected on modern macOS."
    except Exception as e_metal_load:
        metal_load_error_reason = f"Metal (Apple) library bindings failed to load: {e_metal_load}."

# KCPP Capabilities Cache
_kcpp_capabilities_cache: Dict[str, Dict[str, Any]] = {}


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
        # Version comparison using tuples of integers
        # Remove -TT suffix for comparison if present
        current_version_tuple = tuple(map(int, current_template_version_str.replace("-TT","").split('.')))
    except ValueError:
        current_version_tuple = (0,0,0) # Fallback if parsing fails

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
                loaded_version_tuple = tuple(map(int, loaded_version_str.replace("-TT","").split('.')))
            except ValueError:
                loaded_version_tuple = (0,0,0)

            if loaded_version_tuple < current_version_tuple:
                print(f"INFO: Launcher configuration version mismatch (Loaded: {loaded_version_str}, Current Template: {current_template_version_str}). Migrating and updating version.")
                user_config_loaded["launcher_core_version"] = current_template_version_str # Update to new version string
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
        
        current_vram_for_query = 0.0
        if isinstance(current_available_dedicated_vram_mb, (int, float)):
            current_vram_for_query = float(current_available_dedicated_vram_mb)
        
        target_vram_for_sort = current_vram_for_query if current_vram_for_query > 0 else 8192.0 # Default if current is 0
        
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
              CASE WHEN h.launch_outcome LIKE '%_USER_MARKED_AS_BEST_GUI' THEN -2 -- Highest priority for GUI marked best
                   WHEN h.launch_outcome LIKE '%_USER_MARKED_AS_BEST_CLI' THEN -1 -- Next for CLI marked best
                   WHEN h.launch_outcome LIKE 'SUCCESS_USER_CONFIRMED%' THEN 0
                   WHEN h.launch_outcome LIKE '%_USER_SAVED_GOOD_GPU_%' THEN 1 -- Catches _GUI and _CLI variants
                   ELSE 10 END ASC,
              CASE WHEN h.launch_outcome LIKE 'SUCCESS_LOAD_VRAM_OK%' AND
                        h.approx_vram_used_kcpp_mb IS NOT NULL AND
                        (h.approx_vram_used_kcpp_mb + ?) < ? THEN 2 -- Historical actual used + buffer < current actual available
                   ELSE 10 END ASC,
              CASE WHEN h.launch_outcome LIKE 'SUCCESS%' AND
                        h.approx_vram_used_kcpp_mb IS NOT NULL AND
                        (h.approx_vram_used_kcpp_mb + ?) < ? THEN 3
                   ELSE 10 END ASC,
              CASE WHEN h.launch_outcome LIKE 'SUCCESS_USER_DIRECT_LAUNCH%' THEN 4
                   WHEN h.launch_outcome LIKE '%_USER_ACCEPTED_TUNED_%' THEN 5 -- Catches _GUI and _CLI variants
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
        # Parameters for the ORDER BY CASE statements that use current_vram_for_query and vram_safety_buffer_mb
        base_params.extend([
            vram_safety_buffer_mb, current_vram_for_query, # For CASE index 2
            vram_safety_buffer_mb, current_vram_for_query, # For CASE index 3
            target_vram_for_sort, target_vram_for_sort  # For ABS diff and final sort
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
                # This might happen if args_json is corrupt in DB for some reason
                print(f"Warning: Could not parse JSON args from historical DB entry for model {current_model_analysis['filepath']}.")
                return None
        return None
    except sqlite3.Error as e_sql:
        print(f"Database error in find_best_historical_config: {e_sql}")
        return None
    except Exception as e_general: 
        print(f"Unexpected error in find_best_historical_config: {type(e_general).__name__}: {e_general}")
        import traceback
        traceback.print_exc() # Log full traceback for unexpected errors
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
    elif psutil_load_error_reason:
        print(f"System Info Warning: {psutil_load_error_reason}")
    return info

# --- GPU Listing and Info Functions ---

def _read_sysfs_long(path: str) -> Optional[int]:
    try:
        with open(path, 'r') as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError, PermissionError, OSError):
        return None

def _get_gpu_name_from_pci_ids_linux(vendor_id: str, device_id: str) -> str:
    # This is a very basic mapper, can be expanded
    vendor_map = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel"}
    name = f"{vendor_map.get(vendor_id, 'Unknown Vendor')} Device ({vendor_id}/{device_id})"
    # Could try to parse `lspci -nnk` here for a better name, but adds complexity and lspci dependency
    return name

def _list_gpus_linux_sysfs(vendor_id_str: str, vendor_name_for_display: str) -> List[Dict[str, Any]]:
    gpus = []
    gpu_idx_counter = 0
    drm_base_path = Path("/sys/class/drm/")
    if not drm_base_path.exists():
        return gpus

    for entry in drm_base_path.iterdir():
        if entry.is_dir() and entry.name.startswith("card"):
            try:
                uevent_path = entry / "device/uevent"
                modalias_path = entry / "device/modalias"
                pci_id_from_uevent = None
                pci_id_from_modalias = None

                if uevent_path.exists():
                    with open(uevent_path, 'r') as f_uevent:
                        for line in f_uevent:
                            if line.startswith("PCI_ID="): # Example: PCI_ID=10DE:2484
                                pci_id_from_uevent = line.strip().split('=')[1]
                                break
                            if line.startswith("DRIVER=") and "nvidia" in line: # Helpful for NVIDIA
                                pci_id_from_uevent = "nvidia_driver_hint" # Will use vendor_id_str
                
                if modalias_path.exists() and not pci_id_from_uevent: # Fallback to modalias
                    # modalias: pci:v000010DEd00002484sv000010DEsd00001467bc03sc00i00
                    with open(modalias_path, 'r') as f_modalias:
                        modalias_content = f_modalias.read().strip()
                        match = re.search(r"pci:v0000([0-9A-Fa-f]{4})d0000([0-9A-Fa-f]{4})", modalias_content)
                        if match:
                            pci_id_from_modalias = f"{match.group(1).upper()}:{match.group(2).upper()}"

                current_vendor = None
                current_device = None

                if pci_id_from_uevent and pci_id_from_uevent != "nvidia_driver_hint" and ":" in pci_id_from_uevent:
                    current_vendor, current_device = pci_id_from_uevent.split(":")
                elif pci_id_from_modalias and ":" in pci_id_from_modalias:
                    current_vendor, current_device = pci_id_from_modalias.split(":")
                
                # For NVIDIA, uevent might not have PCI_ID but DRIVER=nvidia.
                # For Intel/AMD, device/vendor and device/device files are more reliable if uevent/modalias fail.
                if not current_vendor:
                    vendor_file = entry / "device/vendor"
                    device_file = entry / "device/device"
                    if vendor_file.exists() and device_file.exists():
                        read_vendor = _read_sysfs_long(str(vendor_file))
                        read_device = _read_sysfs_long(str(device_file))
                        if read_vendor is not None: current_vendor = f"{read_vendor:#0{6}x}"[2:].upper() # Format as hex string
                        if read_device is not None: current_device = f"{read_device:#0{6}x}"[2:].upper()


                if current_vendor and current_vendor.lower() == vendor_id_str.lower().replace("0x",""):
                    gpu_name = f"{vendor_name_for_display} Graphics Device (sysfs: {entry.name})"
                    if current_device:
                        gpu_name = _get_gpu_name_from_pci_ids_linux(f"0x{current_vendor}", f"0x{current_device}") + f" ({entry.name})"
                    
                    gpus.append({"id": gpu_idx_counter, "name": gpu_name, "sysfs_card": entry.name})
                    gpu_idx_counter += 1
            except Exception: # Ignore errors for individual card parsing
                continue
    return gpus

def _get_gpu_info_linux_sysfs(vendor_id_str: str, vendor_name_for_display: str, device_index: int) -> Optional[Dict[str, Any]]:
    listed_gpus = _list_gpus_linux_sysfs(vendor_id_str, vendor_name_for_display)
    if not listed_gpus or device_index >= len(listed_gpus) or device_index < 0:
        return {"success": False, "type": f"{vendor_name_for_display.upper()}_SYSFS_NO_DEVICE" if not listed_gpus else f"{vendor_name_for_display.upper()}_SYSFS_BAD_INDEX",
                "message": f"SysFS: {'No ' + vendor_name_for_display + ' GPUs found' if not listed_gpus else f'Index {device_index} out of range ({len(listed_gpus)} found)'}."}

    target_gpu_sysfs = listed_gpus[device_index]
    gpu_name = target_gpu_sysfs["name"]
    sysfs_card_path = Path("/sys/class/drm/") / target_gpu_sysfs["sysfs_card"]

    total_mb = 0
    vram_total_path = sysfs_card_path / "device/mem_info_vram_total"
    if vram_total_path.exists():
        vram_bytes = _read_sysfs_long(str(vram_total_path))
        if vram_bytes is not None:
            total_mb = vram_bytes / (1024**2)

    # Free/Used VRAM is not reliably available via sysfs for most cards
    return {"success": True, "type": f"{vendor_name_for_display.upper()}_SYSFS", "name": f"{gpu_name} (ID {device_index})",
            "free_mb": 0, "total_mb": round(total_mb, 1), "used_percent": 0, # Reporting 0 free/used as unknown
            "message": f"{vendor_name_for_display} {gpu_name} (ID {device_index}): Total {total_mb:.0f}MB (SysFS fallback, usage info N/A)"}


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
    amd_gpu_idx_counter = 0 # Consistent indexing across sources

    if pyadlx_available:
        try:
            helper = pyadlx.ADLXHelper()
            adlx_gpus_list = helper.get_gpus()
            for gpu_obj in adlx_gpus_list:
                if hasattr(gpu_obj, 'name') and ("amd" in gpu_obj.name.lower() or "radeon" in gpu_obj.name.lower()):
                    gpus.append({"id": amd_gpu_idx_counter, "name": gpu_obj.name, "source": "pyadlx"})
                    amd_gpu_idx_counter += 1
        except Exception: # pyadlx.ADLXNotFoundException or others
            pass # Error reason already stored in pyadlx_load_error_reason

    if wmi_available:
        try:
            c = wmi.WMI()
            for gpu_wmi_item in c.Win32_VideoController():
                if hasattr(gpu_wmi_item, 'Name') and ("amd" in gpu_wmi_item.Name.lower() or "radeon" in gpu_wmi_item.Name.lower()):
                    # Add only if not already listed by pyADLX (prefer pyADLX name)
                    if not any(g['name'] == gpu_wmi_item.Name and g.get('source') == 'pyadlx' for g in gpus):
                        # Check if a GPU with a similar name (but different source) is already there to avoid re-adding under new ID
                        existing_wmi_match = next((g for g in gpus if g['name'] == gpu_wmi_item.Name and g.get('source') == 'wmi'), None)
                        if not existing_wmi_match:
                            gpus.append({"id": amd_gpu_idx_counter, "name": gpu_wmi_item.Name, "source": "wmi"})
                            amd_gpu_idx_counter += 1
        except Exception:
            pass # Error reason already stored in wmi_load_error_reason
    return gpus

def list_amd_gpus_linux() -> List[Dict[str, Any]]:
    gpus = []
    try:
        # Try rocm-smi first
        if subprocess.run(["which", "rocm-smi"], capture_output=True, text=True, check=False, timeout=2).returncode == 0:
            json_proc = subprocess.run(["rocm-smi", "--showproductname", "--json"], capture_output=True, text=True, check=True, timeout=5, errors='replace')
            data = json.loads(json_proc.stdout)
            gpu_idx_counter = 0
            sorted_card_keys = sorted([key for key in data.keys() if key.startswith("card")], key=lambda x: int(x.replace("card","")))

            for card_key in sorted_card_keys:
                gpu_name = data[card_key].get("Card SKU", data[card_key].get("Card series", f"AMD GPU {gpu_idx_counter} (rocm-smi)"))
                gpus.append({"id": gpu_idx_counter, "name": gpu_name, "source": "rocm-smi"})
                gpu_idx_counter +=1
            if gpus: return gpus # Return if rocm-smi was successful
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass # Will fall through to sysfs if rocm-smi fails
    except Exception:
        pass

    # Fallback to SysFS if rocm-smi fails or yields no GPUs
    try:
        sysfs_gpus = _list_gpus_linux_sysfs("1002", "AMD") # AMD PCI Vendor ID is 1002
        for i, gpu_info in enumerate(sysfs_gpus):
            # Ensure unique IDs if rocm-smi partially succeeded or to avoid overlap if logic changes
            existing_ids = {g['id'] for g in gpus}
            new_id = len(gpus)
            while new_id in existing_ids: new_id +=1
            gpus.append({"id": new_id, "name": gpu_info["name"], "source": "sysfs"})
    except Exception:
        pass # SysFS fallback also failed
    return gpus

def list_intel_gpus() -> List[Dict[str, Any]]:
    gpus = []
    idx_counter = 0
    if pyze_available:
        try:
            # zeInit was already called globally, but check availability
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
                                gpus.append({"id": idx_counter, "name": gpu_name, "source": "pyze"})
                                idx_counter += 1
                        pyze_api.delete_ze_device_handle_t_array(devices_array)
                pyze_api.delete_ze_driver_handle_t_array(drivers_array)
            if gpus: return gpus # Return if pyze was successful
        except Exception:
            pass # Will fall through to sysfs if on Linux

    if sys.platform == "linux": # Fallback to SysFS if pyze fails or yields no GPUs (and on Linux)
        try:
            sysfs_gpus = _list_gpus_linux_sysfs("8086", "Intel") # Intel PCI Vendor ID is 8086
            for i, gpu_info in enumerate(sysfs_gpus):
                existing_ids = {g['id'] for g in gpus}
                new_id = len(gpus)
                while new_id in existing_ids: new_id +=1
                gpus.append({"id": new_id, "name": gpu_info["name"], "source": "sysfs"})
        except Exception:
            pass # SysFS fallback also failed
    return gpus


def list_apple_gpus() -> List[Dict[str, Any]]:
    if not (sys.platform == "darwin" and metal_available): return []
    gpus = []
    try:
        devices = metal.MTLCopyAllDevices()
        for i, device in enumerate(devices):
            name_suffix = " (Low Power)" if hasattr(device, 'isLowPower') and device.isLowPower() else ""
            gpu_name = "Unknown Apple GPU"
            if hasattr(device, 'name') and callable(device.name):
                raw_name = device.name()
                gpu_name = str(raw_name) if raw_name else "Unknown Apple GPU" # Ensure name is a string
            gpus.append({"id": i, "name": gpu_name + name_suffix, "source": "metal"})
    except Exception:
        pass
    return gpus

def get_gpu_info_nvidia(device_index: int = 0) -> Optional[Dict[str, Any]]:
    if not pynvml_available:
        return {"success": False, "type": "NVIDIA_LIB_UNAVAILABLE", "message": pynvml_load_error_reason or "PyNVML (NVIDIA) library not available."}
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
        # Differentiate critical errors from runtime query errors
        if e_nvml.value in [pynvml.NVML_ERROR_FUNCTION_NOT_FOUND, pynvml.NVML_ERROR_LIBRARY_NOT_FOUND, pynvml.NVML_ERROR_DRIVER_NOT_LOADED, pynvml.NVML_ERROR_UNINITIALIZED]:
            return {"success": False, "type": "NVIDIA_DRIVER_ISSUE", "message": f"NVML Critical Error (ID {device_index}): {e_nvml}. Check NVIDIA drivers."}
        return {"success": False, "type": "NVIDIA_RUNTIME_ERROR", "message": f"NVML Runtime Error (ID {device_index}): {e_nvml}"}
    except Exception as e_generic:
        return {"success": False, "type": "NVIDIA_GENERIC_ERROR", "message": f"NVIDIA Generic Error (ID {device_index}): {e_generic}"}

def _format_amd_win_message(method: str, status: str, detail: str = "") -> str:
    return f"AMD Win ({method}): {status}. {detail}".strip()

def get_gpu_info_amd(device_index: int = 0) -> Optional[Dict[str, Any]]:
    if sys.platform == "linux":
        rocm_smi_path = shutil.which("rocm-smi")
        if not rocm_smi_path:
            sysfs_fallback_result = _get_gpu_info_linux_sysfs("1002", "AMD", device_index)
            if sysfs_fallback_result and sysfs_fallback_result["success"]:
                return sysfs_fallback_result
            return {"success": False, "type": "AMD_ROCM_SMI_NOT_FOUND", "message": "rocm-smi utility not found. Please ensure ROCm is installed and rocm-smi is in your PATH. SysFS fallback also failed or found no matching GPU."}

        try:
            target_card_key_for_query = f"card{device_index}"
            # Get all cards' info at once to map device_index correctly if rocm-smi's own indexing is sparse
            json_proc_all = subprocess.run([rocm_smi_path, "--showmeminfo", "vram", "--showproductname", "--json"], capture_output=True, text=True, check=True, timeout=5, errors='replace')
            data_all = json.loads(json_proc_all.stdout)

            # Sort keys like "card0", "card1", ... to handle sparse indexing if rocm-smi provides it
            sorted_card_keys_from_data = sorted([k for k in data_all if k.startswith("card")], key=lambda x: int(x.replace("card","")))

            actual_card_key_to_use = None
            if 0 <= device_index < len(sorted_card_keys_from_data):
                actual_card_key_to_use = sorted_card_keys_from_data[device_index]
            else: # Index out of bounds for detected cards
                 sysfs_fallback_result = _get_gpu_info_linux_sysfs("1002", "AMD", device_index)
                 if sysfs_fallback_result and sysfs_fallback_result["success"]: return sysfs_fallback_result
                 return {"success": False, "type": "AMD_ROCM_SMI_BAD_INDEX", "message": f"rocm-smi: Card index {device_index} out of range for {len(sorted_card_keys_from_data)} AMD GPU(s) found by rocm-smi. SysFS fallback also failed or no match."}


            gpu_data = data_all[actual_card_key_to_use]
            total_mem_bytes_str, used_mem_bytes_str = gpu_data.get("VRAM Total Memory (B)"), gpu_data.get("VRAM Used Memory (B)")
            if not total_mem_bytes_str or not used_mem_bytes_str:
                 sysfs_fallback_result = _get_gpu_info_linux_sysfs("1002", "AMD", device_index)
                 if sysfs_fallback_result and sysfs_fallback_result["success"]: return sysfs_fallback_result
                 return {"success": False, "type": "AMD_ROCM_SMI_PARSE_ERROR", "message": f"VRAM info missing for {actual_card_key_to_use} in rocm-smi JSON. SysFS fallback also failed."}


            total_mb_val, used_mb_val = int(total_mem_bytes_str) / (1024**2), int(used_mem_bytes_str) / (1024**2)
            free_mb_val = total_mb_val - used_mb_val
            gpu_name_val = gpu_data.get("Card SKU", gpu_data.get("Card series", f"AMD GPU {actual_card_key_to_use}"))
            used_percent_val = (used_mb_val / total_mb_val * 100) if total_mb_val > 0 else 0.0
            return {"success": True, "type": "AMD", "name": f"{gpu_name_val} (ID {device_index} -> {actual_card_key_to_use})",
                    "free_mb": round(free_mb_val,1), "total_mb": round(total_mb_val,1), "used_percent": round(used_percent_val,1),
                    "message": f"AMD {gpu_name_val} (ID {device_index} -> {actual_card_key_to_use}): {free_mb_val:.0f}/{total_mb_val:.0f}MB free ({used_percent_val:.1f}% used) [rocm-smi]"}
        except FileNotFoundError: # Should be caught by shutil.which, but as a safeguard
             sysfs_fallback_result = _get_gpu_info_linux_sysfs("1002", "AMD", device_index)
             if sysfs_fallback_result and sysfs_fallback_result["success"]: return sysfs_fallback_result
             return {"success": False, "type": "AMD_ROCM_SMI_NOT_FOUND", "message": "rocm-smi utility not found. Please ensure ROCm is installed and rocm-smi is in your PATH. SysFS fallback also failed."}
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as e_rocm_cmd:
            sysfs_fallback_result = _get_gpu_info_linux_sysfs("1002", "AMD", device_index)
            if sysfs_fallback_result and sysfs_fallback_result["success"]: return sysfs_fallback_result
            return {"success": False, "type": "AMD_ROCM_SMI_CMD_ERROR", "message": f"AMD rocm-smi command error (ID {device_index}): {e_rocm_cmd}. SysFS fallback also failed."}
        except Exception as e_rocm_generic:
            sysfs_fallback_result = _get_gpu_info_linux_sysfs("1002", "AMD", device_index)
            if sysfs_fallback_result and sysfs_fallback_result["success"]: return sysfs_fallback_result
            return {"success": False, "type": "AMD_LINUX_ERROR", "message": f"AMD Linux generic error (ID {device_index}): {e_rocm_generic}. SysFS fallback also failed."}

    elif sys.platform == "win32":
        if pyadlx_available:
            try:
                helper = pyadlx.ADLXHelper()
                all_gpus_adlx_objs = helper.get_gpus()
                amd_gpu_candidates_adlx = [gpu for gpu in all_gpus_adlx_objs if hasattr(gpu, 'name') and ("amd" in gpu.name.lower() or "radeon" in gpu.name.lower())]

                if amd_gpu_candidates_adlx and 0 <= device_index < len(amd_gpu_candidates_adlx):
                    amd_gpu_obj_selected = amd_gpu_candidates_adlx[device_index]
                    vram_info_adlx = amd_gpu_obj_selected.get_vram_usage() # This can raise
                    total_mb_adlx = float(vram_info_adlx.get('vram_total_mb', 0))
                    used_mb_adlx = float(vram_info_adlx.get('vram_used_mb', 0))
                    if total_mb_adlx > 0:
                        free_mb_adlx = total_mb_adlx - used_mb_adlx
                        used_percent_adlx = (used_mb_adlx / total_mb_adlx * 100) if total_mb_adlx > 0 else 0.0
                        return {"success": True, "type": "AMD", "name": f"{amd_gpu_obj_selected.name} (ID {device_index})",
                                "free_mb": round(free_mb_adlx,1), "total_mb": round(total_mb_adlx,1), "used_percent": round(used_percent_adlx,1),
                                "message": _format_amd_win_message("pyADLX", "Success", f"{amd_gpu_obj_selected.name} (ID {device_index}): {free_mb_adlx:.0f}/{total_mb_adlx:.0f}MB free ({used_percent_adlx:.1f}% used)")}
                elif amd_gpu_candidates_adlx and device_index >= len(amd_gpu_candidates_adlx):
                     # Don't return yet, WMI might find it if pyADLX indexing is off
                     pass
            except Exception as e_pyadlx_runtime:
                # pyADLX available but failed at runtime
                if not wmi_available: # If WMI is also not available, then this is the final error.
                    return {"success": False, "type": "AMD_ADLX_RUNTIME_ERROR", "message": _format_amd_win_message("pyADLX", "Runtime Error", str(e_pyadlx_runtime))}
                # Else, WMI will be tried.
        elif pyadlx_load_error_reason and not wmi_available: # pyADLX lib itself failed to load/init, and no WMI fallback
             return {"success": False, "type": "AMD_ADLX_LOAD_ERROR", "message": pyadlx_load_error_reason}


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
                                "free_mb": 0, "total_mb": round(total_mb_wmi,1), "used_percent": 0, # WMI doesn't provide free/used for VRAM
                                "message": _format_amd_win_message("WMI", "Success (limited info)", f"{gpu_wmi_item_selected.Name} (ID {device_index}): Total {total_mb_wmi:.0f}MB (Usage info N/A via WMI)")}
                # Fall through if WMI also doesn't find the specific index or has no VRAM info
            except Exception as e_wmi_runtime:
                return {"success": False, "type": "AMD_WMI_QUERY_ERROR", "message": _format_amd_win_message("WMI", "Error during WMI query", str(e_wmi_runtime))}
        elif wmi_load_error_reason: # WMI lib itself failed to load/init
             return {"success": False, "type": "AMD_WMI_LOAD_ERROR", "message": wmi_load_error_reason}


        # If all methods failed or yielded no results for the index
        final_error_message_parts = []
        if pyadlx_load_error_reason: final_error_message_parts.append(f"pyADLX: {pyadlx_load_error_reason}")
        elif not pyadlx_available: final_error_message_parts.append("pyADLX: Not attempted or unavailable.")
        else: final_error_message_parts.append("pyADLX: Attempted but found no matching GPU or VRAM info for this index.")

        if wmi_load_error_reason: final_error_message_parts.append(f"WMI: {wmi_load_error_reason}")
        elif not wmi_available: final_error_message_parts.append("WMI: Not attempted or unavailable.")
        else: final_error_message_parts.append("WMI: Attempted but found no matching GPU or VRAM info for this index.")

        return {"success": False, "type": "AMD_WIN_DETECTION_UNAVAILABLE", "message": _format_amd_win_message("All Methods", "Failed", " ; ".join(final_error_message_parts))}

    return {"success": False, "type": "AMD_OS_UNSUPPORTED", "message": f"AMD GPU detection not implemented for OS: {sys.platform}"}


def get_gpu_info_intel(device_index: int = 0) -> Optional[Dict[str, Any]]:
    if not pyze_available:
        if sys.platform == "linux":
            sysfs_fallback_result = _get_gpu_info_linux_sysfs("8086", "Intel", device_index)
            if sysfs_fallback_result and sysfs_fallback_result["success"]:
                return sysfs_fallback_result
            final_message = (pyze_load_error_reason or "PyZE (Intel) library not available.") + " SysFS fallback also failed or found no matching GPU."
            return {"success": False, "type": "INTEL_LIB_UNAVAILABLE", "message": final_message}
        return {"success": False, "type": "INTEL_LIB_UNAVAILABLE", "message": pyze_load_error_reason or "PyZE (Intel) library not available."}

    try:
        # zeInit was called globally. Here we re-check context if needed or proceed.
        num_drivers_ptr = pyze_api.new_uint32_tp()
        pyze_api.zeDriverGet(num_drivers_ptr, None) # Check if drivers are still accessible
        num_drivers = pyze_api.uint32_tp_value(num_drivers_ptr); pyze_api.delete_uint32_tp(num_drivers_ptr)
        if num_drivers == 0:
            if sys.platform == "linux": # Attempt SysFS fallback if no PyZE drivers found
                sysfs_fallback_result = _get_gpu_info_linux_sysfs("8086", "Intel", device_index)
                if sysfs_fallback_result and sysfs_fallback_result["success"]: return sysfs_fallback_result
            return {"success": False, "type": "INTEL_PYZE_NO_DRIVERS", "message": "Intel PyZE: No drivers found. SysFS fallback also failed (Linux) or N/A."}


        drivers_array = pyze_api.new_ze_driver_handle_t_array(num_drivers)
        pyze_api.zeDriverGet(pyze_api.new_uint32_tp_assign(num_drivers), drivers_array)

        target_device_obj = None
        current_gpu_device_idx_overall = 0 # Counts only GPU type devices across all drivers

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
                    if current_gpu_device_idx_overall == device_index:
                        target_device_obj = device_candidate
                        break
                    current_gpu_device_idx_overall += 1
            pyze_api.delete_ze_device_handle_t_array(devices_array_drv)
            if target_device_obj: break
        pyze_api.delete_ze_driver_handle_t_array(drivers_array)

        if not target_device_obj:
            if sys.platform == "linux": # Attempt SysFS fallback if PyZE index not found
                sysfs_fallback_result = _get_gpu_info_linux_sysfs("8086", "Intel", device_index)
                if sysfs_fallback_result and sysfs_fallback_result["success"]: return sysfs_fallback_result
            return {"success": False, "type": "INTEL_PYZE_BAD_INDEX", "message": f"Intel PyZE: GPU Index {device_index} not found among {current_gpu_device_idx_overall} GPU(s). SysFS fallback also failed (Linux) or N/A."}


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
                # Sum up all device local memory regions
                mem_prop_item = pyze_api.ze_device_memory_properties_t_array_getitem(mem_props_array_ptr,k)
                if mem_prop_item.flags & pyze_api.ZE_DEVICE_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL:
                    total_vram_bytes += mem_prop_item.totalSize
            pyze_api.delete_ze_device_memory_properties_t_array(mem_props_array_ptr)

        total_mb = total_vram_bytes / (1024**2)
        # PyZE does not directly provide "free" VRAM in a simple way like NVML.
        # zeCommandQueueExecuteCommandLists and zeEventQueryStatus would be needed for more complex tracking.
        return {"success": True, "type": "Intel", "name": f"{gpu_name} (ID {device_index})",
                "free_mb": 0, "total_mb": round(total_mb,1), "used_percent": 0, # Reporting 0 free/used as unknown
                "message": f"Intel {gpu_name} (ID {device_index}): Total {total_mb:.0f}MB (PyZE, usage info N/A via this method)"}
    except Exception as e_pyze_runtime:
        if sys.platform == "linux": # Attempt SysFS fallback if PyZE runtime error
            sysfs_fallback_result = _get_gpu_info_linux_sysfs("8086", "Intel", device_index)
            if sysfs_fallback_result and sysfs_fallback_result["success"]: return sysfs_fallback_result
        return {"success": False, "type": "INTEL_PYZE_RUNTIME_ERROR", "message": f"Intel PyZE runtime error (ID {device_index}): {e_pyze_runtime}. SysFS fallback also failed (Linux) or N/A."}


def get_gpu_info_apple_metal(device_index: int = 0) -> Optional[Dict[str, Any]]:
    if not (sys.platform == "darwin" and metal_available):
        return {"success": False, "type": "APPLE_METAL_LIB_UNAVAILABLE", "message": metal_load_error_reason or "Metal (Apple) library/bindings not available."}
    try:
        devices = metal.MTLCopyAllDevices()
        if not devices or device_index >= len(devices) or device_index < 0:
            return {"success": False, "type": "APPLE_METAL_NO_DEVICE" if not devices else "APPLE_METAL_BAD_INDEX",
                    "message": f"Metal: {'No devices found' if not devices else f'Index {device_index} out of range ({len(devices)} found)'}."}

        selected_device = devices[device_index]
        gpu_name_raw = selected_device.name()
        gpu_name = str(gpu_name_raw) if gpu_name_raw else "Unknown Apple GPU"

        # Unified memory specifics for Apple Silicon
        total_mb_approx = 0
        # recommendedMaxWorkingSetSize is a hint for total usable memory for this device by the OS
        if hasattr(selected_device, 'recommendedMaxWorkingSetSize') and selected_device.recommendedMaxWorkingSetSize():
            total_mb_approx = selected_device.recommendedMaxWorkingSetSize() / (1024**2)
        else: # Fallback for older APIs or if the property isn't there
            if hasattr(selected_device, 'heapTextureSizeAndAlignWithDescriptor'): # Try to infer from system memory if no direct prop
                 sys_mem_info = get_system_info()
                 total_mb_approx = sys_mem_info.get('ram_total_gb',0) * 1024 * 0.75 # Heuristic: 75% of system RAM

        current_allocated_size_mb = 0
        if hasattr(selected_device, 'currentAllocatedSize'):
            current_allocated_size_mb = selected_device.currentAllocatedSize() / (1024**2)

        free_mb_approx, used_percent_approx = 0, 0
        if total_mb_approx > 0:
            free_mb_approx = total_mb_approx - current_allocated_size_mb
            used_percent_approx = (current_allocated_size_mb / total_mb_approx * 100) if total_mb_approx > 0 else 0

        return {"success": True, "type": "APPLE_METAL", "name": f"{gpu_name} (ID {device_index})",
                "free_mb": round(free_mb_approx,1), "total_mb": round(total_mb_approx,1), "used_percent": round(used_percent_approx,1),
                "message": f"Metal {gpu_name} (ID {device_index}): Approx {free_mb_approx:.0f}/{total_mb_approx:.0f}MB free ({used_percent_approx:.1f}% used) [Unified Memory]"}
    except Exception as e_metal_runtime:
        return {"success": False, "type": "APPLE_METAL_RUNTIME_ERROR", "message": f"Apple Metal runtime error (ID {device_index}): {e_metal_runtime}"}


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
        # Prioritize NVIDIA > AMD > Intel > Apple for auto-detection if multiple are present and enabled
        preferred_auto_order = [
            ("nvidia", get_gpu_info_nvidia, 0),
            ("amd", get_gpu_info_amd, 0),
            ("intel", get_gpu_info_intel, 0), # Assuming index 0 is primary for auto
        ]
        if sys.platform == "darwin": # Apple only on darwin
             preferred_auto_order.append(("apple", get_gpu_info_apple_metal, 0))


        for vendor, func, idx_to_try in preferred_auto_order:
            if gpu_detection_prefs.get(vendor, True):
                info = func(idx_to_try)
                if info:
                    all_detection_results.append(info) # Store all attempts
                    if info.get("success") and info.get("total_mb", 0.0) > 0: # Found a primary usable GPU
                        raw_gpu_info_dict = info
                        break # Stop after first successful detection with VRAM

        if not raw_gpu_info_dict: # No successful detection with VRAM > 0
            raw_gpu_info_dict = {"type": "Unknown/None_Auto", "name": "N/A", "free_mb": 0.0, "total_mb": 0.0, "success": False, "message": "Auto-detect: No primary GPU with VRAM info detected or supported."}
            # If all attempts failed, pick the "best" failure message
            for res_dict_iter in all_detection_results:
                if res_dict_iter and not res_dict_iter.get("success"):
                    # Prioritize library/init errors over "not found" if multiple types were tried
                    if "LIB_UNAVAILABLE" in res_dict_iter.get("type","") or "LOAD_ERROR" in res_dict_iter.get("type","") or "DRIVER_ISSUE" in res_dict_iter.get("type",""):
                        raw_gpu_info_dict = res_dict_iter
                        break
                    # If no lib/init error, take the first non-success as a representative error
                    if raw_gpu_info_dict.get("type") == "Unknown/None_Auto":
                         raw_gpu_info_dict = res_dict_iter

            if raw_gpu_info_dict.get("type") == "Unknown/None_Auto" or "No primary GPU" in raw_gpu_info_dict.get("message","") :
                sys_ram_info = get_system_info()
                msg_suffix = f" System RAM: {sys_ram_info['ram_free_gb']:.1f}/{sys_ram_info['ram_total_gb']:.1f}GB free."
                raw_gpu_info_dict["message"] = raw_gpu_info_dict.get("message","").rstrip('.') + msg_suffix

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
        # ... (calculations for free_mb_budgeted, used_percent_budgeted) ...

        original_hw_message = raw_gpu_info_dict.get("message", "N/A") # Message from direct HW scan
        
        # Attempt to extract vendor and the rest of the message
        vendor_prefix_found = ""
        details_after_vendor = original_hw_message
        
        for vendor_key in ["NVIDIA", "AMD", "Intel", "Metal", "APPLE_METAL"]:
            if original_hw_message.upper().startswith(vendor_key.upper()):
                # Try to split by the first colon to separate vendor/device name from stats
                parts = original_hw_message.split(":", 1)
                if len(parts) > 1:
                    vendor_prefix_found = parts[0].strip() # e.g., "NVIDIA RTX 4000 SFF Ada Generation (ID 0)"
                    details_after_vendor = parts[1].strip() # e.g., "15177/19195MB free (20.9% used) [NVML]"
                else: # No colon, assume the whole thing is the device name part before stats might be implied
                    vendor_prefix_found = original_hw_message.strip() # Treat full message as prefix if no colon
                    details_after_vendor = "" # No stats part if no colon
                break # Found vendor

        if vendor_prefix_found:
             final_return_dict["message"] = f"{vendor_prefix_found}: Manual Budget {manual_total_mb:.0f}MB ({details_after_vendor})"
        else:
             # If no clear vendor prefix was identified, fallback to a generic format
             final_return_dict["message"] = f"Manual Budget {manual_total_mb:.0f}MB ({original_hw_message})"


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
      
      desc_map_data = {
          True: {10:"MoE ULTRA MAX CPU", 8:"MoE SUPER MAX CPU", 6:"MoE SUPER CPU++", 4:"MoE SUPER CPU+", 2:"MoE SUPER CPU", 1:"MoE SUPER MAX EXPERT CPU",0:"MoE MAX EXPERT CPU", -2:"MoE CPU++", -4:"MoE CPU+", -6:"MoE CPU/GPU Bal++", -8:"MoE CPU/GPU Bal+", -10:"MoE GPU Focus",-12:"MoE GPU+", -15:"MoE GPU++", -18:"MoE GPU+++", -21:"MoE GPU++++", -25:"MoE MAX GPU"},
          False: {9:"Dense ULTRA MAX CPU", 7:"Dense SUPER MAX CPU", 5:"Dense SUPER CPU", 3:"Dense SUPER CPU-", 0:"Dense MAX FFN CPU", -1:"Dense CPU++",-3:"Dense CPU+", -5:"Dense GPU Focus", -7:"Dense GPU+", -9:"Dense GPU++", -11:"Dense GPU+++", -14:"Dense GPU++++", -17:"Dense MAX GPU"}
      }
      desc_map = desc_map_data[is_moe]
      
      applicable_keys = [k for k in desc_map if k <= attempt_level]
      closest_key = max(applicable_keys) if applicable_keys else min(desc_map.keys())
      base_desc_from_map = f"{'MoE' if is_moe else 'Dense'} Level {attempt_level} (near '{desc_map.get(closest_key, 'Custom')}')" if attempt_level not in desc_map else desc_map.get(attempt_level, f"{'MoE' if is_moe else 'Dense'} Custom (Lvl {attempt_level})")

      gpu_layers_count_for_this_level = get_gpu_layers_for_level(model_analysis, attempt_level) # This will be used
      ot_str_preview = current_ot_string[:30] + '...' if current_ot_string and len(current_ot_string) > 30 else current_ot_string
      
      layers_info: str
      if current_ot_string:
          layers_info = (f"Cmd uses '--gpulayers {gpu_layers_count_for_this_level}'. "
                         f"OT string ('{ot_str_preview}') also handles CPU offload. "
                         f"({total_model_layers} total layers)")
      else:
          layers_info = (f"Cmd will use '--gpulayers {gpu_layers_count_for_this_level}'. "
                         f"(Based on level {attempt_level}, {total_model_layers} total layers). No OT string.")
      
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

def build_command(model_path: str,
                    override_tensor_str_from_tuning: Optional[str],
                    model_analysis: dict,
                    session_base_args_dict: dict,
                    current_attempt_level_for_tuning: Optional[int] = None) -> List[str]: # Added new param
      current_cmd_args_dict = session_base_args_dict.copy()
      current_cmd_args_dict["--model"] = model_path
      model_analysis_dict = model_analysis if isinstance(model_analysis, dict) else {}

      # Auto-threads (ensure psutil_available is checked or handled if not imported globally here)
      if "--threads" in current_cmd_args_dict and str(current_cmd_args_dict["--threads"]).lower() == "auto":
          if psutil_available: # Assuming psutil_available is defined globally in the core
              try:
                  phys_cores = psutil.cpu_count(logical=False)
                  current_cmd_args_dict["--threads"] = str(max(1, phys_cores - 1 if phys_cores and phys_cores > 1 else 1)) if phys_cores and phys_cores > 0 else str(max(1, (psutil.cpu_count(logical=True) or 2) // 2))
              except Exception: current_cmd_args_dict["--threads"] = "4"
          else: current_cmd_args_dict["--threads"] = "4"

      # Auto nblas
      if current_cmd_args_dict.get("--nblas") is None or \
         (isinstance(current_cmd_args_dict.get("--nblas"), str) and \
          current_cmd_args_dict.get("--nblas").lower() == 'auto'):
          if "--nblas" in current_cmd_args_dict:
              del current_cmd_args_dict["--nblas"] # KCPP handles auto if param not present

      # GPU Layers and OverrideTensors Logic
      if override_tensor_str_from_tuning and override_tensor_str_from_tuning != "FAILURE_MAX_ATTEMPTS":
          current_cmd_args_dict["--overridetensors"] = override_tensor_str_from_tuning
          # When OT is active, gpulayers is determined by the tuning level (if provided)
          # or from session_base_args if level isn't specified (e.g. launching a remembered config)
          num_gpu_layers_for_ot_case: str
          if current_attempt_level_for_tuning is not None:
              num_gpu_layers_for_ot_case = str(get_gpu_layers_for_level(model_analysis_dict, current_attempt_level_for_tuning))
          else:
              # OT string is present, but no current tuning level context (e.g., direct launch with a pre-set OT string from model_specific_args, or history)
              # Use gpulayers from session_base_args (which might be from global/model-specific config)
              # or default to 999 if not specified, to let OT fully control eligible layers if no other gpulayers info.
              num_gpu_layers_for_ot_case = str(session_base_args_dict.get("--gpulayers", "999"))
          current_cmd_args_dict["--gpulayers"] = num_gpu_layers_for_ot_case
      else: # No OT string from tuning (or FAILURE_MAX_ATTEMPTS was hit)
          if "--overridetensors" in current_cmd_args_dict: # Ensure no stale OT string from base_args
              del current_cmd_args_dict["--overridetensors"]

          # When no OT, gpulayers determined by tuning level (if provided) or from session_base_args_dict
          num_gpu_layers_no_ot_case: str
          if current_attempt_level_for_tuning is not None:
              num_gpu_layers_no_ot_case = str(get_gpu_layers_for_level(model_analysis_dict, current_attempt_level_for_tuning))
          else:
              # This is for direct launch without tuning, use configured gpulayers
              num_gpu_layers_no_ot_case = str(session_base_args_dict.get("--gpulayers", DEFAULT_CONFIG_TEMPLATE["default_args"].get("--gpulayers", "auto")))
          current_cmd_args_dict["--gpulayers"] = num_gpu_layers_no_ot_case

      # Handle gpulayers 'auto', 'off', '0' and --nogpulayers flag
      # This must come AFTER gpulayers is determined above.
      gpulayers_val_str = str(current_cmd_args_dict.get("--gpulayers", "auto")).lower()

      if gpulayers_val_str == "auto":
          # If 'auto' is the final value (e.g. from config, not tuning level), let KCPP handle it.
          pass
      elif gpulayers_val_str in ['off', '0']:
          if "--gpulayers" in current_cmd_args_dict:
              del current_cmd_args_dict["--gpulayers"]
          current_cmd_args_dict["--nogpulayers"] = True
      else: # It's a number (e.g. "30", "999") or a KCPP specific string like "max"
          if "--nogpulayers" in current_cmd_args_dict: # If it was previously set true, remove it
              del current_cmd_args_dict["--nogpulayers"]
          # Ensure gpulayers value is a string. get_gpu_layers_for_level returns int.
          current_cmd_args_dict["--gpulayers"] = str(current_cmd_args_dict["--gpulayers"])


      # quantkv logic
      quantkv_val = current_cmd_args_dict.get("--quantkv")
      if quantkv_val is None or (isinstance(quantkv_val, str) and quantkv_val.lower() == 'auto'):
          if "--quantkv" in current_cmd_args_dict: del current_cmd_args_dict["--quantkv"]
          quant_upper, size_b_val = model_analysis_dict.get('quant', 'unknown').upper(), model_analysis_dict.get('size_b', 0)
          if any(q_str in quant_upper for q_str in ['Q5','Q6','Q8','F16','BF16','K_M','K_L','K_XL']) or \
             'XL' in quant_upper or \
             (isinstance(size_b_val, (int,float)) and size_b_val >=30):
              current_cmd_args_dict["--quantkv"] = "1" # Typically Q8_0 for K/V cache
      elif isinstance(quantkv_val, str) and quantkv_val.lower() == 'off':
          if "--quantkv" in current_cmd_args_dict: del current_cmd_args_dict["--quantkv"]

      # blasbatchsize logic
      bbs_val = current_cmd_args_dict.get("--blasbatchsize")
      if bbs_val is None or (isinstance(bbs_val, str) and bbs_val.lower() == 'auto'):
          if "--blasbatchsize" in current_cmd_args_dict: del current_cmd_args_dict["--blasbatchsize"]
          size_b_bbs = model_analysis_dict.get('size_b', 0)
          current_cmd_args_dict["--blasbatchsize"] = "128" if model_analysis_dict.get('is_moe',False) else \
                                                 ("256" if isinstance(size_b_bbs,(int,float)) and size_b_bbs > 20 else "512")
      elif isinstance(bbs_val, str) and bbs_val.lower() == 'off':
          if "--blasbatchsize" in current_cmd_args_dict: del current_cmd_args_dict["--blasbatchsize"]

      # Boolean flags
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
    if not psutil_available:
        print(f"Process Cleanup Warning: {psutil_load_error_reason or 'Psutil not available.'}")
        return False, "psutil not available for process scan."
    killed_any, results_list = False, []
    try:
        for proc_psutil in psutil.process_iter(['pid', 'name', 'exe', 'cmdline']):
            try:
                p_info_psutil = proc_psutil.info
                
                # Robust handling of name
                name_val = p_info_psutil.get('name')
                proc_name_psutil = str(name_val).lower() if name_val is not None else ""

                # Robust handling of exe
                exe_val = p_info_psutil.get('exe')
                proc_exe_psutil = str(exe_val).lower() if exe_val is not None else ""

                # Robust handling of cmdline
                cmdline_val = p_info_psutil.get('cmdline') # This can be None or a list
                
                proc_cmdline_list_psutil = []
                if cmdline_val is not None: # Explicitly check if cmdline_val is not None
                    try:
                        for arg in cmdline_val: # Iterate only if cmdline_val is a list (or other iterable)
                            proc_cmdline_list_psutil.append(str(arg).lower() if arg is not None else "")
                    except TypeError:
                        # This handles cases where cmdline_val is not None but also not iterable
                        # (e.g., an unexpected integer value from psutil for some reason).
                        # In such a case, proc_cmdline_list_psutil will remain empty, which is safe.
                        pass
                
                # proc_cmdline_list_psutil now contains lowercased string arguments or empty strings
                proc_cmdline_str_psutil = ' '.join(proc_cmdline_list_psutil)
                # proc_cmdline_str_psutil will also be effectively lowercase.

                target_name_lower = str(process_name_pattern).lower()
                cmdline_filter_lower = str(cmdline_substr_filter).lower() if cmdline_substr_filter is not None else None

                match_found = False
                # All proc_ strings are now guaranteed to be lowercase strings
                if target_name_lower in proc_name_psutil or \
                   target_name_lower in proc_exe_psutil or \
                   target_name_lower in proc_cmdline_str_psutil:

                    if cmdline_filter_lower:
                        # proc_cmdline_list_psutil contains already lowercased strings
                        match_found = any(cmdline_filter_lower in arg for arg in proc_cmdline_list_psutil)
                    else:
                        match_found = True

                if match_found:
                    success_kill, message_kill = kill_process(proc_psutil.pid)
                    if success_kill: killed_any = True
                    results_list.append(f"PID {str(proc_psutil.pid)} ('{str(p_info_psutil.get('name','NA'))}'): {str(message_kill)}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception as e_inner_loop: 
                results_list.append(f"Error processing PID {str(proc_psutil.pid if hasattr(proc_psutil, 'pid') else 'Unknown')}: {str(e_inner_loop)}")
                continue 
    except Exception as e_scan:
        return False, f"Error scanning processes: {e_scan}"

    filter_str = f" (filter: '{cmdline_substr_filter}')" if cmdline_substr_filter else ""
    msg_prefix_scan = f"Sweep for '{process_name_pattern}'{filter_str}: "
    if not results_list:
        return False, msg_prefix_scan + "No matching processes found."
    
    # Refined message for when processes are found but not killed, or errors occurred.
    if not killed_any and any("Error processing PID" in res for res in results_list):
        # This suggests all "matches" (if any) either weren't killed or errored out during processing.
        details_str = "; ".join(results_list)
        return False, msg_prefix_scan + "Matching processes may have been found, but none were killed. Errors occurred during processing. Details: " + details_str
    elif not killed_any:
        # Matches found, but kill_process reported they were not killed (e.g. already terminated, permission error from kill_process)
        details_str = "; ".join(results_list)
        return False, msg_prefix_scan + "Matching processes found but none effectively killed by this sweep (or already terminated). Details: " + details_str

    # If killed_any is True
    return True, msg_prefix_scan + "Kill attempts made. Details: " + "; ".join(results_list)

def detect_koboldcpp_capabilities(executable_path: str, force_redetect: bool = False) -> dict:
    global _kcpp_capabilities_cache
    resolved_exe_path = shutil.which(executable_path) or executable_path # Resolve path early

    if not force_redetect and resolved_exe_path in _kcpp_capabilities_cache:
        return _kcpp_capabilities_cache[resolved_exe_path]

    if not resolved_exe_path or not (os.path.exists(resolved_exe_path) or shutil.which(resolved_exe_path)): # Check again after potential `shutil.which`
        err_res = {"error": "KoboldCpp executable path not provided or invalid.",
                   "cuda": False, "rocm": False, "opencl": False, "vulkan": False,
                   "flash_attn": False, "auto_quantkv": False, "overridetensors": False,
                   "available_args": []}
        _kcpp_capabilities_cache[resolved_exe_path] = err_res # Cache error result
        return err_res

    try:
        cmd_to_run = [sys.executable, resolved_exe_path, "--help"] if resolved_exe_path.lower().endswith(".py") else [resolved_exe_path, "--help"]
        process = subprocess.run(cmd_to_run, capture_output=True, text=True,
                                 check=False, timeout=10, errors='replace')
        if process.returncode != 0 and not ("--help" in process.stdout or "--port" in process.stdout):
            error_detail = process.stderr.strip() if process.stderr else process.stdout.strip() or "Unknown error"
            err_res = {"error": f"Failed to run '{os.path.basename(resolved_exe_path)} --help'. RC={process.returncode}. Detail: {error_detail}",
                       "cuda": False, "rocm": False, "opencl": False, "vulkan": False,
                       "flash_attn": False, "auto_quantkv": False, "overridetensors": False,
                       "available_args": []}
            _kcpp_capabilities_cache[resolved_exe_path] = err_res
            return err_res

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
        _kcpp_capabilities_cache[resolved_exe_path] = features
        return features
    except subprocess.TimeoutExpired:
        err_res = {"error": f"Timeout running '{os.path.basename(resolved_exe_path)} --help'.", "cuda": False, "rocm": False, "opencl": False, "vulkan": False, "flash_attn": False, "auto_quantkv": False, "overridetensors": False, "available_args": []}
    except FileNotFoundError: # Should be caught by earlier check, but as safeguard
        err_res = {"error": f"Executable '{resolved_exe_path}' not found.", "cuda": False, "rocm": False, "opencl": False, "vulkan": False, "flash_attn": False, "auto_quantkv": False, "overridetensors": False, "available_args": []}
    except Exception as e:
        err_res = {"error": f"Error detecting KCPP capabilities: {type(e).__name__}: {e}", "cuda": False, "rocm": False, "opencl": False, "vulkan": False, "flash_attn": False, "auto_quantkv": False, "overridetensors": False, "available_args": []}

    _kcpp_capabilities_cache[resolved_exe_path] = err_res
    return err_res

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

def validate_and_resolve_koboldcpp_exe_path(exe_path_input: str) -> Tuple[Optional[str], str]:
    """
    Validates and resolves the path to the KoboldCpp executable.
    Returns a tuple: (resolved_absolute_path_or_None, message_string).
    """
    if not exe_path_input:
        return None, "KoboldCpp executable path is empty."

    # 1. Check if the provided path is absolute and valid
    potential_abs_path = os.path.abspath(exe_path_input)
    is_py_script = potential_abs_path.lower().endswith(".py")

    if os.path.exists(potential_abs_path) and (os.access(potential_abs_path, os.X_OK) or is_py_script):
        return potential_abs_path, f"Path '{exe_path_input}' resolved directly to '{potential_abs_path}'."

    # 2. Check if it's in PATH (using shutil.which)
    found_in_path_shutil = shutil.which(exe_path_input)
    if found_in_path_shutil:
        resolved_path = os.path.abspath(found_in_path_shutil)
        return resolved_path, f"Found '{os.path.basename(exe_path_input)}' in PATH: {resolved_path}."

    # 3. If it's a Python script, check relative to this core script's location (or frozen exe location)
    if exe_path_input.lower().endswith(".py"):
        try:
            # Determine the directory of the current script (core or frozen exe)
            # __file__ is path to core script if run as .py
            # sys.executable is path to frozen exe if bundled
            script_dir = os.path.dirname(os.path.abspath(__file__)) if not getattr(sys, 'frozen', False) else os.path.dirname(sys.executable)
            potential_relative_script_path = os.path.join(script_dir, exe_path_input)
            if os.path.exists(potential_relative_script_path):
                resolved_path = os.path.abspath(potential_relative_script_path)
                return resolved_path, f"Found Python script '{os.path.basename(exe_path_input)}' relative to launcher/core: {resolved_path}."
        except NameError: # __file__ might not be defined in some contexts (e.g. interactive)
            pass
        except Exception as e_rel_check: # Catch any other error during relative check
             return None, f"Error checking relative path for Python script '{exe_path_input}': {e_rel_check}. Path not found or not executable."


    # 4. If all else fails
    return None, f"KoboldCpp target '{exe_path_input}' (resolved to '{potential_abs_path}') not found, not executable, not in PATH, or not a valid relative script."


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
            "koboldcpp_capabilities": kcpp_caps,
            "config_file_path": CONFIG_FILE} # Added for CLI to display

def _cleanup_nvml():
    global pynvml_available # Ensure we check the potentially modified global
    if pynvml_available:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass
        except Exception:
            pass

if __name__ == "__main__":
    print("TensorTune Core Library Test - v" + DEFAULT_CONFIG_TEMPLATE["launcher_core_version"])
    if appdirs_load_error_reason: print(f"Appdirs status: {appdirs_load_error_reason}")
    if psutil_load_error_reason: print(f"Psutil status: {psutil_load_error_reason}")
    if pynvml_load_error_reason: print(f"PyNVML status: {pynvml_load_error_reason}")
    if pyadlx_load_error_reason: print(f"PyADLX status: {pyadlx_load_error_reason}")
    if wmi_load_error_reason: print(f"WMI status: {wmi_load_error_reason}")
    if pyze_load_error_reason: print(f"PyZE status: {pyze_load_error_reason}")
    if metal_load_error_reason: print(f"Metal status: {metal_load_error_reason}")


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
    for g in list_amd_gpus_linux(): print(f"  ID {g['id']}: {g['name']} (Source: {g.get('source','N/A')})")
    print("Listing Intel GPUs:")
    for g in list_intel_gpus(): print(f"  ID {g['id']}: {g['name']} (Source: {g.get('source','N/A')})")
    print("Listing Apple GPUs:")
    for g in list_apple_gpus(): print(f"  ID {g['id']}: {g['name']} (Source: {g.get('source','N/A')})")


    _cleanup_nvml()

import atexit
atexit.register(_cleanup_nvml)
