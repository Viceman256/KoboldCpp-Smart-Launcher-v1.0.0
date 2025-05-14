import os
import sys
import subprocess
import re
import json
import time
import threading  # Not used in this slimmed down core, but often useful
import signal  # Not used in this slimmed down core
import sqlite3
from datetime import datetime, timezone
import pathlib  # Now used for Path.home()
import shutil
import platform
from pathlib import Path

# --- Appdirs Integration ---
APP_NAME = "KoboldCppLauncher"
APP_AUTHOR = "KoboldCppLauncherUser" # Generic author, can be changed

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
        # Fallback to a common pattern: ~/.config/AppName on Linux/Mac, ~/AppData/Roaming/AppName on Win (simplified)
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
        # Fallback to a common pattern: ~/.local/share/AppName on Linux/Mac, ~/AppData/Local/AppName on Win (simplified)
        if sys.platform == "win32":
            path = os.path.join(Path.home(), "AppData", "Local", APP_NAME, "Data")
        elif sys.platform == "darwin":
            path = os.path.join(Path.home(), "Library", "Application Support", APP_NAME, "Data") # Often same as config on Mac
        else: # Linux and other XDG-like systems
            path = os.path.join(Path.home(), ".local", "share", APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path

# --- Constants and Configuration ---
_CONFIG_FILE_BASENAME = "kobold_launcher_config.json"
CONFIG_FILE = os.path.join(_get_user_app_config_dir(), _CONFIG_FILE_BASENAME)

_DB_FILE_BASENAME_DEFAULT = "kobold_launcher_history.db" # Default name, path resolved at runtime

DEFAULT_CONFIG_TEMPLATE = {
    "koboldcpp_executable": "koboldcpp.exe" if sys.platform == "win32" else "./koboldcpp",
    "default_gguf_dir": "",
    "last_used_gguf_dir": "",
    "db_file": _DB_FILE_BASENAME_DEFAULT, # Store basename, actual path resolved in load_config
    "first_run_completed": False,
    "first_run_intro_shown": False,
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
    "kobold_success_pattern": r"Starting Kobold API on port (\d+)",
    "oom_error_keywords": [
        "cuda out of memory", "outofmemory", "out of vram", "cublasstatusallocfailed",
        "ggml_cuda_host_malloc: failed to allocate", "ggml-cuda backend: failed to allocate",
        "failed to allocate memory on gpu", "vram allocation failed",
        "llama_new_context_with_model: failed to load model", "unable to initialize backend",
        "failed to load model", "model load failed", "segmentation fault", "aborted",
        "illegal instruction", "clblast error", "opencl error", "rocm error", "hip error"
    ]
}

# --- Runtime Variables & Optional Dependency Management ---
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
        pyadlx_available = True 
    except ImportError:
        pyadlx_available = False
    try:
        import wmi
        wmi_available = True
    except ImportError:
        wmi_available = False

try:
    import pyze.api as pyze_api 
    pyze_api.zeInit(0) 
    num_drivers_ptr = pyze_api.new_uint32_tp()
    pyze_api.zeDriverGet(num_drivers_ptr, None)
    if pyze_api.uint32_tp_value(num_drivers_ptr) > 0:
         pyze_available = True
    pyze_api.delete_uint32_tp(num_drivers_ptr) 
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

# --- Datetime Adapters for SQLite ---
def adapt_datetime_iso(val):
    if val.tzinfo is None:
        val = val.astimezone(timezone.utc)
    else:
        val = val.astimezone(timezone.utc)
    return val.isoformat().replace("+00:00", "Z")

def convert_datetime(val):
    try:
        dt_obj = datetime.fromisoformat(val.decode().replace('Z', '+00:00'))
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        return dt_obj
    except ValueError:
        try:
            dt_obj = datetime.strptime(val.decode(), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            return dt_obj
        except ValueError:
            return None

sqlite3.register_adapter(datetime, adapt_datetime_iso)
sqlite3.register_converter("DATETIME", convert_datetime)
sqlite3.register_converter("TIMESTAMP", convert_datetime)


# --- Configuration Management ---
def save_launcher_config(config_to_save):
    """Saves the configuration. Expects config_to_save["db_file"] to be absolute path."""
    try:
        # Ensure config dir exists (should be handled by _get_user_app_config_dir())
        # os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True) # Redundant if _get_user_app_config_dir used correctly

        config_copy_for_saving = config_to_save.copy()
        
        # If db_file is in the default data directory, store only its basename in the JSON
        # Otherwise, store the full path (allows user to place DB elsewhere)
        default_data_dir = _get_user_app_data_dir()
        if "db_file" in config_copy_for_saving:
            db_file_abs = os.path.abspath(config_copy_for_saving["db_file"])
            if os.path.dirname(db_file_abs) == default_data_dir:
                config_copy_for_saving["db_file"] = os.path.basename(db_file_abs)
            # else: it's already an absolute path, store as is.

        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_copy_for_saving, f, indent=4)
        return True, f"Configuration saved to {CONFIG_FILE}"
    except Exception as e:
        return False, f"Error saving configuration: {e}"

def load_config():
    """Loads configuration. Resolves db_file to an absolute path."""
    config_data = DEFAULT_CONFIG_TEMPLATE.copy()
    
    # Resolve default db_file path initially (in case config file doesn't exist or db_file is missing)
    default_db_basename = DEFAULT_CONFIG_TEMPLATE["db_file"]
    config_data["db_file"] = os.path.join(_get_user_app_data_dir(), default_db_basename)
    # Ensure default data dir exists for the DB
    os.makedirs(os.path.dirname(config_data["db_file"]), exist_ok=True)

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # Merge user_config into config_data (which started as DEFAULT_CONFIG_TEMPLATE)
            for key, default_value in DEFAULT_CONFIG_TEMPLATE.items():
                if key in user_config:
                    if isinstance(default_value, dict) and isinstance(user_config[key], dict):
                        # Deep merge for specific dictionary keys
                        if key in ["default_args", "gpu_detection", "model_specific_args"]:
                            merged_sub_dict = default_value.copy()
                            merged_sub_dict.update(user_config[key])
                            config_data[key] = merged_sub_dict
                        else: # Shallow update for other dicts (overwrites default)
                            config_data[key] = user_config[key]
                    else: # Non-dict values, user_config takes precedence
                        config_data[key] = user_config[key]
            
            # Ensure all default_args keys are present in the loaded config_data["default_args"]
            if "default_args" in config_data:
                for arg_key, arg_default_val in DEFAULT_CONFIG_TEMPLATE["default_args"].items():
                    if arg_key not in config_data["default_args"]:
                        config_data["default_args"][arg_key] = arg_default_val
            else: # Should not happen if merge logic is correct, but as a safeguard
                config_data["default_args"] = DEFAULT_CONFIG_TEMPLATE["default_args"].copy()

            # Special handling for db_file path resolution
            db_file_from_user_config = user_config.get("db_file", default_db_basename)
            if not os.path.isabs(db_file_from_user_config):
                config_data["db_file"] = os.path.join(_get_user_app_data_dir(), db_file_from_user_config)
            else: # User specified an absolute path
                config_data["db_file"] = db_file_from_user_config
            
            # Ensure directory for the potentially custom db_file exists
            os.makedirs(os.path.dirname(config_data["db_file"]), exist_ok=True)

            return config_data, True, f"Loaded configuration from {CONFIG_FILE}"

        except Exception as e:
            # On error, config_data is already based on DEFAULT_CONFIG_TEMPLATE
            # and its db_file is already resolved to default data dir.
            # Ensure all default_args keys are present if the partial load messed them up.
            if "default_args" not in config_data:
                 config_data["default_args"] = DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
            else:
                for arg_key, arg_default_val in DEFAULT_CONFIG_TEMPLATE["default_args"].items():
                    if arg_key not in config_data["default_args"]:
                        config_data["default_args"][arg_key] = arg_default_val
            return config_data, False, f"Error loading {CONFIG_FILE}: {e}. Using defaults and ensuring all default_args keys. DB path set to default."
    else:
        # No config file found, config_data is DEFAULT_CONFIG_TEMPLATE, db_file already resolved.
        # Ensure all default_args keys are present (should be by default from template copy)
        config_data["default_args"] = DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
        return config_data, False, f"No config file found at {CONFIG_FILE}. Using defaults. DB path set to default."


# --- Database Functions ---
def init_db(db_file): # db_file is expected to be an absolute path
    conn = None
    try:
        # Directory for db_file should have been created by load_config or _get_user_app_data_dir
        # but an extra check here is harmless.
        db_dir = os.path.dirname(db_file)
        if db_dir: # Ensure db_dir is not empty (e.g. if db_file is just "name.db" in CWD)
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
        # Add new columns if they don't exist (idempotent)
        cols_to_check = {"launch_outcome": "TEXT", "approx_vram_used_kcpp_mb": "INTEGER"}
        for col, col_type in cols_to_check.items():
            try:
                cursor.execute(f"SELECT {col} FROM launch_history LIMIT 1")
            except sqlite3.OperationalError: # Column does not exist
                cursor.execute(f"ALTER TABLE launch_history ADD COLUMN {col} {col_type}")
        
        # Add Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lh_model_details ON launch_history (model_filepath, model_quant_type, is_moe);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lh_timestamp ON launch_history (timestamp DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lh_model_filepath ON launch_history (model_filepath);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lh_launch_outcome ON launch_history (launch_outcome);")
        # Composite index for find_best_historical_config's common query pattern
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_lh_lookup_for_best ON launch_history (
                model_filepath, model_quant_type, is_moe, 
                launch_outcome, vram_at_launch_decision_mb, attempt_level_used, timestamp DESC
            );
        """)

        conn.commit()
        return True, f"Database initialized at {db_file}"
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
           command_args_list_with_exe[1].lower().endswith(".py"):
            num_prefix_items_to_skip = 2

        args_to_save = command_args_list_with_exe[num_prefix_items_to_skip:] if command_args_list_with_exe else []
        args_json_str = json.dumps(args_to_save)
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
            ''', (model_filepath, model_size_to_db, model_analysis.get('quant'), model_analysis.get('is_moe', False),
                  vram_at_decision_mb_int, args_json_str, attempt_level, outcome, approx_vram_used_kcpp_mb_int, current_timestamp))
            success_msg = f"Saved configuration to database (Outcome: {outcome})."
        except sqlite3.IntegrityError: # Entry already exists, update it
            cursor.execute('''
                UPDATE launch_history SET launch_outcome = ?, approx_vram_used_kcpp_mb = ?, timestamp = ?
                WHERE model_filepath = ? AND (vram_at_launch_decision_mb = ? OR (vram_at_launch_decision_mb IS NULL AND ? IS NULL))
                AND kobold_args_json = ? AND attempt_level_used = ?
            ''', (outcome, approx_vram_used_kcpp_mb_int, current_timestamp, model_filepath,
                  vram_at_decision_mb_int, vram_at_decision_mb_int, args_json_str, attempt_level))
            if cursor.rowcount == 0: return False, f"Failed to update existing DB entry for outcome {outcome}."
            else: success_msg = f"Updated configuration in database (Outcome: {outcome})."
        conn.commit()
        return True, success_msg
    except Exception as e:
        return False, f"Could not save/update config to DB: {e}"
    finally:
        if conn: conn.close()

def find_best_historical_config(db_file, current_model_analysis, current_available_dedicated_vram_mb):
    conn = None
    try:
        conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        cursor = conn.cursor()
        vram_tolerance_percent = 0.25  # Tolerance for VRAM matching (e.g., +/- 25%)
        model_size_tolerance_b = 0.5   # Absolute tolerance for model size matching in Billion Parameters (e.g., +/- 0.5B)

        if current_available_dedicated_vram_mb is None or current_available_dedicated_vram_mb <= 0:
            min_hist_vram, max_hist_vram, target_hist_vram_sort = -1, float('inf'), 8192 # Default sort target if no VRAM info
        else:
            min_hist_vram = current_available_dedicated_vram_mb * (1 - vram_tolerance_percent)
            max_hist_vram = current_available_dedicated_vram_mb * (1 + vram_tolerance_percent)
            target_hist_vram_sort = current_available_dedicated_vram_mb
        
        model_size_query_val = current_model_analysis.get('size_b')
        model_size_query_for_db = None # Handle if size_b is not a valid float for DB query
        if not (isinstance(model_size_query_val, str) or model_size_query_val is None):
            try: model_size_query_for_db = float(model_size_query_val)
            except (ValueError, TypeError): pass

        # Updated query with model_size_tolerance_b
        query = """
            SELECT kobold_args_json, attempt_level_used, vram_at_launch_decision_mb, launch_outcome, approx_vram_used_kcpp_mb
            FROM launch_history
            WHERE model_filepath = ? AND model_quant_type = ? AND is_moe = ?
              AND (? IS NULL OR model_size_b IS NULL OR ABS(model_size_b - ?) < ?) -- Model size match (or one is NULL, or within tolerance)
              AND (vram_at_launch_decision_mb BETWEEN ? AND ? OR vram_at_launch_decision_mb IS NULL) -- VRAM match or no VRAM history
            ORDER BY
              CASE WHEN launch_outcome LIKE 'SUCCESS_USER_CONFIRMED%' THEN 0
                   WHEN launch_outcome LIKE 'SUCCESS_USER_SAVED_GOOD%' THEN 1
                   WHEN launch_outcome LIKE 'SUCCESS_LOAD_VRAM_OK%' THEN 2
                   WHEN launch_outcome LIKE 'SUCCESS_USER_DIRECT_LAUNCH%' THEN 3
                   ELSE 10 END ASC, -- Prioritize best outcomes
              ABS(COALESCE(vram_at_launch_decision_mb, ?) - ?) ASC, -- Closest VRAM match
              attempt_level_used ASC, -- Prefer lower attempt levels (more GPU) for similar VRAM/outcome
              timestamp DESC -- Newest entry as tie-breaker
            LIMIT 1
        """
        cursor.execute(query, (current_model_analysis['filepath'], current_model_analysis.get('quant'),
                               current_model_analysis.get('is_moe', False), 
                               model_size_query_for_db, model_size_query_for_db, model_size_tolerance_b,
                               min_hist_vram, max_hist_vram,
                               target_hist_vram_sort, target_hist_vram_sort))
        row = cursor.fetchone()
        if row:
            try:
                loaded_args = json.loads(row[0])
                return {"args_list": [str(item) for item in loaded_args], "attempt_level": row[1],
                        "historical_vram_mb": row[2], "outcome": row[3], "approx_vram_used_kcpp_mb": row[4]}
            except json.JSONDecodeError: return None
        return None
    except sqlite3.Error as e:
        # Consider logging this error if you have a logging mechanism
        # print(f"Database error in find_best_historical_config: {e}") # For debugging
        return None 
    finally:
        if conn: conn.close()

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
    except sqlite3.Error: return []
    finally:
        if conn: conn.close()

# --- System Information and Hardware Detection --- (Content from original prompt, no changes needed here for this request)
def get_system_info():
    info = {"cpu_model": "Unknown", "cpu_cores_physical": "N/A", "cpu_cores_logical": "N/A",
            "ram_total_gb": 0, "ram_free_gb": 0, "ram_used_percent": 0,
            "os_name": sys.platform, "os_version": "Unknown", "python_version": sys.version}
    try:
        if sys.platform == "win32": info["os_name"] = "Windows"; info["os_version"] = platform.version()
        elif sys.platform == "linux":
            info["os_name"] = "Linux"
            try:
                with open('/etc/os-release') as f: info["os_version"] = next((l.split('=')[1].strip().strip('"') for l in f if l.startswith('PRETTY_NAME=')), "Unknown")
            except Exception: info["os_version"] = "Unknown (os-release error)"
        elif sys.platform == "darwin": info["os_name"] = "macOS"; info["os_version"] = platform.mac_ver()[0]

        if sys.platform == "win32": info["cpu_model"] = platform.processor()
        elif sys.platform == "linux":
            try:
                with open('/proc/cpuinfo', 'r') as f: info["cpu_model"] = next((l.split(':')[1].strip() for l in f if l.startswith('model name')), "Unknown")
            except Exception: info["cpu_model"] = "Unknown (cpuinfo error)"
        elif sys.platform == "darwin":
            try:
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True, check=False, timeout=2)
                if result.returncode == 0 and result.stdout.strip(): info["cpu_model"] = result.stdout.strip()
                else: info["cpu_model"] = "Unknown (sysctl error)"
            except Exception: info["cpu_model"] = "Unknown (sysctl exception)"
    except Exception: pass

    if psutil_available:
        try:
            info["cpu_cores_physical"] = psutil.cpu_count(logical=False) or "N/A"
            info["cpu_cores_logical"] = psutil.cpu_count(logical=True) or "N/A"
            mem = psutil.virtual_memory()
            info["ram_total_gb"] = mem.total / (1024**3)
            info["ram_free_gb"] = mem.available / (1024**3)
            info["ram_used_percent"] = mem.percent
        except Exception: pass
    return info

def detect_koboldcpp_capabilities(executable_path):
    """Detect available features/flags in KoboldCpp executable"""
    try:
        process = subprocess.run([executable_path, "--help"], 
                               capture_output=True, text=True, check=False, timeout=5)
        
        if process.returncode != 0:
            error_detail = process.stderr.strip() if process.stderr else "Unknown error"
            return {"error": f"Failed to run KoboldCpp with --help. RC={process.returncode}. Detail: {error_detail}", 
                    "cuda": False, "rocm": False, "vulkan": False, "flash_attn": False, 
                    "auto_quantkv": False, "overridetensors": False, "available_args": []}
        
        output = process.stdout
        
        features = {
            "cuda": "--usecublas" in output,
            "rocm": "--usehipblas" in output or "--userocmblas" in output,
            "vulkan": "--vulkan" in output, 
            "flash_attn": "--flashattention" in output,
            "auto_quantkv": "--quantkv auto" in output or ("--quantkv" in output and "auto" in output), 
            "overridetensors": "--overridetensors" in output,
            "available_args": []
        }
        
        arg_pattern = r'(-{1,2}[\w-]+)' 
        features["available_args"] = re.findall(arg_pattern, output)
        
        return features
    except subprocess.TimeoutExpired:
        return {"error": "Timeout running KoboldCpp with --help", 
                "cuda": False, "rocm": False, "vulkan": False, "flash_attn": False, 
                "auto_quantkv": False, "overridetensors": False, "available_args": []}
    except Exception as e:
        return {"error": str(e), 
                "cuda": False, "rocm": False, "vulkan": False, "flash_attn": False, 
                "auto_quantkv": False, "overridetensors": False, "available_args": []}

def get_gpu_info_nvidia():
    if not pynvml_available: return None
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mb, total_mb = mem_info.free / (1024**2), mem_info.total / (1024**2)
            name_raw = pynvml.nvmlDeviceGetName(handle)
            name = name_raw.decode('utf-8') if isinstance(name_raw, bytes) else name_raw
            used_percent = ((total_mb - free_mb) / total_mb * 100) if total_mb > 0 else 0
            return {"success": True, "type": "NVIDIA", "name": name, "free_mb": free_mb,
                    "total_mb": total_mb, "used_percent": used_percent,
                    "message": f"NVIDIA {name}: {free_mb:.0f}/{total_mb:.0f}MB free ({used_percent:.1f}% used)"}
        return {"success": False, "type": "NVIDIA_NONE_FOUND", "message": "No NVIDIA GPUs detected."}
    except pynvml.NVMLError as e: 
        return {"success": False, "type": "NVIDIA_ERROR", "message": f"NVIDIA NVML error: {e}"}
    except Exception as e:
        return {"success": False, "type": "NVIDIA_ERROR", "message": f"NVIDIA generic error: {e}"}

def get_gpu_info_amd():
    if sys.platform == "linux":
        try:
            rocm_check = subprocess.run(["which", "rocm-smi"], capture_output=True, text=True, check=False, timeout=2)
            if rocm_check.returncode != 0: return None
            try:
                json_proc = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json"],
                    capture_output=True, text=True, check=True, timeout=5
                )
                data = json.loads(json_proc.stdout)
                card_key = next(iter(data)) 
                gpu_data = data[card_key]
                total_mem_bytes_str = gpu_data.get("VRAM Total Memory (B)")
                used_mem_bytes_str = gpu_data.get("VRAM Used Memory (B)")
                if not total_mem_bytes_str or not used_mem_bytes_str:
                     return {"success": False, "type": "AMD", "message": "VRAM info missing in rocm-smi JSON."}
                total_mb_val = int(total_mem_bytes_str) / (1024**2)
                used_mb_val = int(used_mem_bytes_str) / (1024**2)
                free_mb_val = total_mb_val - used_mb_val
                gpu_name_val = gpu_data.get("Card SKU", gpu_data.get("Card series", "AMD Radeon GPU"))
                used_percent_val = (used_mb_val / total_mb_val * 100) if total_mb_val > 0 else 0
                return {
                    "success": True, "type": "AMD", "name": gpu_name_val,
                    "free_mb": free_mb_val, "total_mb": total_mb_val,
                    "used_percent": used_percent_val,
                    "message": f"AMD {gpu_name_val}: {free_mb_val:.0f}/{total_mb_val:.0f}MB free ({used_percent_val:.1f}% used) [rocm-smi JSON]"
                }
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, StopIteration):
                rocm_info = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, check=False, timeout=3)
                if rocm_info.returncode != 0: return None
                output = rocm_info.stdout
                total_match = re.search(r"vram\s+total\s+memory\s*:\s*(\d+)\s*(\w+)", output, re.IGNORECASE)
                used_match = re.search(r"vram\s+used\s+memory\s*:\s*(\d+)\s*(\w+)", output, re.IGNORECASE)
                if total_match and used_match:
                    total_val = float(total_match.group(1)); used_val = float(used_match.group(1))
                    total_unit = total_match.group(2).lower(); used_unit = used_match.group(2).lower()
                    if total_unit == "gb": total_val *= 1024
                    if used_unit == "gb": used_val *= 1024
                    free_mb = total_val - used_val
                    gpu_info_name = subprocess.run(["rocm-smi", "--showproductname"], capture_output=True, text=True, check=False, timeout=3)
                    gpu_name = "AMD GPU"
                    if gpu_info_name.returncode == 0:
                        name_match = re.search(r"GPU\s+\[\d+\]\s*:\s*(.+)", gpu_info_name.stdout)
                        if name_match: gpu_name = name_match.group(1).strip()
                    used_percent = (used_val / total_val * 100) if total_val > 0 else 0
                    return {"success": True, "type": "AMD", "name": gpu_name, "free_mb": free_mb,
                            "total_mb": total_val, "used_percent": used_percent,
                            "message": f"AMD {gpu_name}: {free_mb:.0f}/{total_val:.0f}MB free ({used_percent:.1f}% used) [rocm-smi Regex]"}
                return None 
        except (FileNotFoundError, subprocess.SubprocessError, Exception) as e:
            return {"success": False, "type": "AMD_LINUX_ERROR", "message": f"AMD Linux rocm-smi error: {e}"}

    elif sys.platform == "win32":
        if wmi_available:
            try:
                c = wmi.WMI()
                for gpu_wmi in c.Win32_VideoController():
                    name_lower = gpu_wmi.Name.lower()
                    if "amd" in name_lower or "radeon" in name_lower:
                        total_mb = int(gpu_wmi.AdapterRAM) / (1024**2) if gpu_wmi.AdapterRAM else 0
                        return {
                            "success": True, "type": "AMD", "name": gpu_wmi.Name,
                            "free_mb": 0, "total_mb": total_mb, "used_percent": 0,
                            "message": f"AMD {gpu_wmi.Name}: Total {total_mb:.0f}MB (Free VRAM not available via WMI)"
                        }
                return None 
            except Exception as e_wmi:
                return {"success": False, "type": "AMD_WMI_ERROR", "message": f"AMD WMI error: {e_wmi}"}
    return None

def get_gpu_info_intel():
    if pyze_available:
        try:
            num_drivers_ptr = pyze_api.new_uint32_tp()
            pyze_api.zeDriverGet(num_drivers_ptr, None)
            num_drivers = pyze_api.uint32_tp_value(num_drivers_ptr)
            if num_drivers == 0:
                pyze_api.delete_uint32_tp(num_drivers_ptr); return None 
            drivers_array = pyze_api.new_ze_driver_handle_t_array(num_drivers)
            pyze_api.zeDriverGet(num_drivers_ptr, drivers_array)
            pyze_api.delete_uint32_tp(num_drivers_ptr)
            for i in range(num_drivers):
                driver = pyze_api.ze_driver_handle_t_array_getitem(drivers_array, i)
                num_devices_ptr = pyze_api.new_uint32_tp()
                pyze_api.zeDeviceGet(driver, num_devices_ptr, None)
                num_devices = pyze_api.uint32_tp_value(num_devices_ptr)
                if num_devices == 0:
                    pyze_api.delete_uint32_tp(num_devices_ptr); continue
                devices_array = pyze_api.new_ze_device_handle_t_array(num_devices)
                pyze_api.zeDeviceGet(driver, num_devices_ptr, devices_array)
                pyze_api.delete_uint32_tp(num_devices_ptr)
                for j in range(num_devices):
                    device = pyze_api.ze_device_handle_t_array_getitem(devices_array, j)
                    props = pyze_api.ze_device_properties_t()
                    pyze_api.memset(pyze_api.addressof(props), 0, pyze_api.sizeof(props))
                    props.stype = pyze_api.ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
                    pyze_api.zeDeviceGetProperties(device, props)
                    if props.type == pyze_api.ZE_DEVICE_TYPE_GPU:
                        gpu_name_bytes = bytes(props.name) 
                        gpu_name = gpu_name_bytes.decode('utf-8', errors='ignore').rstrip('\x00')
                        mem_props_count_ptr = pyze_api.new_uint32_tp()
                        pyze_api.zeDeviceGetMemoryProperties(device, mem_props_count_ptr, None)
                        mem_props_count = pyze_api.uint32_tp_value(mem_props_count_ptr)
                        total_vram_bytes = 0
                        if mem_props_count > 0:
                            mem_props_array = pyze_api.new_ze_device_memory_properties_t_array(mem_props_count)
                            for k_mem in range(mem_props_count): 
                                mem_prop_item_ptr = pyze_api.ze_device_memory_properties_t_array_getitem_ptr(mem_props_array, k_mem)
                                pyze_api.memset(mem_prop_item_ptr, 0, pyze_api.sizeof(pyze_api.ze_device_memory_properties_t()))
                                pyze_api.ze_device_memory_properties_t_stype_set(mem_prop_item_ptr, pyze_api.ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES)
                            pyze_api.zeDeviceGetMemoryProperties(device, mem_props_count_ptr, mem_props_array)
                            for k_mem in range(mem_props_count):
                                mem_prop_item = pyze_api.ze_device_memory_properties_t_array_getitem(mem_props_array, k_mem)
                                total_vram_bytes += mem_prop_item.totalSize
                            pyze_api.delete_ze_device_memory_properties_t_array(mem_props_array)
                        pyze_api.delete_uint32_tp(mem_props_count_ptr)
                        total_mb = total_vram_bytes / (1024**2)
                        pyze_api.delete_ze_device_handle_t_array(devices_array)
                        pyze_api.delete_ze_driver_handle_t_array(drivers_array)
                        return {
                            "success": True, "type": "Intel", "name": gpu_name,
                            "free_mb": 0, "total_mb": total_mb, "used_percent": 0,
                            "message": f"Intel {gpu_name}: Total {total_mb:.0f}MB (Free VRAM not available via pyze)"
                        }
                pyze_api.delete_ze_device_handle_t_array(devices_array)
            pyze_api.delete_ze_driver_handle_t_array(drivers_array)
            return None 
        except Exception as e_pyze:
            return {"success": False, "type": "INTEL_PYZE_ERROR", "message": f"Intel pyze error: {e_pyze}"}

    if sys.platform == "win32" and wmi_available:
        try:
            c = wmi.WMI()
            for gpu_wmi in c.Win32_VideoController():
                if "intel" in gpu_wmi.Name.lower():
                    total_mb = int(gpu_wmi.AdapterRAM) / (1024**2) if gpu_wmi.AdapterRAM else 0
                    return {
                        "success": True, "type": "Intel", "name": gpu_wmi.Name,
                        "free_mb": 0, "total_mb": total_mb, "used_percent": 0,
                        "message": f"Intel {gpu_wmi.Name}: Total {total_mb:.0f}MB (Free VRAM not available via WMI)"
                    }
            return None 
        except Exception as e_wmi:
            return {"success": False, "type": "INTEL_WMI_ERROR", "message": f"Intel WMI error: {e_wmi}"}
    return None

def get_gpu_info_apple_metal():
    if not (sys.platform == "darwin" and metal_available):
        return None
    try:
        devices = metal.MTLCopyAllDevices()
        if not devices:
            return {"success": False, "message": "Metal: No devices found."}
        selected_device = devices[0]
        for dev_test in devices: 
            if not dev_test.isLowPower():
                selected_device = dev_test; break
        gpu_name = selected_device.name()
        total_mb_approx = selected_device.recommendedMaxWorkingSetSize() / (1024**2) if selected_device.recommendedMaxWorkingSetSize() else 0
        return {
            "success": True, "type": "APPLE_METAL", "name": gpu_name,
            "free_mb": 0, 
            "total_mb": total_mb_approx, "used_percent": 0,
            "message": f"Metal {gpu_name}: Approx {total_mb_approx:.0f}MB GPU working set"
        }
    except Exception as e_metal:
        return {"success": False, "type": "APPLE_METAL_ERROR", "message": f"Apple Metal error: {e_metal}"}
                           
def get_available_vram_mb():
    nvidia_info = get_gpu_info_nvidia()
    if nvidia_info and nvidia_info.get("success"):
        return nvidia_info["free_mb"], nvidia_info["total_mb"], nvidia_info["message"], nvidia_info
    amd_info = get_gpu_info_amd()
    if amd_info and amd_info.get("success"):
        return amd_info.get("free_mb", 0.0), amd_info.get("total_mb", 0.0), amd_info["message"], amd_info
    intel_info = get_gpu_info_intel()
    if intel_info and intel_info.get("success"):
        return intel_info.get("free_mb", 0.0), intel_info.get("total_mb", 0.0), intel_info["message"], intel_info
    if sys.platform == "darwin":
        metal_info = get_gpu_info_apple_metal()
        if metal_info and metal_info.get("success"):
            return metal_info.get("free_mb", 0.0), metal_info.get("total_mb", 0.0), metal_info["message"], metal_info
    sys_ram_info = get_system_info()
    fallback_msg = f"No dedicated GPU detected or info error. System RAM: {sys_ram_info['ram_free_gb']:.1f}/{sys_ram_info['ram_total_gb']:.1f}GB free."
    return 0.0, 0.0, fallback_msg, {"type": "Unknown", "name": "N/A", "free_mb": 0.0, "total_mb": 0.0, "message": fallback_msg, "success": False}

# --- Model Analysis Functions --- (Content from original prompt, no changes needed here for this request)
def analyze_filename(filepath):
    filename_lower = os.path.basename(filepath).lower()
    analysis = {'filepath': filepath, 'is_moe': False, 'quant': 'unknown', 'size_b': 0, 'details': {}, 'num_layers': 32}
    if 'moe' in filename_lower or 'mixtral' in filename_lower or 'grok' in filename_lower or re.search(r'-a\d+(\.\d+)?[bB]', filename_lower):
        analysis['is_moe'] = True
    quant_match = re.search(r'(q[2-8](?:_[0ksmKSML]{1,2})?|iq[1-4](?:_[smlxSMLX]{1,2})?|bpw\d+|bf16|f16|f32|ggml|exl\d|awq|gptq|q_k_l|q_k_m|q_k_s|q_k_xl)', filename_lower)
    if quant_match: analysis['quant'] = quant_match.group(1).upper()
    size_match = re.search(r'(?<![a-zA-Z0-9_])(\d{1,3}(?:\.\d{1,2})?)[bB](?![piA-Z0-9_])', filename_lower)
    if not size_match: size_match = re.search(r'(?<![a-zA-Z0-9_])(\d{1,3}(?:\.\d{1,2})?)[bB][-_]', filename_lower)
    if size_match:
        try:
            size_val = float(size_match.group(1))
            analysis['size_b'] = int(size_val) if size_val.is_integer() else size_val
        except ValueError: pass
    current_size_b = analysis.get('size_b', 0)
    if (isinstance(current_size_b, (int, float)) and current_size_b == 0) or not isinstance(current_size_b, (int, float)):
        if os.path.exists(filepath):
            file_size_gb = os.path.getsize(filepath) / (1024**3); quant_l = analysis.get('quant', 'unknown').lower(); est_b = 0
            q_map = {'iq1': 0.28, 'q2_0': 0.28, 'q2_k_s': 0.28, 'iq2_xs': 0.30, 'iq2_s': 0.30, 'q2_k': 0.30,
                     'iq2': 0.35, 'q3_0': 0.35, 'q3_k_s': 0.35, 'iq3_xs': 0.40, 'iq3_s': 0.40, 'q3_k': 0.40, 'q3_k_m': 0.40,
                     'iq3_m': 0.50, 'iq3_l': 0.50, 'q4_0': 0.50, 'q4_k_s': 0.50, 'iq4_xs': 0.55, 'iq4_s': 0.55, 'q4_k_m': 0.55,
                     'q5_0': 0.60, 'q5_k_s': 0.60, 'q5_k_m': 0.65, 'q5_1': 0.65, 'q6_k': 0.80, 'q8_0': 1.05,
                     'f16': 2.05, 'bf16': 2.05, 'f32': 4.05}
            for q_key, div in q_map.items():
                if q_key in quant_l: est_b = file_size_gb / div; break
            if est_b > 0.5:
                common_sizes = sorted([1, 1.5, 2.7, 3, 7, 8, 11, 13, 15, 20, 22, 27, 30, 32, 33, 34, 35, 40, 65, 70, 120, 180, 235])
                closest_size = min(common_sizes, key=lambda x: abs(x - est_b))
                if abs(closest_size - est_b) < closest_size * 0.25:
                    analysis['size_b'] = int(closest_size) if float(closest_size).is_integer() else closest_size
                    analysis['details']['size_is_estimated'] = True
    layer_patterns = [r'(\d+)l', r'l(\d+)', r'-(\d+)layers', r'(\d+)layers']
    model_layer_defaults = {'gemma': 28 if analysis.get('size_b') and analysis.get('size_b') < 10 else 32, 
                            'llama2': 32, 'llama3': 32, 'mistral': 32, 'mixtral': 32, 'qwen': 32,
                            'phi': 32, 'gpt-j': 28, 'gpt-neox': 44, 'pythia': 32, 'falcon': 32}
    num_layers = None
    for pattern in layer_patterns:
        match = re.search(pattern, filename_lower)
        if match:
            try: num_layers = int(match.group(1)); break
            except ValueError: pass
    if num_layers is None:
        for model_name, default_layers in model_layer_defaults.items():
            if model_name in filename_lower: num_layers = default_layers; break
    if num_layers is None:
        size_b = analysis.get('size_b', 0)
        if isinstance(size_b, (int, float)):
            if size_b <= 3: num_layers = 24
            elif size_b <= 7: num_layers = 32
            elif size_b <= 13: num_layers = 40
            elif size_b <= 20: num_layers = 48
            elif size_b <= 35: num_layers = 60
            elif size_b <= 70: num_layers = 80
            else: num_layers = 96
    analysis['num_layers'] = num_layers or 32
    return analysis

def get_gpu_layers_for_level(model_analysis, attempt_level):
    total_layers = model_analysis.get('num_layers', 32); is_moe = model_analysis.get('is_moe', False)
    if is_moe:
        if attempt_level <= -23: return 999 
        elif attempt_level <= -19: return int(total_layers * 0.95)
        elif attempt_level <= -16: return int(total_layers * 0.9)
        elif attempt_level <= -13: return int(total_layers * 0.85)
        elif attempt_level <= -10: return int(total_layers * 0.8)
        elif attempt_level <= -7: return int(total_layers * 0.75)
        elif attempt_level <= -4: return int(total_layers * 0.7)
        elif attempt_level <= -1: return int(total_layers * 0.6)
        elif attempt_level <= 2: return int(total_layers * 0.5)
        elif attempt_level <= 5: return int(total_layers * 0.4)
        else: return int(total_layers * 0.3)
    else:
        if attempt_level <= -15: return 999
        elif attempt_level <= -12: return int(total_layers * 0.95)
        elif attempt_level <= -10: return int(total_layers * 0.9)
        elif attempt_level <= -8: return int(total_layers * 0.85)
        elif attempt_level <= -6: return int(total_layers * 0.8)
        elif attempt_level <= -4: return int(total_layers * 0.7)
        elif attempt_level <= -2: return int(total_layers * 0.6)
        elif attempt_level <= 0: return int(total_layers * 0.5)
        elif attempt_level <= 2: return int(total_layers * 0.4)
        elif attempt_level <= 4: return int(total_layers * 0.3)
        else: return int(total_layers * 0.2)

def get_level_from_overridetensors(override_tensor_str, model_analysis):
    is_moe = model_analysis.get('is_moe', False)
    if not override_tensor_str: return -25 if is_moe else -17
    if is_moe:
        if "attn\\.(q|k|v|o)\\.weight" in override_tensor_str: return 8
        elif "attn\\.(q|k)\\.weight" in override_tensor_str: return 6
        elif "attn\\.q\\.weight" in override_tensor_str: return 4
        if "all_exp_ffn" in override_tensor_str or "ffn_.*_exps\\.weight=CPU" in override_tensor_str:
            if "ffn_(down|up|gate).*weight=CPU" in override_tensor_str: return 1
            return 0
        elif re.search(r"\.\d+\.ffn_(up|down|gate)_exps", override_tensor_str):
            if "ffn_down_exps" in override_tensor_str and "ffn_up_exps" in override_tensor_str:
                if "ffn_gate_exps" in override_tensor_str: return -2
                return -4
            return -8
        elif "ffn_down_exps" in override_tensor_str:
            if re.search(r"blk\.\d+\.ffn_down_exps", override_tensor_str):
                if "blk\\.(0|4|8|12|16|20|24|28)\\." in override_tensor_str: return -18
                elif "blk\\.\\d*[02468]\\." in override_tensor_str: return -12
                return -10
            return -20
    else:
        if "attn\\.(q|k|v|o)\\.weight" in override_tensor_str: return 7
        elif "attn\\.(q|k)\\.weight" in override_tensor_str: return 5
        elif "attn\\.q\\.weight" in override_tensor_str: return 3
        if re.search(r"\.\d+\.(ffn_up|ffn_down|ffn_gate)\.weight=CPU", override_tensor_str): return 0
        elif re.search(r"\.\d+\.(ffn_up|ffn_down)\.weight=CPU", override_tensor_str): return -1
        elif re.search(r"\.\d+\.ffn_up\.weight=CPU", override_tensor_str):
            if "blk\\.\\d*[0369]\\." in override_tensor_str: return -7
            elif "blk\\.\\d*[048]\\." in override_tensor_str: return -9
            return -5
    return -10 if is_moe else -5

def get_offload_description(model_analysis, attempt_level, current_ot_string):
    if current_ot_string == "FAILURE_MAX_ATTEMPTS":
        return "MAX ATTEMPTS REACHED. No further CPU offload possible with current strategy."

    is_moe = model_analysis.get('is_moe', False)
    total_model_layers = model_analysis.get('num_layers', 32)
    
    # Determine base description using user's original detailed mapping logic
    base_desc_from_map = "" # This will be populated by the logic below
    
    if is_moe:
        moe_desc_map = {
            10: "MoE ULTRA MAX CPU (All FFN)", 8: "MoE SUPER MAX CPU (All FFN + 50% Attn)", 
            6: "MoE SUPER CPU++ (All FFN + 25% Attn)", 4: "MoE SUPER CPU+ (All FFN + 10% Attn)", 
            2: "MoE SUPER CPU (All FFN + 5% Attn)", 1: "MoE SUPER MAX EXPERT CPU",
            0: "MoE MAX EXPERT CPU", -2: "MoE CPU++ (All Exp FFN)", 
            -4: "MoE CPU+ (Exp: down,up + 1/2gate)", -6: "MoE CPU/GPU Bal++ (Exp: down,up)", 
            -8: "MoE CPU/GPU Bal+ (Exp: down+1/2up)", -10: "MoE GPU Focus (Exp: all down)",
            -12: "MoE GPU+ (Exp: 1/2 down)", -15: "MoE GPU++ (Exp: 1/4 down)", 
            -18: "MoE GPU+++ (Exp: 1/8 down)", -21: "MoE GPU++++ (Exp: 1/16 down)", 
            -25: "MoE MAX GPU" # Represents no OT string, full GPU layers attempt
        }
        # Find the key in the map that is less than or equal to the attempt_level
        # and is the largest such key (closest defined level at or below current attempt).
        # If no such key (e.g., attempt_level is below the lowest defined key), default to the lowest.
        applicable_keys = [k for k in moe_desc_map.keys() if k <= attempt_level]
        closest_level_key = max(applicable_keys) if applicable_keys else min(moe_desc_map.keys())
        
        if attempt_level not in moe_desc_map: # If current level is between defined map points
            lower_key = closest_level_key # This is the highest defined level <= attempt_level
            # Find the smallest defined level > attempt_level for "between" description
            upper_keys_candidates = [k for k in moe_desc_map.keys() if k > attempt_level]
            upper_key = min(upper_keys_candidates) if upper_keys_candidates else lower_key # Fallback to lower_key if no upper
            
            if lower_key != upper_key and lower_key in moe_desc_map and upper_key in moe_desc_map:
                base_desc_from_map = f"MoE Level {attempt_level} (between '{moe_desc_map[lower_key]}' and '{moe_desc_map[upper_key]}')"
            else: # Fallback if "between" logic can't find distinct points or one is missing
                base_desc_from_map = moe_desc_map.get(closest_level_key, f"MoE Custom (Lvl {attempt_level})")
        else: # attempt_level is an exact key in the map
            base_desc_from_map = moe_desc_map.get(attempt_level, f"MoE Custom (Lvl {attempt_level})")
    else: # Dense models
        dense_desc_map = {
            9: "Dense ULTRA MAX CPU (All Layers)", 7: "Dense SUPER MAX CPU (All FFN + 50% Attn)", 
            5: "Dense SUPER CPU (All FFN + 25% Attn)", 3: "Dense SUPER CPU- (All FFN + 10% Attn)", 
            0: "Dense MAX FFN CPU", -1: "Dense CPU++ (Base FFN: up,down)",
            -3: "Dense CPU+ (Base FFN: up+1/2down)", -5: "Dense GPU Focus (Base FFN: all up)", 
            -7: "Dense GPU+ (Base FFN: 1/4 up)", -9: "Dense GPU++ (Base FFN: 1/8 up)", 
            -11: "Dense GPU+++ (Base FFN: 1/16 up)", -14: "Dense GPU++++ (Base FFN: 1/32 up)", 
            -17: "Dense MAX GPU" # Represents no OT string, full GPU layers attempt
        }
        applicable_keys = [k for k in dense_desc_map.keys() if k <= attempt_level]
        closest_level_key = max(applicable_keys) if applicable_keys else min(dense_desc_map.keys())

        if attempt_level not in dense_desc_map:
            lower_key = closest_level_key
            upper_keys_candidates = [k for k in dense_desc_map.keys() if k > attempt_level]
            upper_key = min(upper_keys_candidates) if upper_keys_candidates else lower_key
            
            if lower_key != upper_key and lower_key in dense_desc_map and upper_key in dense_desc_map:
                 base_desc_from_map = f"Dense Level {attempt_level} (between '{dense_desc_map[lower_key]}' and '{dense_desc_map[upper_key]}')"
            else:
                base_desc_from_map = dense_desc_map.get(closest_level_key, f"Dense Custom (Lvl {attempt_level})")
        else:
            base_desc_from_map = dense_desc_map.get(attempt_level, f"Dense Custom (Lvl {attempt_level})")

    # Get the descriptive GPU layer count for this attempt level (for UI/understanding)
    descriptive_gpu_layers_count = get_gpu_layers_for_level(model_analysis, attempt_level)
    
    # Construct the final detailed description
    layers_info_detail = ""
    if current_ot_string: # If an OT string is generated and active for this level
        layers_info_detail = (f"Command will use '--gpulayers 999' (all {total_model_layers} layers targeted for GPU). "
                              f"The OT string ('{current_ot_string}') then moves specific tensors to CPU. "
                              f"(Descriptive layer equivalent for this OT level: {descriptive_gpu_layers_count}/{total_model_layers})")
    else: # No OT string (e.g., "Max GPU" level where generate_overridetensors returns None)
          # In this case, descriptive_gpu_layers_count (often 999 for max GPU levels) will be the actual target for --gpulayers in the command.
        layers_info_detail = (f"Command will use '--gpulayers {descriptive_gpu_layers_count}' "
                              f"(based on level {attempt_level}, out of {total_model_layers} total layers). "
                              f"No specific tensor offload pattern active for this level.")

    return f"{base_desc_from_map}. {layers_info_detail}"

def generate_overridetensors(model_analysis, attempt_level):
    moe_s, dense_s = "_exps\\.weight", "\\.weight"; d, u, g = "ffn_down", "ffn_up", "ffn_gate"
    all_exp_ffn, all_base_ffn = f"({d}|{u}|{g}){moe_s}", f"({d}|{u}|{g}){dense_s}"
    all_l, even_l, l0369, l048 = "blk\\.\\d+\\.", "blk\\.\\d*[02468]\\.", "blk\\.\\d*[0369]\\.", "blk\\.\\d*[048]\\."
    parts = []
    if model_analysis.get('is_moe'):
        if attempt_level >= 8: parts.extend([f"{all_l}{all_exp_ffn}", f"{all_l}{all_base_ffn}", f"{all_l}attn\\.(q|k|v|o)\\.weight"])
        elif attempt_level >= 6: parts.extend([f"{all_l}{all_exp_ffn}", f"{all_l}{all_base_ffn}", f"{even_l}attn\\.(q|k)\\.weight"])
        elif attempt_level >= 4: parts.extend([f"{all_l}{all_exp_ffn}", f"{all_l}{all_base_ffn}", f"{l0369}attn\\.q\\.weight"])
        elif attempt_level >= 2: parts.extend([f"{all_l}{all_exp_ffn}", f"{all_l}{all_base_ffn}", f"{l048}attn\\.q\\.weight"])
        elif attempt_level == 1: parts.extend([f"{all_l}{all_exp_ffn}", f"{all_l}{all_base_ffn}"])
        elif attempt_level == 0: parts.extend([f"{all_l}{all_exp_ffn}", f"{all_l}{d}{dense_s}"])
        elif attempt_level >= -1: parts.extend([f"{all_l}{all_exp_ffn}", f"blk\\.[0-3]\\d\\.{d}{dense_s}"])
        elif attempt_level >= -2: parts.append(f"{all_l}{all_exp_ffn}")
        elif attempt_level >= -3: parts.extend([f"{all_l}({d}|{u}){moe_s}", f"blk\\.[0-2]\\d\\.{g}{moe_s}"])
        elif attempt_level >= -4: parts.extend([f"{all_l}({d}|{u}){moe_s}", f"{even_l}{g}{moe_s}"])
        elif attempt_level >= -5: parts.extend([f"{all_l}({d}|{u}){moe_s}", f"blk\\.[0-9]\\.{g}{moe_s}"])
        elif attempt_level >= -6: parts.append(f"{all_l}({d}|{u}){moe_s}")
        elif attempt_level >= -7: parts.extend([f"{all_l}{d}{moe_s}", f"blk\\.[0-3]\\d\\.{u}{moe_s}"])
        elif attempt_level >= -8: parts.extend([f"{all_l}{d}{moe_s}", f"{even_l}{u}{moe_s}"])
        elif attempt_level >= -9: parts.extend([f"{all_l}{d}{moe_s}", f"blk\\.[0-9]\\.{u}{moe_s}"])
        elif attempt_level >= -10: parts.append(f"{all_l}{d}{moe_s}")
        elif attempt_level >= -11: parts.append(f"blk\\.[0-3]\\d\\.{d}{moe_s}")
        elif attempt_level >= -12: parts.append(f"{even_l}{d}{moe_s}")
        elif attempt_level >= -15: parts.append(f"{l048}{d}{moe_s}")
        elif attempt_level >= -18: parts.append(f"blk\\.(0|4|8|12|16|20|24|28)\\.{d}{moe_s}")
        elif attempt_level >= -21: parts.append(f"blk\\.(0|8|16|24|32)\\.{d}{moe_s}")
    else:
        if attempt_level >= 7: parts.extend([f"{all_l}{all_base_ffn}", f"{all_l}attn\\.(q|k|v|o)\\.weight"])
        elif attempt_level >= 5: parts.extend([f"{all_l}{all_base_ffn}", f"{even_l}attn\\.(q|k)\\.weight"])
        elif attempt_level >= 3: parts.extend([f"{all_l}{all_base_ffn}", f"{l0369}attn\\.q\\.weight"])
        elif attempt_level >= 1: parts.extend([f"{all_l}{all_base_ffn}", f"{l048}attn\\.q\\.weight"])
        elif attempt_level == 0: parts.append(f"{all_l}{all_base_ffn}")
        elif attempt_level >= -1: parts.append(f"{all_l}({d}|{u}){dense_s}")
        elif attempt_level >= -2: parts.extend([f"{all_l}{u}{dense_s}", f"blk\\.[0-3]\\d\\.{d}{dense_s}"])
        elif attempt_level >= -3: parts.extend([f"{all_l}{u}{dense_s}", f"{even_l}{d}{dense_s}"])
        elif attempt_level >= -4: parts.extend([f"{all_l}{u}{dense_s}", f"blk\\.[0-9]\\.{d}{dense_s}"])
        elif attempt_level >= -5: parts.append(f"{all_l}{u}{dense_s}")
        elif attempt_level >= -6: parts.append(f"blk\\.[0-3]\\d\\.{u}{dense_s}")
        elif attempt_level >= -7: parts.append(f"{l0369}{u}{dense_s}")
        elif attempt_level >= -9: parts.append(f"{l048}{u}{dense_s}")
        elif attempt_level >= -11: parts.append(f"blk\\.(0|4|8|12|16|20|24|28)\\.{u}{dense_s}")
        elif attempt_level >= -14: parts.append(f"blk\\.(0|8|16|24|32)\\.{u}{dense_s}")
    if not parts: return None
    return f"({'|'.join(parts)})=CPU" if len(parts) > 1 else f"{parts[0]}=CPU"

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
    preferred_order = ["--threads", "--nblas", "--port", "--contextsize", "--promptlimit", 
                       "--gpulayers", "--nogpulayers", "--usecublas", "--flashattention", 
                       "--nommap", "--lowvram", "--quantkv", "--blasbatchsize", 
                       "--overridetensors", "--defaultgenamt"]
    processed_keys = set(["--model"])
    for key in preferred_order:
        if key in args_dict and key not in processed_keys:
            value = args_dict[key]
            if value is True: cmd_list_part.append(key)
            elif value is not False: cmd_list_part.extend([key, str(value)])
            processed_keys.add(key)
    for key in sorted([k for k in args_dict if k not in processed_keys]):
        value = args_dict[key]
        if value is True: cmd_list_part.append(key)
        elif value is not False: cmd_list_part.extend([key, str(value)])
    return cmd_list_part

def format_command_for_display(cmd_list):
    if not cmd_list: return ""
    exe_part = cmd_list[0]; args_start_index = 1
    if len(cmd_list) > 1 and cmd_list[0].lower() == sys.executable.lower() and cmd_list[1].lower().endswith(".py"):
        exe_part = f'"{cmd_list[0]}" "{cmd_list[1]}"'; args_start_index = 2
    else:
        if " " in cmd_list[0] and not (cmd_list[0].startswith('"') and cmd_list[0].endswith('"')): exe_part = f'"{cmd_list[0]}"'
    formatted_cmd = [exe_part]; i = args_start_index
    while i < len(cmd_list):
        item = cmd_list[i]
        if item.startswith("--"):
            formatted_cmd.append(f"\n  {item}")
            if i + 1 < len(cmd_list) and not cmd_list[i+1].startswith("--"):
                value = cmd_list[i+1]
                if (' ' in value or any(c in value for c in ['\\', '/', ':'])) and \
                   not value.startswith("blk.") and not value.isdigit() and \
                   not (value.startswith("(") and value.endswith("=CPU")) and \
                   not (value.startswith('"') and value.endswith('"')):
                    formatted_cmd.append(f' "{value}"')
                else: formatted_cmd.append(f" {value}")
                i += 1
        i += 1
    return "".join(formatted_cmd)

def get_command_to_run(executable_path, args_list):
    if executable_path.lower().endswith(".py"): return [sys.executable, executable_path] + args_list
    else: return [executable_path] + args_list

def build_command(model_path, override_tensor_str_from_tuning, model_analysis, session_base_args_dict):
    current_cmd_args_dict = session_base_args_dict.copy()
    current_cmd_args_dict["--model"] = model_path # Ensure model path is always set/updated

    model_analysis = model_analysis if isinstance(model_analysis, dict) else {}

    # Auto-threads logic
    if "--threads" in current_cmd_args_dict and str(current_cmd_args_dict["--threads"]).lower() == "auto":
        if psutil_available: # global psutil_available should be defined
            try:
                physical_cores = psutil.cpu_count(logical=False)
                if physical_cores and physical_cores > 0:
                    current_cmd_args_dict["--threads"] = str(max(1, physical_cores - 1 if physical_cores > 1 else 1))
                else: # Fallback to logical cores if physical not determined
                    logical_cores = psutil.cpu_count(logical=True) or 2 # Default to 2 if cpu_count is None
                    current_cmd_args_dict["--threads"] = str(max(1, logical_cores // 2))
            except Exception: # Broad except for any psutil issue
                current_cmd_args_dict["--threads"] = "4" # Fallback
        else:
            current_cmd_args_dict["--threads"] = "4" # Fallback if psutil not available

    # Auto-nblas logic (remove if "auto", KoboldCpp handles its own default then)
    nblas_val = current_cmd_args_dict.get("--nblas")
    if nblas_val is None or (isinstance(nblas_val, str) and nblas_val.lower() == 'auto'):
        if "--nblas" in current_cmd_args_dict:
            del current_cmd_args_dict["--nblas"]

    # --- GPU Layers and OverrideTensors Logic ---
    if override_tensor_str_from_tuning and override_tensor_str_from_tuning != "FAILURE_MAX_ATTEMPTS":
        current_cmd_args_dict["--overridetensors"] = override_tensor_str_from_tuning
        # CRITICAL CHANGE: If OT is active from tuning, always aim for max GPU layers as the base.
        # The OT string will then carve out what's needed.
        current_cmd_args_dict["--gpulayers"] = "999"
    else:
        # No OT string active from tuning logic (e.g., "Max GPU" level where OT string is None, or direct launch).
        # Ensure --overridetensors is not in the command if it was in session_base_args_dict or if OT string is None.
        if "--overridetensors" in current_cmd_args_dict:
            del current_cmd_args_dict["--overridetensors"]
        
        # --gpulayers should be what's in session_base_args_dict (reflecting config/user session edits for non-OT scenarios).
        # current_cmd_args_dict started as a copy of session_base_args_dict.
        base_gpulayers_val_from_session = session_base_args_dict.get("--gpulayers")
        
        if base_gpulayers_val_from_session is not None:
            current_cmd_args_dict["--gpulayers"] = str(base_gpulayers_val_from_session)
        else:
            # Fallback if --gpulayers is somehow missing from session_base_args_dict (e.g. very old config or empty dict passed)
            # Use the launcher's core template's default for --gpulayers.
            template_default_gpulayers = DEFAULT_CONFIG_TEMPLATE["default_args"].get("--gpulayers", "auto")
            current_cmd_args_dict["--gpulayers"] = str(template_default_gpulayers)

    # Subsequent logic for --gpulayers "off"/"0" to --nogpulayers conversion:
    gpulayers_val_finalized = current_cmd_args_dict.get("--gpulayers")
    if isinstance(gpulayers_val_finalized, str) and gpulayers_val_finalized.lower() in ['off', '0']:
        if "--gpulayers" in current_cmd_args_dict: # Remove --gpulayers if it's "off" or "0"
            del current_cmd_args_dict["--gpulayers"]
        current_cmd_args_dict["--nogpulayers"] = True # Use --nogpulayers instead
    elif "--nogpulayers" in current_cmd_args_dict:
        # If --gpulayers is being set to something other than "off" or "0" (e.g., "999", "auto", a number),
        # ensure --nogpulayers (which might have been True in session_base_args_dict) is removed.
        if not (isinstance(gpulayers_val_finalized, str) and gpulayers_val_finalized.lower() in ['off', '0']):
            del current_cmd_args_dict["--nogpulayers"]
    
    # Auto-quantkv logic
    quantkv_val = current_cmd_args_dict.get("--quantkv")
    if quantkv_val is None or (isinstance(quantkv_val, str) and quantkv_val.lower() == 'auto'):
        if "--quantkv" in current_cmd_args_dict:
            # Remove it if "auto" was specified, allowing KCPP default or smart default below to take over.
            del current_cmd_args_dict["--quantkv"] 
        
        # Smart default for quantkv if "auto" was specified in config and not overridden by user
        quant_upper = model_analysis.get('quant', 'unknown').upper()
        model_size_b = model_analysis.get('size_b', 0)
        if any(q_check in quant_upper for q_check in ['Q5', 'Q6', 'Q8', 'F16', 'BF16', 'K_M', 'K_L', 'K_XL']) or \
           'XL' in quant_upper or (isinstance(model_size_b, (int, float)) and model_size_b >= 30):
            current_cmd_args_dict["--quantkv"] = "1" # Use Q8_0 for larger/higher quant models
        # else: KCPP default will apply (often F16 if --quantkv is not present)
    elif isinstance(quantkv_val, str) and quantkv_val.lower() == 'off':
        if "--quantkv" in current_cmd_args_dict: # Remove if explicitly "off"
            del current_cmd_args_dict["--quantkv"]

    # Auto-blasbatchsize logic
    blasbatchsize_val = current_cmd_args_dict.get("--blasbatchsize")
    if blasbatchsize_val is None or (isinstance(blasbatchsize_val, str) and blasbatchsize_val.lower() == 'auto'):
        if "--blasbatchsize" in current_cmd_args_dict:
            # Remove it if "auto", allowing KCPP default or smart default below.
            del current_cmd_args_dict["--blasbatchsize"]
        
        # Smart default for blasbatchsize if "auto" was specified
        model_size_b = model_analysis.get('size_b', 0)
        if model_analysis.get('is_moe', False):
            current_cmd_args_dict["--blasbatchsize"] = "128"
        elif isinstance(model_size_b, (int, float)) and model_size_b > 20:
            current_cmd_args_dict["--blasbatchsize"] = "256"
        else: # Smaller models or unknown size
            current_cmd_args_dict["--blasbatchsize"] = "512"
    elif isinstance(blasbatchsize_val, str) and blasbatchsize_val.lower() == 'off':
        if "--blasbatchsize" in current_cmd_args_dict: # Remove if explicitly "off"
            del current_cmd_args_dict["--blasbatchsize"]

    # Ensure boolean flags are correctly represented (True) or removed if False
    boolean_flags_to_manage = ["--usecublas", "--flashattention", "--nommap", "--lowvram"]
    for flag in boolean_flags_to_manage:
        if flag in current_cmd_args_dict and current_cmd_args_dict[flag] is False:
            del current_cmd_args_dict[flag]
        # If current_cmd_args_dict[flag] is True, it will be included by args_dict_to_list.
        # If the flag is not in current_cmd_args_dict at all, it won't be included.

    return args_dict_to_list(current_cmd_args_dict)
    
def kill_process(pid, force=True):
    if not pid: return False, "No PID provided."
    try:
        if sys.platform == "win32":
            args = ["taskkill"]; startupinfo = subprocess.STARTUPINFO()
            if force: args.extend(["/F", "/T"]); startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            args.extend(["/PID", str(pid)])
            process = subprocess.Popen(args, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
            stdout, stderr = process.communicate(timeout=5)
            if process.returncode == 0: return True, f"Process {pid} terminated."
            elif process.returncode == 128 or "could not find the process" in stderr.decode(errors='ignore').lower() or \
                 "no running instance" in stderr.decode(errors='ignore').lower():
                return True, f"Process {pid} not found or already terminated."
            else: return False, f"Failed to kill process {pid}: RC={process.returncode}, Err={stderr.decode(errors='ignore').strip()}"
        else: os.kill(pid, signal.SIGKILL if force else signal.SIGTERM); return True, f"Signal sent to process {pid}."
    except ProcessLookupError: return True, f"Process {pid} not found (already terminated)."
    except subprocess.TimeoutExpired: return False, f"Killing process {pid} timed out."
    except Exception as e: return False, f"Error killing process {pid}: {e}"

def kill_processes_by_name(process_name_pattern, cmdline_substr_filter=None):
    if not psutil_available: return False, "psutil not available for process scan."
    killed_any, results = False, []
    try:
        for proc in psutil.process_iter(['pid', 'name', 'exe', 'cmdline']):
            try:
                proc_name = proc.info.get('name', '').lower(); proc_exe = (proc.info.get('exe', '') or '').lower()
                proc_cmdline_list = proc.info.get('cmdline', []); proc_cmdline_str = ' '.join(proc_cmdline_list).lower()
                target = process_name_pattern.lower(); match_found = False
                if target in proc_name or target in proc_exe or target in proc_cmdline_str:
                    if cmdline_substr_filter:
                        if any(cmdline_substr_filter.lower() in arg.lower() for arg in proc_cmdline_list): match_found = True
                    else: match_found = True
                if match_found:
                    success, message = kill_process(proc.pid)
                    if success: killed_any = True
                    results.append(f"PID {proc.pid} ('{proc.info.get('name','')}'): {message}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess): continue
    except Exception as e: return False, f"Error scanning processes: {e}"
    if killed_any: return True, f"Sweep for '{process_name_pattern}' (filter: '{cmdline_substr_filter}'): " + "; ".join(results)
    else: return False, f"No processes matching '{process_name_pattern}' (filter: '{cmdline_substr_filter}') found or killed by sweep."

def launch_process(cmd, capture_output=True, new_console=False, use_text_mode=True):
    try:
        kwargs = {}
        if capture_output:
            kwargs.update({'stdout': subprocess.PIPE, 'stderr': subprocess.STDOUT})
            if use_text_mode: kwargs.update({'text': True, 'universal_newlines': True, 'bufsize': 1})
        if sys.platform == "win32":
            creation_flags = 0
            if new_console: creation_flags = subprocess.CREATE_NEW_CONSOLE
            elif capture_output: creation_flags = subprocess.CREATE_NO_WINDOW
            if creation_flags != 0: kwargs['creationflags'] = creation_flags
        process = subprocess.Popen(cmd, **kwargs)
        return process, None
    except FileNotFoundError: return None, f"Executable '{cmd[0]}' not found."
    except PermissionError: return None, f"Permission denied for '{cmd[0]}'."
    except Exception as e: return None, f"Launch error: {type(e).__name__}: {e}"

def initialize_launcher():
    """Initializes the launcher: loads config, inits DB, sets up dynamic defaults."""
    config, config_loaded, config_message = load_config()
    
    # db_file from config is now an absolute path resolved by load_config
    db_absolute_path = config.get("db_file") # Should always be present and absolute
    if not db_absolute_path: # Should not happen if load_config is correct
        db_absolute_path = os.path.join(_get_user_app_data_dir(), _DB_FILE_BASENAME_DEFAULT)
        config["db_file"] = db_absolute_path # Ensure it's in config for future saves
        os.makedirs(os.path.dirname(db_absolute_path), exist_ok=True)


    db_success, db_message = init_db(db_absolute_path)
    
    # Set auto threads if psutil available and config is 'auto'
    if psutil_available and config["default_args"].get("--threads") == "auto":
        try:
            physical_cores = psutil.cpu_count(logical=False)
            if physical_cores and physical_cores > 0:
                config["default_args"]["--threads"] = str(max(1, physical_cores - 1 if physical_cores > 1 else 1))
            else:
                config["default_args"]["--threads"] = str(max(1, (psutil.cpu_count(logical=True) or 2) // 2))
        except Exception:
            config["default_args"]["--threads"] = "4" # Fallback
    elif config["default_args"].get("--threads") == "auto": # psutil not available but still auto
        config["default_args"]["--threads"] = "4" # Fallback
    
    # Determine best GPU for conditional flags based on detection
    _, _, _, gpu_details = get_available_vram_mb()
    if gpu_details and gpu_details.get("type") == "NVIDIA":
        config["default_args"]["--usecublas"] = True 
    else:
        config["default_args"]["--usecublas"] = False
        # Example for AMD/ROCm (if KCPP adds a specific flag like --usehipblas)
        # if gpu_details and gpu_details.get("type") == "AMD" and sys.platform == "linux":
        #     # Assuming a flag like '--usehipblas' or '--userocmblas'
        #     # Check if KCPP capabilities report this flag before setting
        #     kcpp_caps = detect_koboldcpp_capabilities(config.get("koboldcpp_executable"))
        #     if kcpp_caps.get("rocm"): # "rocm" capability implies hipblas/rocmblas is available
        #          config["default_args"]["--usehipblas"] = True # Or the correct flag name

    return {
        "initialized": config_loaded and db_success, "config": config,
        "config_loaded": config_loaded, "config_message": config_message,
        "db_success": db_success, "db_message": db_message,
        "system_info": get_system_info(), "gpu_info": gpu_details
    }

# At the end of the script, for pynvml proper shutdown:
def _cleanup_nvml():
    if pynvml_available:
        try:
            pynvml.nvmlShutdown()
        except: # noqa: E722
            pass # Ignore errors during shutdown

if __name__ == "__main__":
    print("KoboldCpp Core Library Test")
    print(f"Default Config File Path: {CONFIG_FILE}")
    init_results = initialize_launcher()
    print(f"Initialization successful: {init_results['initialized']}")
    print(f"Config loaded: {init_results['config_loaded']} ({init_results['config_message']})")
    print(f"  Config 'db_file' resolved to: {init_results['config']['db_file']}")
    print(f"DB status: {init_results['db_success']} ({init_results['db_message']})")
    
    print("\nSystem Info:")
    for k, v in init_results['system_info'].items():
        print(f"  {k}: {v}")
        
    print("\nGPU Info:")
    gpu_dict = init_results['gpu_info']
    if gpu_dict and gpu_dict.get("success"):
        print(f"  Type: {gpu_dict.get('type')}")
        print(f"  Name: {gpu_dict.get('name')}")
        print(f"  Total VRAM: {gpu_dict.get('total_mb'):.0f} MB")
        print(f"  Free VRAM: {gpu_dict.get('free_mb'):.0f} MB")
        print(f"  Message: {gpu_dict.get('message')}")
    else:
        print(f"  Message: {gpu_dict.get('message', 'GPU detection failed or no GPU found.')}")

    dummy_exe_name = "dummy_koboldcpp.bat" if sys.platform == "win32" else "dummy_koboldcpp.sh"
    if not os.path.exists(init_results["config"]["koboldcpp_executable"]):
        print(f"\nKoboldCpp executable '{init_results['config']['koboldcpp_executable']}' not found.")
        print(f"Creating a dummy executable '{dummy_exe_name}' in CWD for capability detection test.")
        # Create dummy in CWD for simplicity of test, real exe path is in config
        dummy_exe_path_for_test_local = os.path.join(os.getcwd(), dummy_exe_name)
        if sys.platform == "win32":
            with open(dummy_exe_path_for_test_local, "w") as f:
                f.write("@echo off\n"); f.write("echo KoboldCpp Dummy Help\n")
                f.write("echo --usecublas\n"); f.write("echo --usehipblas\n"); f.write("echo --vulkan\n")
                f.write("echo --flashattention\n"); f.write("echo --quantkv auto\n"); f.write("echo --overridetensors\n")
                f.write("echo --model <path>\n"); f.write("echo --port <port>\n")
        else:
            with open(dummy_exe_path_for_test_local, "w") as f:
                f.write("#!/bin/sh\n"); f.write("echo 'KoboldCpp Dummy Help'\n")
                f.write("echo '--usecublas'\n"); f.write("echo '--usehipblas'\n"); f.write("echo '--vulkan'\n")
                f.write("echo '--flashattention'\n"); f.write("echo '--quantkv auto'\n"); f.write("echo '--overridetensors'\n")
                f.write("echo '--model <path>'\n"); f.write("echo '--port <port>'\n")
            os.chmod(dummy_exe_path_for_test_local, 0o755)
        kcpp_exe_path_for_test = dummy_exe_path_for_test_local
    else:
        kcpp_exe_path_for_test = init_results["config"]["koboldcpp_executable"]

    print(f"\nDetecting capabilities for: {kcpp_exe_path_for_test}")
    caps = detect_koboldcpp_capabilities(kcpp_exe_path_for_test)
    if "error" in caps:
        print(f"  Error detecting capabilities: {caps['error']}")
    else:
        print(f"  CUDA: {caps.get('cuda')}, ROCm: {caps.get('rocm')}, Vulkan: {caps.get('vulkan')}")
        print(f"  FlashAttn: {caps.get('flash_attn')}, AutoQuantKV: {caps.get('auto_quantkv')}, OverrideTens: {caps.get('overridetensors')}")

    if 'dummy_exe_path_for_test_local' in locals() and os.path.exists(dummy_exe_path_for_test_local):
        os.remove(dummy_exe_path_for_test_local)
        print(f"\nRemoved dummy executable '{dummy_exe_path_for_test_local}'.")

    _cleanup_nvml()

import atexit
atexit.register(_cleanup_nvml)
