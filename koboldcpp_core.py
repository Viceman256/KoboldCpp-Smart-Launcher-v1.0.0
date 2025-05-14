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
import pathlib  # Not directly used but good for path manipulation if needed
import shutil
import platform
from pathlib import Path

def detect_koboldcpp_capabilities(executable_path):
    """Detect available features/flags in KoboldCpp executable"""
    try:
        process = subprocess.run([executable_path, "--help"], 
                               capture_output=True, text=True, check=False, timeout=5)
        
        if process.returncode != 0:
            # Try to get more info from stderr if available
            error_detail = process.stderr.strip() if process.stderr else "Unknown error"
            return {"error": f"Failed to run KoboldCpp with --help. RC={process.returncode}. Detail: {error_detail}", 
                    "cuda": False, "rocm": False, "vulkan": False, "flash_attn": False, 
                    "auto_quantkv": False, "overridetensors": False, "available_args": []} # More pessimistic defaults
        
        output = process.stdout
        
        features = {
            "cuda": "--usecublas" in output,
            "rocm": "--usehipblas" in output or "--userocmblas" in output,
            "vulkan": "--vulkan" in output, # Assuming --vulkan for Vulkan backend
            "flash_attn": "--flashattention" in output,
            "auto_quantkv": "--quantkv auto" in output or ("--quantkv" in output and "auto" in output), # Check for "auto" specifically
            "overridetensors": "--overridetensors" in output,
            "available_args": []
        }
        
        arg_pattern = r'(-{1,2}[\w-]+)' # Simple pattern for flags
        features["available_args"] = re.findall(arg_pattern, output)
        
        return features
    except subprocess.TimeoutExpired:
        return {"error": "Timeout running KoboldCpp with --help", 
                "cuda": False, "rocm": False, "vulkan": False, "flash_attn": False, 
                "auto_quantkv": False, "overridetensors": False, "available_args": []}
    except Exception as e:
        # Default to some common features if help fails, but log the error
        # This fallback is optimistic; ideally, a failed help means we don't know.
        # For broader compatibility, let's be more conservative on error.
        return {"error": str(e), 
                "cuda": False, "rocm": False, "vulkan": False, "flash_attn": False, 
                "auto_quantkv": False, "overridetensors": False, "available_args": []}


# --- Constants and Configuration ---
CONFIG_FILE = "kobold_launcher_config.json"
DEFAULT_CONFIG_TEMPLATE = {
    "koboldcpp_executable": "koboldcpp.exe" if sys.platform == "win32" else "./koboldcpp",
    "default_gguf_dir": "",
    "last_used_gguf_dir": "",
    "db_file": "kobold_launcher_history.db",
    "first_run_completed": False,
    "first_run_intro_shown": False,
    "color_mode": "auto",
    "auto_open_webui": True, 
    "gpu_detection": {
        "nvidia": True,
        "amd": True,
        "intel": True,
        "apple": True # For Apple Silicon
    },
    "model_specific_args": {},
    "default_args": {
        "--threads": "auto",
        "--usecublas": True, # This will be conditional based on detection later
        "--contextsize": "16384",
        "--promptlimit": "16000",
        "--flashattention": True, # Also conditional
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
        "illegal instruction", "clblast error", "opencl error", "rocm error", "hip error" # Added ROCm errors
    ]
}

# --- Runtime Variables & Optional Dependency Management ---
pynvml_available = False
psutil_available = False
pyadlx_available = False # For AMD on Windows via Adrenalin SDK
wmi_available = False    # For Windows Management Instrumentation
pyze_available = False   # For Intel Level Zero
metal_available = False  # For Apple Metal on macOS

try:
    import pynvml
    pynvml.nvmlInit() # Initialize NVML early if available
    pynvml_available = True
except ImportError:
    pynvml_available = False
except pynvml.NVMLError: # Catch init errors too
    pynvml_available = False


try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False

if sys.platform == "win32":
    try:
        import pyadlx # Placeholder, actual import and init might be more complex
        # A more robust check would be to try initializing or getting a GPU list
        # For now, import success is the check.
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
    # Basic check: try to initialize and get driver count
    pyze_api.zeInit(0) 
    num_drivers_ptr = pyze_api.new_uint32_tp()
    pyze_api.zeDriverGet(num_drivers_ptr, None)
    if pyze_api.uint32_tp_value(num_drivers_ptr) > 0:
         pyze_available = True
    # pyze_api.zeDriverGet_post(num_drivers_ptr) # If using SWIG's typical post-call cleanup
    pyze_api.delete_uint32_tp(num_drivers_ptr) # Manual cleanup if needed for SWIG pointers
except ImportError:
    pyze_available = False
except Exception: # Catch pyze init errors (DLL not found, etc.)
    pyze_available = False


if sys.platform == "darwin":
    try:
        import metal 
        if metal.MTLCopyAllDevices(): # Check if any Metal devices are actually found
             metal_available = True
    except ImportError:
        metal_available = False
    except Exception: # Other errors during Metal init
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
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=4)
        return True, f"Configuration saved to {CONFIG_FILE}"
    except Exception as e:
        return False, f"Error saving configuration: {e}"

def load_config():
    config_data = DEFAULT_CONFIG_TEMPLATE.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            for key, default_value in DEFAULT_CONFIG_TEMPLATE.items():
                if key in user_config:
                    if isinstance(default_value, dict) and isinstance(user_config[key], dict):
                        if key in ["default_args", "gpu_detection", "model_specific_args"]: # Keys to deep merge
                            merged_sub_dict = default_value.copy()
                            merged_sub_dict.update(user_config[key])
                            config_data[key] = merged_sub_dict
                        else:
                             config_data[key] = user_config[key]
                    else:
                        config_data[key] = user_config[key]
            
            if "default_args" in config_data:
                for arg_key, arg_default_val in DEFAULT_CONFIG_TEMPLATE["default_args"].items():
                    if arg_key not in config_data["default_args"]:
                        config_data["default_args"][arg_key] = arg_default_val
            else:
                config_data["default_args"] = DEFAULT_CONFIG_TEMPLATE["default_args"].copy()

            return config_data, True, f"Loaded configuration from {CONFIG_FILE}"
        except Exception as e:
            if "default_args" not in config_data:
                 config_data["default_args"] = DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
            else: # Ensure all keys are present even on load error
                for arg_key, arg_default_val in DEFAULT_CONFIG_TEMPLATE["default_args"].items():
                    if arg_key not in config_data["default_args"]:
                        config_data["default_args"][arg_key] = arg_default_val
            return config_data, False, f"Error loading {CONFIG_FILE}: {e}. Using defaults and ensuring all default_args keys."
    else:
        config_data["default_args"] = DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
        return config_data, False, f"No config file found at {CONFIG_FILE}. Using defaults."


# --- Database Functions ---
def init_db(db_file):
    conn = None
    try:
        db_dir = os.path.dirname(db_file)
        if db_dir and not os.path.exists(db_dir):
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
        for col, col_type in cols_to_check.items():
            try:
                cursor.execute(f"SELECT {col} FROM launch_history LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute(f"ALTER TABLE launch_history ADD COLUMN {col} {col_type}")
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
        except sqlite3.IntegrityError:
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
        vram_tolerance_percent = 0.25
        if current_available_dedicated_vram_mb is None or current_available_dedicated_vram_mb <= 0:
            min_hist_vram, max_hist_vram, target_hist_vram_sort = -1, float('inf'), 8192
        else:
            min_hist_vram = current_available_dedicated_vram_mb * (1 - vram_tolerance_percent)
            max_hist_vram = current_available_dedicated_vram_mb * (1 + vram_tolerance_percent)
            target_hist_vram_sort = current_available_dedicated_vram_mb
        model_size_query_val = current_model_analysis.get('size_b')
        model_size_query_for_db = None
        if not (isinstance(model_size_query_val, str) or model_size_query_val is None):
            try: model_size_query_for_db = float(model_size_query_val)
            except (ValueError, TypeError): pass

        query = """
            SELECT kobold_args_json, attempt_level_used, vram_at_launch_decision_mb, launch_outcome, approx_vram_used_kcpp_mb
            FROM launch_history
            WHERE model_filepath = ? AND model_quant_type = ? AND is_moe = ?
              AND (? IS NULL OR ABS(model_size_b - ?) < 0.1)
              AND (vram_at_launch_decision_mb BETWEEN ? AND ? OR vram_at_launch_decision_mb IS NULL)
            ORDER BY
              CASE WHEN launch_outcome LIKE 'SUCCESS_USER_CONFIRMED%' THEN 0
                   WHEN launch_outcome LIKE 'SUCCESS_USER_SAVED_GOOD%' THEN 1
                   WHEN launch_outcome LIKE 'SUCCESS_LOAD_VRAM_OK%' THEN 2
                   WHEN launch_outcome LIKE 'SUCCESS_USER_DIRECT_LAUNCH%' THEN 3
                   ELSE 10 END ASC,
              ABS(COALESCE(vram_at_launch_decision_mb, ?) - ?) ASC,
              attempt_level_used ASC, timestamp DESC LIMIT 1
        """
        cursor.execute(query, (current_model_analysis['filepath'], current_model_analysis.get('quant'),
                               current_model_analysis.get('is_moe', False), model_size_query_for_db,
                               model_size_query_for_db, min_hist_vram, max_hist_vram,
                               target_hist_vram_sort, target_hist_vram_sort))
        row = cursor.fetchone()
        if row:
            try:
                loaded_args = json.loads(row[0])
                return {"args_list": [str(item) for item in loaded_args], "attempt_level": row[1],
                        "historical_vram_mb": row[2], "outcome": row[3], "approx_vram_used_kcpp_mb": row[4]}
            except json.JSONDecodeError: return None
        return None
    except sqlite3.Error: return None
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

# --- System Information and Hardware Detection ---
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


def get_gpu_info_nvidia():
    if not pynvml_available: return None
    try:
        # pynvml.nvmlInit() # Already initialized globally
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
    except pynvml.NVMLError as e: # More specific exception for NVML issues
        return {"success": False, "type": "NVIDIA_ERROR", "message": f"NVIDIA NVML error: {e}"}
    except Exception as e:
        return {"success": False, "type": "NVIDIA_ERROR", "message": f"NVIDIA generic error: {e}"}
    # Global nvmlShutdown will be handled at script exit

def get_gpu_info_amd():
    """Get GPU information for AMD GPUs on Linux (rocm-smi) or Windows (WMI)."""
    if sys.platform == "linux":
        try:
            rocm_check = subprocess.run(["which", "rocm-smi"], capture_output=True, text=True, check=False, timeout=2)
            if rocm_check.returncode != 0: return None # rocm-smi not found
            
            # Try JSON output first (newer rocm-smi)
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
                # Fallback to original regex parsing if JSON fails
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
                return None # Regex also failed
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
                return None # No AMD GPU found
            except Exception as e_wmi:
                return {"success": False, "type": "AMD_WMI_ERROR", "message": f"AMD WMI error: {e_wmi}"}
        # pyadlx could be added here if a reliable way to get free VRAM is found
    return None # Unsupported OS for this specific AMD check

def get_gpu_info_intel():
    """Get GPU information for Intel GPUs using pyze (Level Zero) or WMI (Windows)."""
    if pyze_available:
        try:
            # pyze_api.zeInit(0) # Already done in global check for pyze_available
            num_drivers_ptr = pyze_api.new_uint32_tp()
            pyze_api.zeDriverGet(num_drivers_ptr, None)
            num_drivers = pyze_api.uint32_tp_value(num_drivers_ptr)

            if num_drivers == 0:
                pyze_api.delete_uint32_tp(num_drivers_ptr)
                return None 

            drivers_array = pyze_api.new_ze_driver_handle_t_array(num_drivers)
            pyze_api.zeDriverGet(num_drivers_ptr, drivers_array)
            pyze_api.delete_uint32_tp(num_drivers_ptr)

            for i in range(num_drivers):
                driver = pyze_api.ze_driver_handle_t_array_getitem(drivers_array, i)
                num_devices_ptr = pyze_api.new_uint32_tp()
                pyze_api.zeDeviceGet(driver, num_devices_ptr, None)
                num_devices = pyze_api.uint32_tp_value(num_devices_ptr)

                if num_devices == 0:
                    pyze_api.delete_uint32_tp(num_devices_ptr)
                    continue

                devices_array = pyze_api.new_ze_device_handle_t_array(num_devices)
                pyze_api.zeDeviceGet(driver, num_devices_ptr, devices_array)
                pyze_api.delete_uint32_tp(num_devices_ptr)

                for j in range(num_devices):
                    device = pyze_api.ze_device_handle_t_array_getitem(devices_array, j)
                    props = pyze_api.ze_device_properties_t()
                    # Important: Initialize structure before passing to C (SWIG might handle some, but explicit is safer)
                    pyze_api.memset(pyze_api.addressof(props), 0, pyze_api.sizeof(props))
                    props.stype = pyze_api.ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES # Set stype
                    
                    pyze_api.zeDeviceGetProperties(device, props)

                    if props.type == pyze_api.ZE_DEVICE_TYPE_GPU:
                        gpu_name_bytes = bytes(props.name) # Cast char array to bytes
                        gpu_name = gpu_name_bytes.decode('utf-8', errors='ignore').rstrip('\x00')
                        
                        # Get memory properties for total VRAM
                        mem_props_count_ptr = pyze_api.new_uint32_tp()
                        pyze_api.zeDeviceGetMemoryProperties(device, mem_props_count_ptr, None)
                        mem_props_count = pyze_api.uint32_tp_value(mem_props_count_ptr)
                        total_vram_bytes = 0
                        if mem_props_count > 0:
                            mem_props_array = pyze_api.new_ze_device_memory_properties_t_array(mem_props_count)
                            for k_mem in range(mem_props_count): # Initialize each struct in array
                                mem_prop_item_ptr = pyze_api.ze_device_memory_properties_t_array_getitem_ptr(mem_props_array, k_mem)
                                pyze_api.memset(mem_prop_item_ptr, 0, pyze_api.sizeof(pyze_api.ze_device_memory_properties_t()))
                                pyze_api.ze_device_memory_properties_t_stype_set(mem_prop_item_ptr, pyze_api.ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES)

                            pyze_api.zeDeviceGetMemoryProperties(device, mem_props_count_ptr, mem_props_array)
                            for k_mem in range(mem_props_count):
                                mem_prop_item = pyze_api.ze_device_memory_properties_t_array_getitem(mem_props_array, k_mem)
                                # Summing device-local memory. ZE_DEVICE_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL
                                # This requires checking flags, but for now, sum all `totalSize` for simplicity.
                                # A more accurate way: iterate, check `mem_prop_item.flags` for device local bit.
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
            return None # No Intel GPU found by pyze
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
            return None # No Intel GPU found by WMI
        except Exception as e_wmi:
            return {"success": False, "type": "INTEL_WMI_ERROR", "message": f"Intel WMI error: {e_wmi}"}
    return None

def get_gpu_info_apple_metal():
    """Get GPU information for Apple Silicon/Metal GPUs on macOS."""
    if not (sys.platform == "darwin" and metal_available):
        return None
    try:
        devices = metal.MTLCopyAllDevices()
        if not devices:
            return {"success": False, "message": "Metal: No devices found."}
        
        # Prioritize discrete GPUs if any, otherwise take the first. M-series are integrated.
        # This logic could be more sophisticated if Macs with dGPUs + M-chips become common.
        # For now, assume the first device is representative or the primary integrated one.
        selected_device = devices[0]
        for dev_test in devices: # Simple preference for non-low-power
            if not dev_test.isLowPower():
                selected_device = dev_test
                break
        
        gpu_name = selected_device.name()
        # recommendedMaxWorkingSetSize is total VRAM usable by Metal for this device on unified memory.
        total_mb_approx = selected_device.recommendedMaxWorkingSetSize() / (1024**2) if selected_device.recommendedMaxWorkingSetSize() else 0
        
        # Free unified memory for GPU use is not a simple static value.
        return {
            "success": True, "type": "APPLE_METAL", "name": gpu_name,
            "free_mb": 0, # Cannot easily determine "free" portion of unified memory dedicated to GPU
            "total_mb": total_mb_approx, "used_percent": 0,
            "message": f"Metal {gpu_name}: Approx {total_mb_approx:.0f}MB GPU working set"
        }
    except Exception as e_metal:
        return {"success": False, "type": "APPLE_METAL_ERROR", "message": f"Apple Metal error: {e_metal}"}
                           
def get_available_vram_mb():
    # Try NVIDIA first
    nvidia_info = get_gpu_info_nvidia()
    if nvidia_info and nvidia_info.get("success"):
        return nvidia_info["free_mb"], nvidia_info["total_mb"], nvidia_info["message"], nvidia_info
    
    # Try AMD next
    amd_info = get_gpu_info_amd()
    if amd_info and amd_info.get("success"):
        return amd_info.get("free_mb", 0.0), amd_info.get("total_mb", 0.0), amd_info["message"], amd_info

    # Try Intel 
    intel_info = get_gpu_info_intel()
    if intel_info and intel_info.get("success"):
        return intel_info.get("free_mb", 0.0), intel_info.get("total_mb", 0.0), intel_info["message"], intel_info

    # Try Apple Metal (especially for M-series on macOS)
    if sys.platform == "darwin":
        metal_info = get_gpu_info_apple_metal()
        if metal_info and metal_info.get("success"):
            return metal_info.get("free_mb", 0.0), metal_info.get("total_mb", 0.0), metal_info["message"], metal_info

    sys_ram_info = get_system_info()
    fallback_msg = f"No dedicated GPU detected or info error. System RAM: {sys_ram_info['ram_free_gb']:.1f}/{sys_ram_info['ram_total_gb']:.1f}GB free."
    return 0.0, 0.0, fallback_msg, {"type": "Unknown", "name": "N/A", "free_mb": 0.0, "total_mb": 0.0, "message": fallback_msg, "success": False}

# --- Model Analysis Functions ---
# ... (rest of the file remains the same as in the prompt, from analyze_filename onwards) ...
# ... Make sure to include the __main__ guard for pynvml.nvmlShutdown() if it's added later ...
def analyze_filename(filepath):
    """Analyze a GGUF filename to extract model information."""
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
    if current_ot_string == "FAILURE_MAX_ATTEMPTS": return "MAX ATTEMPTS REACHED"
    moe_max_gpu, moe_super_max_cpu = -25, 10; dense_max_gpu, dense_max_ffn_cpu = -17, 9
    if not current_ot_string:
        if model_analysis.get('is_moe') and attempt_level <= moe_max_gpu: return "MoE MAX GPU: All layers on GPU."
        if not model_analysis.get('is_moe') and attempt_level <= dense_max_gpu: return "Dense MAX GPU: All layers on GPU."
    gpu_layers = get_gpu_layers_for_level(model_analysis, attempt_level); total_layers = model_analysis.get('num_layers', 32)
    layers_info = f" (GPU Layers: {gpu_layers}/{total_layers})"
    if model_analysis.get('is_moe'):
        desc_map = {10: "MoE ULTRA MAX CPU (All FFN)", 8: "MoE SUPER MAX CPU (All FFN + 50% Attn)", 6: "MoE SUPER CPU++ (All FFN + 25% Attn)",
                    4: "MoE SUPER CPU+ (All FFN + 10% Attn)", 2: "MoE SUPER CPU (All FFN + 5% Attn)", 1: "MoE SUPER MAX EXPERT CPU",
                    0: "MoE MAX EXPERT CPU", -2: "MoE CPU++ (All Exp FFN)", -4: "MoE CPU+ (Exp: down,up + 1/2gate)",
                    -6: "MoE CPU/GPU Bal++ (Exp: down,up)", -8: "MoE CPU/GPU Bal+ (Exp: down+1/2up)", -10: "MoE GPU Focus (Exp: all down)",
                    -12: "MoE GPU+ (Exp: 1/2 down)", -15: "MoE GPU++ (Exp: 1/4 down)", -18: "MoE GPU+++ (Exp: 1/8 down)",
                    -21: "MoE GPU++++ (Exp: 1/16 down)", -25: "MoE MAX GPU"}
        closest = max([k for k in desc_map.keys() if k <= attempt_level])
        if attempt_level not in desc_map:
            between_lower = closest; between_upper = min([k for k in desc_map.keys() if k > attempt_level], default=closest)
            if between_lower != between_upper: return f"MoE Level {attempt_level} (between {desc_map[between_lower]} and {desc_map[between_upper]})" + layers_info
        return desc_map.get(closest, f"MoE Custom (Lvl {attempt_level}, OT: {current_ot_string or 'None'})") + layers_info
    else:
        desc_map = {9: "Dense ULTRA MAX CPU (All Layers)", 7: "Dense SUPER MAX CPU (All FFN + 50% Attn)", 5: "Dense SUPER CPU (All FFN + 25% Attn)",
                    3: "Dense SUPER CPU- (All FFN + 10% Attn)", 0: "Dense MAX FFN CPU", -1: "Dense CPU++ (Base FFN: up,down)",
                    -3: "Dense CPU+ (Base FFN: up+1/2down)", -5: "Dense GPU Focus (Base FFN: all up)", -7: "Dense GPU+ (Base FFN: 1/4 up)",
                    -9: "Dense GPU++ (Base FFN: 1/8 up)", -11: "Dense GPU+++ (Base FFN: 1/16 up)", -14: "Dense GPU++++ (Base FFN: 1/32 up)", -17: "Dense MAX GPU"}
        closest = max([k for k in desc_map.keys() if k <= attempt_level])
        if attempt_level not in desc_map:
            between_lower = closest; between_upper = min([k for k in desc_map.keys() if k > attempt_level], default=closest)
            if between_lower != between_upper: return f"Dense Level {attempt_level} (between {desc_map[between_lower]} and {desc_map[between_upper]})" + layers_info
        return desc_map.get(closest, f"Dense Custom (Lvl {attempt_level}, OT: {current_ot_string or 'None'})") + layers_info

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

def build_command(model_path, override_tensor_str, model_analysis, session_base_args_dict):
    current_cmd_args_dict = session_base_args_dict.copy()
    current_cmd_args_dict["--model"] = model_path
    model_analysis = model_analysis if isinstance(model_analysis, dict) else {}

    if "--threads" in current_cmd_args_dict and current_cmd_args_dict["--threads"] == "auto":
        if psutil_available:
            try:
                physical_cores = psutil.cpu_count(logical=False)
                if physical_cores and physical_cores > 0: current_cmd_args_dict["--threads"] = str(max(1, physical_cores - 1 if physical_cores > 1 else 1))
                else: current_cmd_args_dict["--threads"] = str(max(1, (psutil.cpu_count(logical=True) or 2) // 2))
            except: current_cmd_args_dict["--threads"] = "4"
        else: current_cmd_args_dict["--threads"] = "4"

    nblas_val = current_cmd_args_dict.get("--nblas")
    if nblas_val is None or (isinstance(nblas_val, str) and nblas_val.lower() == 'auto'):
        if "--nblas" in current_cmd_args_dict: del current_cmd_args_dict["--nblas"]

    if override_tensor_str and override_tensor_str != "FAILURE_MAX_ATTEMPTS":
        current_cmd_args_dict["--overridetensors"] = override_tensor_str
        attempt_level = get_level_from_overridetensors(override_tensor_str, model_analysis)
        gpulayers_val = current_cmd_args_dict.get("--gpulayers")
        if gpulayers_val is None or (isinstance(gpulayers_val, str) and gpulayers_val.lower() in ['auto', '999']):
            current_cmd_args_dict["--gpulayers"] = str(get_gpu_layers_for_level(model_analysis, attempt_level))
    else:
        if "--overridetensors" in current_cmd_args_dict: del current_cmd_args_dict["--overridetensors"]
        if current_cmd_args_dict.get("--gpulayers") == "999" and session_base_args_dict.get("--gpulayers") != "999":
            original_gl = session_base_args_dict.get("--gpulayers")
            if original_gl is not None and str(original_gl).lower() not in ['off', '0', 'auto']: current_cmd_args_dict["--gpulayers"] = str(original_gl)
            elif original_gl is None or str(original_gl).lower() == 'auto':
                if "--gpulayers" in current_cmd_args_dict: current_cmd_args_dict["--gpulayers"] = "auto"
            elif str(original_gl).lower() in ['off', '0']: current_cmd_args_dict["--gpulayers"] = str(original_gl)

    gpulayers_val = current_cmd_args_dict.get("--gpulayers")
    if isinstance(gpulayers_val, str) and gpulayers_val.lower() in ['off', '0']:
        if "--gpulayers" in current_cmd_args_dict: del current_cmd_args_dict["--gpulayers"]
        current_cmd_args_dict["--nogpulayers"] = True
    elif "--nogpulayers" in current_cmd_args_dict:
        if not (isinstance(gpulayers_val, str) and gpulayers_val.lower() in ['off', '0']):
            del current_cmd_args_dict["--nogpulayers"]

    quantkv_val = current_cmd_args_dict.get("--quantkv")
    if quantkv_val is None or (isinstance(quantkv_val, str) and quantkv_val.lower() == 'auto'):
        if "--quantkv" in current_cmd_args_dict: del current_cmd_args_dict["--quantkv"]
        quant_upper = model_analysis.get('quant', 'unknown').upper(); model_size_b = model_analysis.get('size_b', 0)
        if any(q_check in quant_upper for q_check in ['Q5', 'Q6', 'Q8', 'F16', 'BF16', 'K_M', 'K_L', 'K_XL']) or \
           'XL' in quant_upper or (isinstance(model_size_b, (int, float)) and model_size_b >= 30):
            current_cmd_args_dict["--quantkv"] = "1"
    elif isinstance(quantkv_val, str) and quantkv_val.lower() == 'off':
        if "--quantkv" in current_cmd_args_dict: del current_cmd_args_dict["--quantkv"]

    blasbatchsize_val = current_cmd_args_dict.get("--blasbatchsize")
    if blasbatchsize_val is None or (isinstance(blasbatchsize_val, str) and blasbatchsize_val.lower() == 'auto'):
        if "--blasbatchsize" in current_cmd_args_dict: del current_cmd_args_dict["--blasbatchsize"]
        model_size_b = model_analysis.get('size_b', 0)
        if model_analysis.get('is_moe', False): current_cmd_args_dict["--blasbatchsize"] = "128"
        elif isinstance(model_size_b, (int, float)) and model_size_b > 20: current_cmd_args_dict["--blasbatchsize"] = "256"
        else: current_cmd_args_dict["--blasbatchsize"] = "512"
    elif isinstance(blasbatchsize_val, str) and blasbatchsize_val.lower() == 'off':
        if "--blasbatchsize" in current_cmd_args_dict: del current_cmd_args_dict["--blasbatchsize"]

    boolean_flags_to_manage = ["--usecublas", "--flashattention", "--nommap", "--lowvram"]
    for flag in boolean_flags_to_manage:
        if flag in current_cmd_args_dict and current_cmd_args_dict[flag] is False:
            del current_cmd_args_dict[flag]
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
    config, config_loaded, config_message = load_config()
    db_file = config.get("db_file", DEFAULT_CONFIG_TEMPLATE["db_file"])
    db_success, db_message = init_db(db_file)
    if psutil_available and config["default_args"].get("--threads") == "auto":
        try:
            physical_cores = psutil.cpu_count(logical=False)
            if physical_cores and physical_cores > 0: config["default_args"]["--threads"] = str(max(1, physical_cores - 1 if physical_cores > 1 else 1))
            else: config["default_args"]["--threads"] = str(max(1, (psutil.cpu_count(logical=True) or 2) // 2))
        except Exception: config["default_args"]["--threads"] = "4"
    elif config["default_args"].get("--threads") == "auto": config["default_args"]["--threads"] = "4"
    
    # Determine best GPU for conditional flags based on detection
    _, _, _, gpu_details = get_available_vram_mb()
    if gpu_details and gpu_details.get("type") == "NVIDIA":
        config["default_args"]["--usecublas"] = True # Prefer True if NVIDIA detected
    else: # If not NVIDIA, or detection failed, cublas is likely not applicable
        config["default_args"]["--usecublas"] = False
        # Could also set other flags like --usehipblas if ROCm AMD detected etc.
        # if gpu_details and gpu_details.get("type") == "AMD" and sys.platform == "linux":
        #     config["default_args"]["--usehipblas"] = True # Or similar flag if KCPP supports it

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
    # This block is for testing the core library directly.
    # It won't run when imported by launcher.py or launcher_gui.py.
    
    print("KoboldCpp Core Library Test")
    init_results = initialize_launcher()
    print(f"Initialization successful: {init_results['initialized']}")
    print(f"Config loaded: {init_results['config_loaded']} ({init_results['config_message']})")
    print(f"DB status: {init_results['db_success']} ({init_results['db_message']})")
    
    print("\nSystem Info:")
    for k, v in init_results['system_info'].items():
        print(f"  {k}: {v}")
        
    print("\nGPU Info:")
    # gpu_free, gpu_total, gpu_msg, gpu_dict = get_available_vram_mb()
    gpu_dict = init_results['gpu_info']
    if gpu_dict and gpu_dict.get("success"):
        print(f"  Type: {gpu_dict.get('type')}")
        print(f"  Name: {gpu_dict.get('name')}")
        print(f"  Total VRAM: {gpu_dict.get('total_mb'):.0f} MB")
        print(f"  Free VRAM: {gpu_dict.get('free_mb'):.0f} MB")
        print(f"  Message: {gpu_dict.get('message')}")
    else:
        print(f"  Message: {gpu_dict.get('message', 'GPU detection failed or no GPU found.')}")

    # Test KoboldCpp capabilities detection (requires a dummy or real koboldcpp executable)
    # Create a dummy executable for testing if one doesn't exist
    dummy_exe_name = "dummy_koboldcpp.bat" if sys.platform == "win32" else "dummy_koboldcpp.sh"
    if not os.path.exists(init_results["config"]["koboldcpp_executable"]):
        print(f"\nKoboldCpp executable '{init_results['config']['koboldcpp_executable']}' not found.")
        print(f"Creating a dummy executable '{dummy_exe_name}' for capability detection test.")
        if sys.platform == "win32":
            with open(dummy_exe_name, "w") as f:
                f.write("@echo off\n")
                f.write("echo KoboldCpp Dummy Help\n")
                f.write("echo --usecublas\n")
                f.write("echo --usehipblas\n")
                f.write("echo --vulkan\n")
                f.write("echo --flashattention\n")
                f.write("echo --quantkv auto\n")
                f.write("echo --overridetensors\n")
                f.write("echo --model <path>\n")
                f.write("echo --port <port>\n")
        else:
            with open(dummy_exe_name, "w") as f:
                f.write("#!/bin/sh\n")
                f.write("echo 'KoboldCpp Dummy Help'\n")
                f.write("echo '--usecublas'\n")
                f.write("echo '--usehipblas'\n")
                f.write("echo '--vulkan'\n")
                f.write("echo '--flashattention'\n")
                f.write("echo '--quantkv auto'\n")
                f.write("echo '--overridetensors'\n")
                f.write("echo '--model <path>'\n")
                f.write("echo '--port <port>'\n")
            os.chmod(dummy_exe_name, 0o755)
        kcpp_exe_path_for_test = os.path.abspath(dummy_exe_name)
    else:
        kcpp_exe_path_for_test = init_results["config"]["koboldcpp_executable"]

    print(f"\nDetecting capabilities for: {kcpp_exe_path_for_test}")
    caps = detect_koboldcpp_capabilities(kcpp_exe_path_for_test)
    if "error" in caps:
        print(f"  Error detecting capabilities: {caps['error']}")
    else:
        print(f"  CUDA: {caps.get('cuda')}")
        print(f"  ROCm: {caps.get('rocm')}")
        print(f"  Vulkan: {caps.get('vulkan')}")
        print(f"  FlashAttention: {caps.get('flash_attn')}")
        print(f"  AutoQuantKV: {caps.get('auto_quantkv')}")
        print(f"  OverrideTensors: {caps.get('overridetensors')}")
        # print(f"  Available Args: {caps.get('available_args')}") # Can be very long

    # Clean up dummy executable if created
    if kcpp_exe_path_for_test.endswith(dummy_exe_name) and os.path.exists(dummy_exe_name):
        os.remove(dummy_exe_name)
        print(f"\nRemoved dummy executable '{dummy_exe_name}'.")

    # Ensure NVML is shut down if it was initialized
    _cleanup_nvml()

# Ensure NVML is shut down when the script/module is unloaded if it was initialized
# This is more for when the core itself might be run standalone or as part of atexit.
import atexit
atexit.register(_cleanup_nvml)
