#!/usr/bin/env python3
"""
TensorTune - Installation Script (v1.1.1 Helper)
This script helps set up TensorTune by checking Python, creating a
default requirements.txt if needed, installing dependencies,
and providing guidance for optional components.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
ENDC = "\033[0m"
BOLD = "\033[1m"

CURRENT_TENSORTUNE_VERSION = "1.1.1"

def print_header(text):
    """Print a formatted header"""
    print(f"\n{BLUE}{BOLD}=== {text} ==={ENDC}\n")

def print_success(text):
    """Print a success message"""
    print(f"{GREEN}✓ {text}{ENDC}")

def print_warning(text):
    """Print a warning message"""
    print(f"{YELLOW}! {text}{ENDC}")

def print_error(text):
    """Print an error message"""
    print(f"{RED}✗ {text}{ENDC}")

def print_info(text):
    """Print an info message"""
    print(f"{BLUE}ℹ {text}{ENDC}")

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Step 1: Checking Python Version")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8): 
        print_error(f"Python 3.8+ is strongly recommended for TensorTune. Found: Python {version.major}.{version.minor}.{version.micro}")
        if version.minor < 7:
            print_error("This Python version is too old and unsupported for core dependencies.")
            return False
        print_warning("You may encounter issues with Python 3.7, particularly with the GUI. Please consider upgrading Python.")
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected.")
    return True

def generate_requirements_file_if_needed():
    """Generates requirements.txt if it doesn't exist."""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_warning(f"{requirements_file} not found. Creating a default one for TensorTune v{CURRENT_TENSORTUNE_VERSION}...")
        default_req_content = f"""# TensorTune v{CURRENT_TENSORTUNE_VERSION} Requirements

# Python 3.8+ is recommended for best compatibility.

# Core GUI dependency
customtkinter>=5.2.0

# --- Optional but highly recommended for full functionality ---
psutil>=5.9.0     # System information and process management
pynvml>=11.5.0    # NVIDIA GPU VRAM monitoring (for NVIDIA GPUs)
rich>=13.4.0      # Improved CLI interface
appdirs>=1.4.4    # For consistent user-specific config/data paths

# --- Platform-Specific / Advanced GPU Info (Install manually or via pip if desired) ---
# For Windows systems (GPU info fallback, general system info)
WMI>=1.5.1 ; sys_platform == 'win32'

# For Intel Arc/Xe GPUs (detailed info, requires drivers & Level Zero runtime)
# pyze-l0>=0.1.0 # Uncomment to have pip attempt to install; see PYZE_SETUP_GUIDE.md

# --- Not pip-installable directly, require manual setup ---
# PyADLX: For detailed AMD GPU info on Windows. Requires manual C++ build.
#         See PYADLX_SETUP_GUIDE.md for details.

# --- Notes ---
# tkinter: Standard library, usually included with Python.
# On Linux, if missing, install via system package manager (e.g., python3-tk).
"""
        try:
            with open(requirements_file, "w", encoding="utf-8") as f:
                f.write(default_req_content)
            print_success(f"Default {requirements_file} created.")
        except IOError as e:
            print_error(f"Could not create {requirements_file}: {e}")
            return False
    else:
        print_info(f"{requirements_file} already exists.")
    return True


def install_dependencies_from_file(create_venv=False):
    """Install dependencies from requirements.txt"""
    header_text = "Step 2: Installing Dependencies"
    if create_venv:
        header_text += " (in Virtual Environment)"
    else:
        header_text += " (in Current Environment)"
    print_header(header_text)
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_error(f"{requirements_file} not found. Cannot install dependencies.")
        print_info("Please ensure requirements.txt is present or run this script again to generate a default.")
        return False

    pip_executable = sys.executable
    if platform.system() == "Windows" and "pythonw.exe" in pip_executable.lower():
        candidate_python_exe = pip_executable.lower().replace("pythonw.exe", "python.exe")
        if Path(candidate_python_exe).exists():
            pip_executable = candidate_python_exe
    
    pip_cmd = [pip_executable, "-m", "pip", "install", "-r", requirements_file]

    if create_venv:
        venv_name = "tensortune_env"
        if not Path(venv_name).exists():
            print_info(f"Creating virtual environment '{venv_name}'...")
            try:
                subprocess.run([sys.executable, "-m", "venv", venv_name], check=True, capture_output=True, text=True, encoding="utf-8")
                print_success(f"Virtual environment '{venv_name}' created successfully.")
            except subprocess.CalledProcessError as e:
                print_error(f"Failed to create virtual environment: {e.stderr or e.stdout}")
                return False

            if platform.system() == "Windows":
                pip_executable_venv = str(Path(venv_name) / "Scripts" / "python.exe")
                activate_instruction = f"  {venv_name}\\Scripts\\activate"
            else:
                pip_executable_venv = str(Path(venv_name) / "bin" / "python")
                activate_instruction = f"  source {venv_name}/bin/activate"
            
            pip_cmd_venv = [pip_executable_venv, "-m", "pip", "install", "-r", requirements_file]
            print_info(f"Installing dependencies into '{venv_name}' (using its pip)...")
            result = subprocess.run(pip_cmd_venv, capture_output=True, text=True, encoding="utf-8")

            if result.returncode == 0:
                print_success("Dependencies installed successfully into virtual environment.")
                print_info(f"\nTo use TensorTune from this virtual environment, first activate it:")
                print(activate_instruction)
                print(f"Then, from this directory, run TensorTune (e.g., python tensortune_gui.py).")
                return True
            else:
                print_error(f"Failed to install dependencies into virtual environment (see output below).")
                print(f"{YELLOW}Pip Command Used: {' '.join(pip_cmd_venv)}{ENDC}")
                print(f"{YELLOW}Pip Output:\n{result.stdout}\n{result.stderr}{ENDC}")
                print_warning(f"Please activate the venv ('{activate_instruction}') and try running: pip install -r {requirements_file}")
                return False
        else:
            print_info(f"Virtual environment '{venv_name}' already exists. Skipping creation.")
            print_warning("If you want to (re)install in this venv, please activate it and run:")
            print_warning(f"  pip install -r {requirements_file}")
            print_warning("Or, delete the '{venv_name}' folder and re-run this script.")
            return True
    else:
        print_info(f"Installing dependencies into the current Python environment: {sys.executable}")
        if not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
            print_warning("Consider using a virtual environment for better project isolation (run this script again and choose option 2).")
        
        result = subprocess.run(pip_cmd, capture_output=True, text=True, encoding="utf-8")
        
        if result.returncode == 0:
            print_success("Dependencies installed successfully.")
            return True
        else:
            print_error(f"Failed to install dependencies (see output below).")
            print(f"{YELLOW}Pip Command Used: {' '.join(pip_cmd)}{ENDC}")
            print(f"{YELLOW}Pip Output:\n{result.stdout}\n{result.stderr}{ENDC}")
            return False

def check_and_advise_optional_components():
    """Checks for optional libraries TensorTune can use and advises."""
    print_header("Step 3: Checking Optional Components & Advising")
    
    pip_optionals = [
        ("psutil", "system/process info (CPU threads, process kill)", "psutil"),
        ("pynvml", "NVIDIA GPU VRAM monitoring (for NVIDIA GPUs)", "pynvml"),
        ("rich", "enhanced CLI experience", "rich"),
        ("appdirs", "consistent config/data paths", "appdirs"),
        ("tkinter", "GUI operation (usually included with Python)", None),
        ("WMI", "Windows Management Instrumentation (Windows AMD/fallback GPU info) - Windows Only", "WMI")
    ]

    special_optionals = [
        ("pyadlx", "AMD ADLX for detailed AMD GPU info (Windows)", None),
        ("pyze.api", "Intel Level Zero (pyze-l0) for Intel Arc GPU info", "pyze-l0")
    ]

    all_good_flag = True
    print_info("Checking common optional libraries:")
    for import_name, purpose, pip_name in pip_optionals:
        if import_name == "WMI" and platform.system() != "Windows":
            continue 
        try:
            __import__(import_name)
            print_success(f"{import_name} is available ({purpose}).")
        except ImportError:
            if import_name == "tkinter":
                print_error(f"{import_name} is NOT available but REQUIRED for the GUI ({purpose}).")
                if platform.system() == "Linux":
                    print_info("  On Linux, try: sudo apt-get install python3-tk  OR  sudo dnf install python3-tkinter")
                else:
                    print_info("  On Windows/macOS, tkinter is usually included with Python. Try reinstalling Python from python.org, ensuring 'tcl/tk and IDLE' is selected.")
                all_good_flag = False
            else:
                print_warning(f"{import_name} not found. Functionality for '{purpose}' may be limited or use fallbacks.")
                if pip_name:
                    print_info(f"  Consider installing: pip install {pip_name}")
    
    print_info("\nChecking specialized optional libraries (may require manual setup):")
    for import_name, purpose, setup_hint_or_pip in special_optionals:
        try:
            __import__(import_name)
            actual_display_name = import_name.split('.')[0]
            print_success(f"{actual_display_name} seems to be available ({purpose}).")
        except ImportError:
            actual_display_name = import_name.split('.')[0]
            print_warning(f"{actual_display_name} not found. TensorTune will have limited info for '{purpose}'.")
            if import_name == "pyadlx":
                print_info(f"  PyADLX requires manual building with AMD's ADLX SDK. See {BOLD}PYADLX_SETUP_GUIDE.md{ENDC} for details.")
            elif import_name == "pyze.api":
                print_info(f"  PyZE (package '{setup_hint_or_pip}') is for Intel Arc/Xe GPUs. Try 'pip install {setup_hint_or_pip}'.")
                print_info(f"  Requires Intel drivers and Level Zero runtime. See {BOLD}PYZE_SETUP_GUIDE.md{ENDC} for details.")
    
    return all_good_flag

def find_koboldcpp_location():
    print_header("Step 4: Locating KoboldCpp (Informational)")
    common_names = ["koboldcpp.exe", "koboldcpp", "koboldcpp.py"]
    if platform.system() != "Windows": common_names.append("./koboldcpp")
    
    for name in common_names:
        # Check in current directory first
        if Path(name).is_file(): # Simplified check
            print_success(f"Found potential KoboldCpp in current directory: {Path(name).resolve()}")
            return
    for name in common_names:
        # Then check in PATH
        found_in_path = shutil.which(name)
        if found_in_path:
            print_success(f"Found potential KoboldCpp in PATH: {Path(found_in_path).resolve()}")
            return
            
    print_warning("KoboldCpp executable/script not automatically found.")
    print_info("  This is okay! You will set the correct path in TensorTune's settings during first run or later.")

def check_tensortune_files():
    print_header("Step 5: Verifying TensorTune Files")
    required = ["tensortune_core.py", "tensortune_cli.py", "tensortune_gui.py"]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        print_error(f"Essential TensorTune script(s) missing: {', '.join(missing)}")
        print_info("Please ensure all scripts are in the same directory as this installer.")
        return False
    print_success("Essential TensorTune script files are present.")
    return True

def create_launch_scripts_if_wanted():
    print_header(f"Step 6: Create Convenience Launch Scripts (Optional) for TensorTune v{CURRENT_TENSORTUNE_VERSION}")
    if not (Path("tensortune_cli.py").exists() and Path("tensortune_gui.py").exists()):
        print_warning("Core GUI/CLI files missing, cannot create launch scripts.")
        return

    python_exe_path_str = sys.executable.replace("pythonw.exe", "python.exe") # Ensure it's python.exe

    scripts_to_create = {
        "launch_tensortune_cli": ("tensortune_cli.py", f"# Launches TensorTune CLI v{CURRENT_TENSORTUNE_VERSION}"),
        "launch_tensortune_gui": ("tensortune_gui.py", f"# Launches TensorTune GUI v{CURRENT_TENSORTUNE_VERSION}")
    }

    for base_name, (target_script, header_comment) in scripts_to_create.items():
        if platform.system() == "Windows":
            script_filename = f"{base_name}.bat"
            # For .bat, %~dp0 resolves to the directory of the .bat script itself
            content = f'@echo off\nrem {header_comment.lstrip("#").strip()}\n"{python_exe_path_str}" "%~dp0{target_script}" %*\nif "%1"=="-nc" (exit /b)\nif not "%1"=="-nc" if not defined TensortuneNoPause (pause)'
            if "gui" in base_name: 
                 content = f'@echo off\nrem {header_comment.lstrip("#").strip()}\nstart "TensorTune GUI" /B "{python_exe_path_str}" "%~dp0{target_script}" %*'
        else: # Linux/macOS
            script_filename = f"{base_name}.sh"
            content = f"""#!/bin/sh
{header_comment}
# Get the directory where the script is located
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"{python_exe_path_str}" "$DIR/{target_script}" "$@"
"""
        try:
            with open(script_filename, "w", newline='\n', encoding="utf-8") as f: f.write(content)
            if platform.system() != "Windows": os.chmod(script_filename, 0o755) # Make executable
            print_success(f"Created launch script: {script_filename}")
        except IOError as e:
            print_error(f"Could not create {script_filename}: {e}")

def confirm_yes_no(prompt_text, default_yes=True):
    suffix = " [Y/n]" if default_yes else " [y/N]"
    while True:
        choice = input(f"{prompt_text} {suffix}: ").strip().lower()
        if not choice: return default_yes
        if choice in ['y', 'yes']: return True
        if choice in ['n', 'no']: return False
        print_warning("Please answer 'yes' or 'no'.")

def main():
    print(f"\n{BOLD}{BLUE}TensorTune v{CURRENT_TENSORTUNE_VERSION} - Installation Helper{ENDC}\n")
    
    if not check_python_version(): sys.exit(1)
    if not check_tensortune_files(): sys.exit(1)
    if not generate_requirements_file_if_needed():
        print_warning("Proceeding without requirements.txt. Dependency installation might fail or use outdated versions.")

    print_header("Dependency Installation Choice")
    print("1. Install into current Python environment (simple, if not using many projects).")
    print("2. Create 'tensortune_env' virtual environment and install (recommended for isolation).")
    print("3. Skip automatic dependency installation (you'll need to manually run: pip install -r requirements.txt).")
    
    dep_choice = ""
    while dep_choice not in ["1", "2", "3"]:
        dep_choice = input("Choice (1-3) [default: 1, or 2 if no venv detected]: ").strip()
        if not dep_choice: # Default logic
            if not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
                dep_choice = "2" # Default to venv if not in one
            else:
                dep_choice = "1" # Default to current if already in venv or user prefers

    deps_ok = False
    if dep_choice == "1": deps_ok = install_dependencies_from_file(create_venv=False)
    elif dep_choice == "2":
        deps_ok = install_dependencies_from_file(create_venv=True)
        if Path("tensortune_env").exists() and deps_ok: # Only show this if venv was targeted and install was ok
            print_info(f"\n{BOLD}IMPORTANT: Virtual environment 'tensortune_env' was used/created.{ENDC}")
            print_info("You MUST activate it before running TensorTune from this directory.")
            act_cmd = ".\\tensortune_env\\Scripts\\activate" if platform.system() == "Windows" else "source ./tensortune_env/bin/activate"
            print_info(f"  To activate (from this directory): {BOLD}{act_cmd}{ENDC}")
            print_info("  Once activated, you can run TensorTune, e.g., 'python tensortune_gui.py'")
            print_header("Setup for Virtual Environment Complete");
            print_success("To proceed, activate the virtual environment as instructed above."); sys.exit(0)
    elif dep_choice == "3":
        print_info("Skipping automatic dependency installation."); 
        print_info(f"Please ensure you manually install packages from {BOLD}requirements.txt{ENDC} if needed."); deps_ok = True # Assume user will handle it

    if not deps_ok and dep_choice != "3":
        print_error("Dependency installation encountered issues. Please review the messages above.")
        print_info(f"You may need to install dependencies manually using: pip install -r requirements.txt")
    
    check_and_advise_optional_components()
    find_koboldcpp_location()
    
    if confirm_yes_no("\nCreate convenience launch scripts (e.g., launch_tensortune_gui.bat/.sh)?", default_yes=True):
        create_launch_scripts_if_wanted()
    
    print_header("Setup Summary")
    print(f"{GREEN}{BOLD}TensorTune v{CURRENT_TENSORTUNE_VERSION} setup helper finished.{ENDC}")
    if dep_choice != "2": # If not using venv path (which exits earlier)
        if deps_ok or dep_choice == "3":
            print_info("You should now be able to run TensorTune.")
            print_info("Use created launch scripts (if any) or run directly: python tensortune_gui.py / tensortune_cli.py")
            print_info("On first launch, TensorTune will guide you if KoboldCpp path needs to be set.")
        else:
            print_error("Some steps had issues. Please review the output above carefully.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt: print_error("\nInstallation cancelled by user."); sys.exit(1)
    except Exception as e:
        print_error(f"\nAn unexpected error occurred during installation: {e}"); 
        import traceback; traceback.print_exc(); sys.exit(1)