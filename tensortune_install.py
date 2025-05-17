#!/usr/bin/env python3
"""
TensorTune - Installation Script (v1.1.0 Helper)
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

CURRENT_TENSORTUNE_VERSION = "1.1.0" # For display purposes in this script

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
        print_error(f"Python 3.8+ is highly recommended for TensorTune (especially GUI). Found: Python {version.major}.{version.minor}.{version.micro}")
        if version.minor < 7: # Python 3.6 and below are definitively too old for some dependencies
            print_error("This Python version is too old and unsupported for core dependencies.")
            return False
        print_warning("You may encounter issues with older Python 3.7, particularly with the GUI. Consider upgrading Python.")
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected.")
    return True

def generate_requirements_file_if_needed():
    """Generates requirements.txt if it doesn't exist."""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_warning(f"{requirements_file} not found. Creating a default one for TensorTune v{CURRENT_TENSORTUNE_VERSION}...")
        default_req_content = f"""# TensorTune v{CURRENT_TENSORTUNE_VERSION} Requirements

# Core GUI dependency
customtkinter>=5.2.0

# --- Optional but highly recommended for full functionality ---
psutil>=5.9.0
pynvml>=11.5.0
rich>=13.4.0
appdirs>=1.4.4

# --- Platform-Specific for Advanced GPU Info (Install manually if needed) ---
# Windows Management Instrumentation (Windows fallback GPU info)
# Consider: pip install WMI
wmi>=1.5.1

# Intel Level Zero (Intel GPU Info) - may require driver/runtime setup
# Consider: pip install pyze-l0
# pyze-l0>=0.1.0

# AMD ADLX (Windows AMD GPU Info) - requires manual build, see README/AMD docs
# Not a direct pip install.

# --- Notes ---
# tkinter: Standard library, install via system package manager if missing (e.g., python3-tk).
"""
        try:
            with open(requirements_file, "w") as f:
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

    # Ensure we use 'python.exe' for pip on Windows, not 'pythonw.exe'
    pip_executable = sys.executable
    if platform.system() == "Windows" and "pythonw.exe" in pip_executable.lower():
        pip_executable = pip_executable.lower().replace("pythonw.exe", "python.exe")
        if not Path(pip_executable).exists(): # Fallback if python.exe isn't where pythonw.exe is
            pip_executable = sys.executable # Revert to original sys.executable if replacement not found

    pip_cmd = [pip_executable, "-m", "pip", "install", "-r", requirements_file]

    if create_venv:
        venv_name = "tensortune_env"
        if not Path(venv_name).exists():
            print_info(f"Creating virtual environment '{venv_name}'...")
            try:
                # Use sys.executable (which should be python.exe path if resolved) to create venv
                subprocess.run([sys.executable, "-m", "venv", venv_name], check=True, capture_output=True, text=True)
                print_success(f"Virtual environment '{venv_name}' created successfully.")
            except subprocess.CalledProcessError as e:
                print_error(f"Failed to create virtual environment: {e.stderr}")
                return False # Critical failure if venv creation fails when requested

            if platform.system() == "Windows":
                pip_executable_venv = str(Path(venv_name) / "Scripts" / "python.exe")
                activate_instruction = f"  {venv_name}\\Scripts\\activate"
            else:
                pip_executable_venv = str(Path(venv_name) / "bin" / "python")
                activate_instruction = f"  source {venv_name}/bin/activate"
            
            # Command to install into the venv's Python
            pip_cmd_venv = [pip_executable_venv, "-m", "pip", "install", "-r", requirements_file]
            print_info(f"Installing dependencies into '{venv_name}' (using its pip)...")
            result = subprocess.run(pip_cmd_venv, capture_output=True, text=True)

            if result.returncode == 0:
                print_success("Dependencies installed successfully into virtual environment.")
                print_info(f"\nTo use TensorTune from this virtual environment, first activate it:")
                print(activate_instruction)
                print(f"Then, from this directory, run TensorTune (e.g., python tensortune_gui.py).")
                return True # Dependencies installed in venv
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
            return True # Indicate user should manage existing venv
    else: # Install in current environment
        print_info(f"Installing dependencies into the current Python environment: {sys.executable}")
        # Check if we are already in a venv
        if not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
            print_warning("Consider using a virtual environment for better project isolation (run this script again and choose option 2).")
        
        result = subprocess.run(pip_cmd, capture_output=True, text=True)
        
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
        ("appdirs", "consistent config/data paths", "appdirs"), # Added appdirs
        ("tkinter", "GUI operation (usually included with Python)", None),
        ("WMI", "Windows Management Instrumentation (Windows AMD/fallback GPU info) - Windows Only", "WMI") # Changed import name to WMI for direct check
    ]

    special_optionals = [
        ("pyadlx", "AMD ADLX for detailed AMD GPU info (Windows, requires manual build/AMD SDK)", None),
        ("pyze.api", "Intel Level Zero (pyze-l0) for Intel Arc GPU info", "pyze-l0")
    ]

    all_good_flag = True
    print_info("Checking common optional libraries:")
    for import_name, purpose, pip_name in pip_optionals:
        if import_name == "WMI" and platform.system() != "Windows": # Corrected import name for check
            continue 
        try:
            __import__(import_name) # Use the actual import name Python would use
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
            __import__(import_name) # e.g., pyadlx or pyze.api
            actual_display_name = import_name.split('.')[0] # Get 'pyze' from 'pyze.api'
            print_success(f"{actual_display_name} seems to be available ({purpose}).")
        except ImportError:
            actual_display_name = import_name.split('.')[0]
            print_warning(f"{actual_display_name} not found. TensorTune will have limited info for '{purpose}'.")
            if import_name == "pyadlx":
                print_info("  PyADLX is for AMD GPUs on Windows and requires manual building with AMD's ADLX SDK.")
                print_info(f"  TensorTune will use WMI as a fallback. See {BOLD}PYADLX_SETUP_GUIDE.md{ENDC} for details.")
            elif import_name == "pyze.api":
                print_info(f"  PyZE (usually package '{setup_hint_or_pip}') is for Intel Arc/Xe GPUs. Try 'pip install {setup_hint_or_pip}'.")
                print_info(f"  Requires Intel drivers and Level Zero runtime. See {BOLD}PYZE_SETUP_GUIDE.md{ENDC} for details.")
    
    return all_good_flag

def find_koboldcpp_location():
    print_header("Step 4: Locating KoboldCpp (Informational)")
    common_names = ["koboldcpp.exe", "koboldcpp", "koboldcpp.py"]
    if platform.system() != "Windows": common_names.append("./koboldcpp")
    
    for name in common_names:
        if Path(name).resolve().is_file():
            print_success(f"Found potential KoboldCpp in current directory: {Path(name).resolve()}")
            return
    for name in common_names:
        found = shutil.which(name)
        if found:
            print_success(f"Found potential KoboldCpp in PATH: {Path(found).resolve()}")
            return
            
    print_warning("KoboldCpp executable/script not automatically found.")
    print_info("  This is okay! You'll set the correct path in TensorTune's settings.")

def check_tensortune_files():
    print_header("Step 5: Verifying TensorTune Files")
    required = ["tensortune_core.py", "tensortune_cli.py", "tensortune_gui.py"]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        print_error(f"Essential TensorTune script(s) missing: {', '.join(missing)}")
        return False
    print_success("Essential TensorTune script files are present.")
    return True

def create_launch_scripts_if_wanted():
    print_header(f"Step 6: Create Convenience Launch Scripts (Optional) for TensorTune v{CURRENT_TENSORTUNE_VERSION}") # Add version
    # ... (rest of the function as before, ensuring CURRENT_TENSORTUNE_VERSION is used in comments) ...
    # Ensure the target_script paths are correct if you renamed files, e.g. if you have TT_CLI.py vs tensortune_cli.py
    python_exe = sys.executable.replace("pythonw.exe", "python.exe")

    scripts_to_create = {
        "launch_tensortune_cli": ("tensortune_cli.py", f"# Launches TensorTune CLI v{CURRENT_TENSORTUNE_VERSION}"),
        "launch_tensortune_gui": ("tensortune_gui.py", f"# Launches TensorTune GUI v{CURRENT_TENSORTUNE_VERSION}")
    }

    for base_name, (target_script, header_comment) in scripts_to_create.items():
        if platform.system() == "Windows":
            script_filename = f"{base_name}.bat"
            content = f'@echo off\nrem {header_comment.lstrip("#").strip()}\n"{python_exe}" "%~dp0{target_script}" %*\nif "%1"=="-nc" (exit /b)\npause'
            if "gui" in base_name: # No pause for GUI by default
                 content = f'@echo off\nrem {header_comment.lstrip("#").strip()}\n"{python_exe}" "%~dp0{target_script}" %*'
        else:
            script_filename = f"{base_name}.sh"
            content = f"""#!/bin/sh
{header_comment}
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"{python_exe}" "$DIR/{target_script}" "$@"
"""
        try:
            with open(script_filename, "w", newline='\n') as f: f.write(content)
            if platform.system() != "Windows": os.chmod(script_filename, 0o755)
            print_success(f"Created launch script: {script_filename}")
        except IOError as e:
            print_error(f"Could not create {script_filename}: {e}")

def confirm_yes_no(prompt_text, default_yes=True):
    suffix = "[Y/n]" if default_yes else "[y/N]"
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
        print_warning("Proceeding without requirements.txt. Dependency installation might fail.")

    print_header("Dependency Installation Choice")
    print("1. Install into current Python environment (simple, if not using many projects).")
    print("2. Create 'tensortune_env' virtual environment and install (recommended for isolation).")
    print("3. Skip automatic dependency installation (manual install from requirements.txt needed).")
    
    dep_choice = ""
    while dep_choice not in ["1", "2", "3"]:
        dep_choice = input("Choice (1-3) [default: 1]: ").strip() or "1"

    deps_ok = False
    if dep_choice == "1": deps_ok = install_dependencies_from_file(create_venv=False)
    elif dep_choice == "2":
        deps_ok = install_dependencies_from_file(create_venv=True)
        if Path("tensortune_env").exists() and deps_ok:
            print_info(f"\n{BOLD}IMPORTANT: Virtual environment 'tensortune_env' was used/created.{ENDC}")
            print_info("You MUST activate it before running TensorTune.")
            act_cmd = "tensortune_env\\Scripts\\activate" if platform.system() == "Windows" else "source tensortune_env/bin/activate"
            print_info(f"  To activate: {BOLD}{act_cmd}{ENDC}")
            print_header("Next Steps"); print_success("Setup for venv complete. Activate it, then run TensorTune."); sys.exit(0)
    elif dep_choice == "3":
        print_info("Skipping auto dependency install. Ensure manual install from requirements.txt."); deps_ok = True

    if not deps_ok and dep_choice != "3":
        print_error("Dependency installation encountered issues. Review messages above.")
    
    check_and_advise_optional_components()
    find_koboldcpp_location()
    
    if confirm_yes_no("\nCreate convenience launch scripts (e.g., launch_tensortune_gui.bat)?", default_yes=True):
        create_launch_scripts_if_wanted()
    
    print_header("Setup Summary")
    print(f"{GREEN}{BOLD}TensorTune v{CURRENT_TENSORTUNE_VERSION} setup helper finished.{ENDC}")
    if dep_choice != "2":
        if deps_ok or dep_choice == "3":
            print_info("You should now be able to run TensorTune.")
            print_info("Use created launch scripts or run python tensortune_gui.py / tensortune_cli.py")
        else:
            print_error("Some steps had issues. Please review output.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt: print_error("\nInstallation cancelled by user."); sys.exit(1)
    except Exception as e:
        print_error(f"\nAn unexpected error during installation: {e}"); import traceback; traceback.print_exc(); sys.exit(1)