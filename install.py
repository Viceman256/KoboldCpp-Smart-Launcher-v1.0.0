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
    if version.major < 3 or (version.major == 3 and version.minor < 8): # Recommended 3.8+ for CTk
        print_error(f"Python 3.8+ is highly recommended for TensorTune (especially GUI). Found: Python {version.major}.{version.minor}.{version.micro}")
        if version.minor < 7:
            print_error("This Python version is too old and unsupported.")
            return False
        print_warning("You may encounter issues with older Python 3.7. Consider upgrading.")
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected.")
    return True

def generate_requirements_file_if_needed():
    """Generates requirements.txt if it doesn't exist."""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_warning(f"{requirements_file} not found. Creating a default one for TensorTune v{CURRENT_TENSORTUNE_VERSION}...")
        default_req_content = """# TensorTune v""" + CURRENT_TENSORTUNE_VERSION + """ Requirements

# Core GUI dependency
customtkinter>=5.2.0

# --- Optional but highly recommended for full functionality ---

# System information, process management (used by core for CPU threads, process killing)
psutil>=5.9.0

# NVIDIA GPU VRAM monitoring (used by core)
# Install if you have an NVIDIA GPU.
pynvml>=11.5.0

# Enhanced CLI user interface (used by tensortune_cli.py)
# Recommended for a better CLI experience.
rich>=13.4.0

# Consistent application directories (used by core if available)
# Provides better platform-agnostic paths for config/data.
appdirs>=1.4.4

# --- Platform-Specific for Advanced GPU Info (Install manually if needed) ---
# These are used by tensortune_core.py for more detailed GPU information.
# TensorTune attempts to use them if installed but functions without them
# (GPU info might be limited for respective vendors). See README for more details.
# Example: pip install wmi pyadlx pyze-l0 (some may have system prerequisites)

# --- Notes ---
# tkinter: Usually part of Python. If missing (e.g., some Linux distros),
# install via system package manager (e.g., python3-tk).
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

    pip_executable = sys.executable.replace("pythonw.exe", "python.exe") # Ensure we use 'python.exe' for pip on Windows
    pip_cmd = [pip_executable, "-m", "pip", "install", "-r", requirements_file]

    if create_venv:
        venv_name = "tensortune_env"
        if not Path(venv_name).exists():
            print_info(f"Creating virtual environment '{venv_name}'...")
            try:
                subprocess.run([sys.executable, "-m", "venv", venv_name], check=True, capture_output=True, text=True)
                print_success(f"Virtual environment '{venv_name}' created successfully.")
            except subprocess.CalledProcessError as e:
                print_error(f"Failed to create virtual environment: {e.stderr}")
                return False # Critical failure if venv creation fails when requested

            if platform.system() == "Windows":
                # On Windows, pip needs to be invoked from within the venv's Scripts path
                pip_executable_venv = str(Path(venv_name) / "Scripts" / "python.exe")
                activate_instruction = f"  {venv_name}\\Scripts\\activate"
            else:
                pip_executable_venv = str(Path(venv_name) / "bin" / "python")
                activate_instruction = f"  source {venv_name}/bin/activate"
            
            pip_cmd_venv = [pip_executable_venv, "-m", "pip", "install", "-r", requirements_file]
            print_info(f"Installing dependencies into '{venv_name}'...")
            result = subprocess.run(pip_cmd_venv, capture_output=True, text=True)

            if result.returncode == 0:
                print_success("Dependencies installed successfully into virtual environment.")
                print_info(f"\nTo use TensorTune from this virtual environment, first activate it:")
                print(activate_instruction)
                print(f"Then, from this directory, run TensorTune (e.g., python tensortune_gui.py).")
                return True # Dependencies installed in venv
            else:
                print_error(f"Failed to install dependencies into virtual environment: {result.stderr}")
                print_warning(f"Please activate the venv ('{activate_instruction}') and try running: pip install -r {requirements_file}")
                return False
        else:
            print_info(f"Virtual environment '{venv_name}' already exists. Skipping creation.")
            print_warning("If you want to reinstall in this venv, please activate it and run pip install -r requirements.txt manually, or delete the venv folder and re-run this script.")
            return True # Indicate user should manage existing venv

    else: # Install in current environment
        print_info(f"Installing dependencies into the current Python environment: {sys.executable}")
        print_warning("Consider using a virtual environment for better project isolation.")
        result = subprocess.run(pip_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success("Dependencies installed successfully.")
            return True
        else:
            print_error(f"Failed to install dependencies: {result.stderr}")
            return False

def check_and_advise_optional_components():
    """Checks for optional libraries TensorTune can use and advises."""
    print_header("Step 3: Checking Optional Components & Advising")
    
    optionals = [
        ("psutil", "system/process info (CPU threads, process kill)", True),
        ("pynvml", "NVIDIA GPU VRAM monitoring", True),
        ("rich", "enhanced CLI experience", True),
        ("appdirs", "consistent config/data paths", True),
        ("tkinter", "GUI operation (usually included with Python)", False), # False = error if missing
        ("wmi", "Windows Management Instrumentation (Windows AMD/fallback GPU info)", False), # pip install wmi
        ("pyadlx", "AMD ADLX SDK Python bindings (Windows AMD GPU info)", False), # Special install often
        ("pyze-l0", "Intel Level Zero Python bindings (Intel GPU info)", False) # Special install often
    ]

    all_good = True
    for name, purpose, is_truly_optional_for_core_function in optionals:
        try:
            __import__(name)
            print_success(f"{name} is available ({purpose}).")
        except ImportError:
            if not is_truly_optional_for_core_function and name == "tkinter":
                print_error(f"{name} is NOT available but REQUIRED for the GUI ({purpose}).")
                if platform.system() == "Linux":
                    print_info("  On Linux, try: sudo apt-get install python3-tk  OR  sudo dnf install python3-tkinter")
                else:
                    print_info("  On Windows/macOS, tkinter is usually included with Python. Try reinstalling Python from python.org, ensuring 'tcl/tk and IDLE' is selected.")
                all_good = False
            elif name in ["wmi", "pyadlx", "pyze-l0"]:
                 print_warning(f"{name} not found. TensorTune will have limited info for {purpose}.")
                 print_info(f"  If you need this, try installing (may have prerequisites): pip install {name}")
                 if name == "pyadlx": print_info("    (pyadlx often needs AMD drivers with ADLX SDK installed)")
                 if name == "pyze-l0": print_info("    (pyze-l0 often needs Intel drivers with Level Zero runtime)")
            else: # psutil, pynvml, rich, appdirs
                print_warning(f"{name} not found. TensorTune functionality related to '{purpose}' may be limited or use fallbacks.")
                print_info(f"  Consider installing: pip install {name}")
                if name == "pynvml": print_info("    (pynvml is for NVIDIA GPUs only)")

    return all_good

def find_koboldcpp_location():
    """Attempts to find KoboldCpp, suggests user input if not found."""
    print_header("Step 4: Locating KoboldCpp (Optional)")
    
    common_names = ["koboldcpp.exe", "koboldcpp", "koboldcpp.py"]
    if platform.system() != "Windows":
        common_names.append("./koboldcpp") # For executable bit set case
    
    # Check current directory first
    for name in common_names:
        if Path(name).exists() and Path(name).is_file():
            abs_path = Path(name).resolve()
            print_success(f"Found potential KoboldCpp in current directory: {abs_path}")
            print_info("  You can confirm or change this path in TensorTune settings later.")
            return str(abs_path)
            
    # Check PATH environment variable
    for name in common_names:
        found_path = shutil.which(name)
        if found_path:
            abs_path = Path(found_path).resolve()
            print_success(f"Found potential KoboldCpp in system PATH: {abs_path}")
            print_info("  You can confirm or change this path in TensorTune settings later.")
            return str(abs_path)
            
    print_warning("KoboldCpp executable/script not automatically found in common locations.")
    print_info("  This is okay! You will need to set the path to your KoboldCpp")
    print_info("  executable or Python script within TensorTune's settings after launching it.")
    return None

def check_tensortune_files():
    """Checks if essential TensorTune script files are present."""
    print_header("Step 5: Verifying TensorTune Files")
    
    required_scripts = ["tensortune_core.py", "tensortune_cli.py", "tensortune_gui.py"]
    missing = []
    for script_file in required_scripts:
        if not Path(script_file).exists():
            missing.append(script_file)
            
    if missing:
        print_error(f"Essential TensorTune script(s) missing: {', '.join(missing)}")
        print_error("Please ensure you have the complete TensorTune package.")
        return False
        
    print_success("All essential TensorTune script files seem to be present.")
    return True

def create_launch_scripts_if_wanted():
    """Creates simple cross-platform launch scripts."""
    print_header("Step 6: Create Convenience Launch Scripts (Optional)")

    if not (Path("tensortune_cli.py").exists() and Path("tensortune_gui.py").exists()):
        print_warning("Cannot create launch scripts because core GUI/CLI files are missing.")
        return False

    if platform.system() == "Windows":
        cli_script_name = "launch_tensortune_cli.bat"
        gui_script_name = "launch_tensortune_gui.bat"
        python_exe = sys.executable.replace("pythonw.exe","python.exe") # Use python.exe for console
        
        cli_content = f'@echo off\n"{python_exe}" "%~dp0tensortune_cli.py" %*\npause'
        gui_content = f'@echo off\n"{python_exe}" "%~dp0tensortune_gui.py" %*' # No pause for GUI
    else: # Linux/macOS
        cli_script_name = "launch_tensortune_cli.sh"
        gui_script_name = "launch_tensortune_gui.sh"
        
        cli_content = f"""#!/bin/sh
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"{sys.executable}" "$DIR/tensortune_cli.py" "$@"
"""
        gui_content = f"""#!/bin/sh
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"{sys.executable}" "$DIR/tensortune_gui.py" "$@"
"""

    try:
        with open(cli_script_name, "w") as f:
            f.write(cli_content)
        if platform.system() != "Windows":
            os.chmod(cli_script_name, 0o755)
        print_success(f"Created CLI launch script: {cli_script_name}")

        with open(gui_script_name, "w") as f:
            f.write(gui_content)
        if platform.system() != "Windows":
            os.chmod(gui_script_name, 0o755)
        print_success(f"Created GUI launch script: {gui_script_name}")
        return True
    except IOError as e:
        print_error(f"Could not create launch scripts: {e}")
        return False

def main():
    print(f"\n{BOLD}{BLUE}TensorTune v{CURRENT_TENSORTUNE_VERSION} - Installation Helper{ENDC}\n")
    
    if not check_python_version():
        sys.exit(1)

    if not check_tensortune_files():
        sys.exit(1)

    if not generate_requirements_file_if_needed(): # Tries to create if missing
        print_warning("Proceeding without requirements.txt. Dependency installation might fail.")
        # Allow to continue if user wants to try anyway or has manual setup

    print_header("Dependency Installation Choice")
    print("TensorTune relies on some Python packages. How would you like to install them?")
    print("  (1) Install into your current Python environment (recommended for simplicity if not using many projects).")
    print("  (2) Create a new Python virtual environment (e.g., 'tensortune_env') and install there (best practice for isolation).")
    print("  (3) Skip automatic dependency installation (you'll need to install them manually from requirements.txt).")
    
    dep_choice = ""
    while dep_choice not in ["1", "2", "3"]:
        dep_choice = input("Enter your choice (1-3) [default: 1]: ").strip() or "1"

    dependencies_installed_ok = False
    if dep_choice == "1":
        dependencies_installed_ok = install_dependencies_from_file(create_venv=False)
    elif dep_choice == "2":
        dependencies_installed_ok = install_dependencies_from_file(create_venv=True)
        if Path("tensortune_env").exists() and dependencies_installed_ok:
            print_info(f"\n{BOLD}IMPORTANT: A virtual environment 'tensortune_env' was created/used.{ENDC}")
            print_info("You MUST activate it before running TensorTune for the installed dependencies to be used.")
            if platform.system() == "Windows":
                print_info(f"  To activate: {BOLD}tensortune_env\\Scripts\\activate{ENDC}")
            else:
                print_info(f"  To activate: {BOLD}source tensortune_env/bin/activate{ENDC}")
            print_info("After activation, you can run TensorTune from this directory.")
            # Exit here to force user to activate venv
            print_header("Next Steps")
            print_success("Setup for virtual environment complete. Please activate it and then run TensorTune.")
            sys.exit(0) 
    elif dep_choice == "3":
        print_info("Skipping automatic dependency installation.")
        print_warning("Please ensure you manually install packages from 'requirements.txt' if needed.")
        dependencies_installed_ok = True # Assume user will handle it

    if not dependencies_installed_ok and dep_choice != "3":
        print_error("Dependency installation failed. Please check the error messages above.")
        print_info("You can try installing dependencies manually using: pip install -r requirements.txt")
        # Don't exit yet, allow other checks to proceed.

    check_and_advise_optional_components()
    find_koboldcpp_location() # Informational, user sets in app
    
    if confirm_yes_no("Create convenience launch scripts (e.g., launch_tensortune_gui.bat/sh)?", default_yes=True):
        create_launch_scripts_if_wanted()
    
    print_header("Setup Summary")
    print(f"{GREEN}{BOLD}TensorTune (v{CURRENT_TENSORTUNE_VERSION}) setup helper finished.{ENDC}")
    if dep_choice != "2": # If not exited due to venv creation
        if dependencies_installed_ok or dep_choice == "3":
            print_info("You should now be able to run TensorTune:")
            if platform.system() == "Windows":
                print_info(f"  Using created scripts (if any): {BOLD}launch_tensortune_gui.bat{ENDC} or {BOLD}launch_tensortune_cli.bat{ENDC}")
                print_info(f"  Or directly: {BOLD}python tensortune_gui.py{ENDC} / {BOLD}python tensortune_cli.py{ENDC}")
            else:
                print_info(f"  Using created scripts (if any): {BOLD}./launch_tensortune_gui.sh{ENDC} or {BOLD}./launch_tensortune_cli.sh{ENDC}")
                print_info(f"  Or directly: {BOLD}python3 tensortune_gui.py{ENDC} / {BOLD}python3 tensortune_cli.py{ENDC}")
        else:
            print_error("Some steps encountered issues. Please review the output above.")

def confirm_yes_no(prompt_text, default_yes=True):
    """Simple Yes/No confirmation prompt."""
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        choice = input(f"{prompt_text} {suffix}: ").strip().lower()
        if not choice:
            return default_yes
        if choice in ['y', 'yes']:
            return True
        if choice in ['n', 'no']:
            return False
        print_warning("Please answer 'yes' or 'no'.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"\nAn unexpected error occurred during installation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)