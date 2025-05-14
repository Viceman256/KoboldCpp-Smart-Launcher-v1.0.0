#!/usr/bin/env python3
"""
KoboldCpp Smart Launcher - Installation Script
This script helps set up the KoboldCpp Smart Launcher by installing 
dependencies and checking for required components.
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
    print_header("Checking Python Version")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print_error(f"Python 3.7+ is required. Found: Python {version.major}.{version.minor}.{version.micro}")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected (compatible)")
    return True

def install_dependencies(create_venv=False):
    """Install required dependencies"""
    print_header("Installing Dependencies")
    
    # Check if in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if create_venv and not in_venv:
        venv_name = "launcher_env"
        print_info(f"Creating virtual environment '{venv_name}'...")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
            
            # Activate instructions
            if platform.system() == "Windows":
                activate_cmd = f"{venv_name}\\Scripts\\activate"
            else:
                activate_cmd = f"source {venv_name}/bin/activate"
                
            print_success(f"Virtual environment created successfully.")
            print_info(f"To activate the virtual environment, run:")
            print(f"    {activate_cmd}")
            print_info(f"Then run this script again after activation.")
            return False
        except subprocess.CalledProcessError:
            print_error("Failed to create virtual environment. Continuing with system Python...")
    
    # Install dependencies
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_warning("requirements.txt not found. Creating default requirements file...")
        with open(requirements_file, "w") as f:
            f.write("""# Core dependencies
customtkinter>=5.2.0

# Optional but recommended for full functionality
psutil>=5.9.0     # System information and process management
pynvml>=11.5.0    # NVIDIA GPU VRAM monitoring
rich>=13.4.0      # Improved CLI interface
""")
    
    print_info("Installing required dependencies...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print_error(f"Failed to install dependencies: {result.stderr}")
        return False
    
    print_success("Dependencies installed successfully")
    return True

def check_optional_dependencies():
    """Check for optional dependencies and provide info"""
    print_header("Checking Optional Dependencies")
    
    # Check for psutil
    try:
        import psutil
        print_success("psutil is installed (enables system monitoring and process management)")
    except ImportError:
        print_warning("psutil is not installed. Some system monitoring features will be limited.")
        print_info("Install with: pip install psutil")
    
    # Check for pynvml (NVIDIA GPU monitoring)
    try:
        import pynvml
        print_success("pynvml is installed (enables NVIDIA GPU VRAM monitoring)")
    except ImportError:
        print_warning("pynvml is not installed. NVIDIA GPU VRAM monitoring will be unavailable.")
        print_info("Install with: pip install pynvml")
    
    # Check for rich (improved CLI interface)
    try:
        import rich
        print_success("rich is installed (enables improved CLI interface)")
    except ImportError:
        print_warning("rich is not installed. CLI interface will use basic formatting.")
        print_info("Install with: pip install rich")
    
    # Check for tkinter (required for GUI)
    try:
        import tkinter
        print_success("tkinter is installed (required for GUI)")
    except ImportError:
        print_error("tkinter is not installed. GUI interface will not work.")
        if platform.system() == "Linux":
            if shutil.which("apt"):
                print_info("Install with: sudo apt-get install python3-tk")
            elif shutil.which("dnf"):
                print_info("Install with: sudo dnf install python3-tkinter")
            else:
                print_info("Please install tkinter using your package manager.")
        elif platform.system() == "Darwin":  # macOS
            print_info("Install tkinter by reinstalling Python with the official installer from python.org")
        else:
            print_info("Install tkinter by reinstalling Python with the official installer from python.org")
    
    return True

def find_koboldcpp():
    """Try to find KoboldCpp installation"""
    print_header("Locating KoboldCpp")
    
    # Common names for KoboldCpp
    common_names = ["koboldcpp.exe", "koboldcpp", "koboldcpp.py"]
    if platform.system() != "Windows":
        common_names.append("./koboldcpp")
    
    # Check current directory
    for name in common_names:
        if os.path.exists(name):
            print_success(f"Found KoboldCpp in current directory: {os.path.abspath(name)}")
            return os.path.abspath(name)
    
    # Check in PATH
    for name in common_names:
        path = shutil.which(name)
        if path:
            print_success(f"Found KoboldCpp in PATH: {path}")
            return path
    
    print_warning("KoboldCpp executable not found automatically.")
    print_info("You will need to specify the path to KoboldCpp in the launcher settings.")
    return None

def check_files():
    """Check if all required files are present"""
    print_header("Checking Required Files")
    
    required_files = ["koboldcpp_core.py", "koboldcpp_launcher.py", "koboldcpp_launcher_gui.py"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print_error(f"Missing required files: {', '.join(missing_files)}")
        return False
    
    print_success("All required files are present")
    return True

def create_launch_scripts():
    """Create convenient launch scripts"""
    print_header("Creating Launch Scripts")
    
    # Create launcher for CLI
    cli_script_name = "launch_cli.py" if platform.system() == "Windows" else "launch_cli.sh"
    
    if platform.system() == "Windows":
        cli_content = """#!/usr/bin/env python3
import os
import sys
import subprocess

# Run the CLI launcher
launcher_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "koboldcpp_launcher.py")
subprocess.run([sys.executable, launcher_path])
"""
    else:
        cli_content = """#!/bin/bash
# Run the CLI launcher
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3 "${DIR}/koboldcpp_launcher.py"
"""
    
    with open(cli_script_name, "w") as f:
        f.write(cli_content)
    
    # Make shell script executable on Unix-like systems
    if platform.system() != "Windows" and cli_script_name.endswith(".sh"):
        os.chmod(cli_script_name, 0o755)
    
    # Create launcher for GUI
    gui_script_name = "launch_gui.py" if platform.system() == "Windows" else "launch_gui.sh"
    
    if platform.system() == "Windows":
        gui_content = """#!/usr/bin/env python3
import os
import sys
import subprocess

# Run the GUI launcher
launcher_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "koboldcpp_launcher_gui.py")
subprocess.run([sys.executable, launcher_path])
"""
    else:
        gui_content = """#!/bin/bash
# Run the GUI launcher
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3 "${DIR}/koboldcpp_launcher_gui.py"
"""
    
    with open(gui_script_name, "w") as f:
        f.write(gui_content)
    
    # Make shell script executable on Unix-like systems
    if platform.system() != "Windows" and gui_script_name.endswith(".sh"):
        os.chmod(gui_script_name, 0o755)
    
    print_success(f"Created CLI launcher: {cli_script_name}")
    print_success(f"Created GUI launcher: {gui_script_name}")
    return True

def main():
    """Main installation function"""
    print(f"\n{BOLD}{BLUE}KoboldCpp Smart Launcher - Installation{ENDC}\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check required files
    if not check_files():
        print_warning("Some required files are missing. Please download the complete package.")
        cont = input("Continue anyway? (y/n): ").lower().strip()
        if cont != 'y':
            sys.exit(1)
    
    # Installation options
    print_header("Installation Options")
    print("1. Install in current Python environment")
    print("2. Create a virtual environment and install")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        create_venv = False
    elif choice == '2':
        create_venv = True
    else:
        print_info("Installation cancelled.")
        sys.exit(0)
    
    # Install dependencies
    if not install_dependencies(create_venv):
        if create_venv:
            # If we just created a venv, exit because user needs to activate it
            sys.exit(0)
        else:
            print_error("Failed to install dependencies. Please try again or install manually.")
            sys.exit(1)
    
    # Check optional dependencies
    check_optional_dependencies()
    
    # Find KoboldCpp
    koboldcpp_path = find_koboldcpp()
    
    # Create launch scripts
    create_launch_scripts()
    
    # Installation complete
    print_header("Installation Complete")
    
    if koboldcpp_path:
        print_success(f"KoboldCpp found at: {koboldcpp_path}")
    else:
        print_warning("KoboldCpp not found. You'll need to set the path in the launcher settings.")
    
    print_info("To run the launcher:")
    if platform.system() == "Windows":
        print("  - CLI: python launch_cli.py")
        print("  - GUI: python launch_gui.py")
    else:
        print("  - CLI: ./launch_cli.sh")
        print("  - GUI: ./launch_gui.sh")
    
    print(f"\n{GREEN}{BOLD}Installation completed successfully!{ENDC}\n")

if __name__ == "__main__":
    main()