# TensorTune: WMI Setup Guide (for Windows System Information)

WMI (Windows Management Instrumentation) is a core technology in Windows that allows applications like TensorTune to query a wide range of system and hardware information. For TensorTune, WMI is primarily used as a fallback method to gather GPU details on Windows if more specific vendor libraries (like PyNVML for NVIDIA, PyADLX for AMD, or PyZE for Intel) are not available or not working. It can also provide general system RAM and CPU information.

**IMPORTANT:**
*   WMI itself is a built-in component of Windows. You do not install WMI.
*   To allow Python applications like TensorTune to *access* WMI, a Python wrapper library is needed.
*   Setup is generally very straightforward via `pip`.

## What is WMI?

Windows Management Instrumentation (WMI) is Microsoft's implementation of Web-Based Enterprise Management (WBEM), an industry standard. It provides a unified way for scripts and applications to access management information and control system components on Windows computers. This includes details about hardware (CPUs, GPUs, memory, disks), software, operating system settings, and more.

## Part 1: Prerequisites for WMI Access from Python

1.  **Windows Operating System:**
    *   WMI is available on all modern versions of Windows (Windows 7, 8, 10, 11, and Server editions).

2.  **Python Environment:**
    *   A working Python installation (Python 3.x recommended).
    *   `pip` (Python package installer) must be available.

## Part 2: Installing the WMI Python Library

TensorTune uses the `WMI` Python package to interface with the Windows WMI subsystem.

1.  **Install the `WMI` Python Package:**
    *   Open your command prompt or PowerShell (it's often best to run it as Administrator if you encounter permission issues during pip install, though usually not required for this package).
    *   Run the following command:
        ```bash
        pip install WMI
        ```
    *   This will download and install the `WMI` package into your current Python environment.

## Part 3: Verifying WMI Access from Python

After installation, you can quickly test if the `WMI` Python library can access system information:

1.  Open a Python interpreter or create a small Python script (`test_wmi.py`):
    ```python
    try:
        import wmi
        print("WMI Python library imported successfully.")
        
        c = wmi.WMI() # Connect to default WMI namespace (root\cimv2)
        
        # Example: Get OS information
        for os_info in c.Win32_OperatingSystem():
            print(f"OS Name: {os_info.Caption}")
            print(f"OS Version: {os_info.Version}")

        # Example: Get basic video controller information
        print("\nVideo Controllers:")
        for gpu in c.Win32_VideoController():
            gpu_name = getattr(gpu, 'Name', 'N/A')
            gpu_ram_bytes = getattr(gpu, 'AdapterRAM', 0)
            gpu_ram_mb = gpu_ram_bytes / (1024**2) if gpu_ram_bytes else 0
            print(f"  - {gpu_name}, AdapterRAM: {gpu_ram_mb:.0f} MB (approx.)")
        
        print("\nWMI access seems to be working.")

    except ImportError:
        print("ERROR: WMI Python library not found. Please install it using 'pip install WMI'.")
    except Exception as e:
        # WMI can sometimes raise com_error or other exceptions if the WMI service has issues.
        import traceback
        print(f"ERROR: An issue occurred while testing WMI access: {type(e).__name__} - {e}")
        print("This might indicate a problem with the WMI service on your Windows system.")
        traceback.print_exc()
    ```
2.  Run the script: `python test_wmi.py`
3.  If successful, you should see your OS information and a list of your video controllers with their approximate adapter RAM.

## How TensorTune Uses WMI

*   TensorTune automatically attempts to import and use the `WMI` Python library on Windows systems.
*   It's primarily used as a **fallback mechanism** for GPU information if:
    *   You have an NVIDIA GPU but `pynvml` is not installed/working.
    *   You have an AMD GPU but `pyadlx` is not set up/working.
    *   You have an Intel GPU but `pyze` is not installed/working.
*   WMI typically provides the GPU name and total adapter RAM. It **does not usually provide real-time VRAM usage (free/used VRAM)** without additional tools.
*   TensorTune also uses `psutil` (if available) for CPU and system RAM info, but WMI can be a fallback for some of that too.

## Troubleshooting WMI

*   **"WMI (Windows) library not found" (from TensorTune):**
    *   The `WMI` Python package is not installed. Run `pip install WMI`.
*   **"WMI (Windows) initialization failed" or other errors during WMI access (e.g., `com_error`):**
    *   This usually indicates an issue with the WMI service itself on your Windows system, not with the Python library or TensorTune directly.
    *   **Restart WMI Service:**
        1.  Open "Services" (search for `services.msc`).
        2.  Find "Windows Management Instrumentation" service.
        3.  Right-click and "Restart" it. If it's stopped, "Start" it.
    *   **Repair WMI Repository:** This is an advanced step and should be done with caution. Search online for "repair WMI repository Windows [your version]" for official Microsoft instructions.
    *   **System File Checker:** Run `sfc /scannow` in an administrator command prompt to check for and repair corrupted system files.
    *   **Permissions:** In rare cases, the user account running Python/TensorTune might lack permissions to query WMI. Try running as Administrator (though this shouldn't typically be needed for basic queries).

---

For most Windows users, a simple `pip install WMI` is all that's needed for TensorTune to have its fallback WMI capabilities ready.