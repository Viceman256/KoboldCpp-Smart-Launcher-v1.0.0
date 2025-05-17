# TensorTune: Optional PyZE Setup Guide (for Intel GPUs)

PyZE is an optional Python library that TensorTune can use to access detailed information from Intel GPUs, particularly modern ones like the Arc series and newer integrated Xe graphics. This can provide more accurate VRAM and performance metrics.

**IMPORTANT:**
*   PyZE is **NOT** strictly required for TensorTune to function on systems with Intel GPUs. If PyZE is not found or not working, TensorTune will attempt to use other methods (like WMI on Windows or sysfs on Linux) for GPU information, which might be less detailed.
*   Setting up PyZE is generally straightforward via `pip`.

## What is PyZE?

PyZE is a Python binding for Intel’s **Level Zero API**. The Level Zero API provides low-level access to Intel compute devices, including GPUs, for monitoring, management, and offloading compute tasks. TensorTune uses PyZE to query GPU properties like VRAM size.

## Part 1: Prerequisites for PyZE

1.  **Supported Intel GPU:**
    *   PyZE and Level Zero are primarily for modern Intel discrete GPUs (e.g., Intel Arc series) and some newer integrated Intel Iris Xe Graphics or newer. Older Intel integrated graphics might not be fully supported or provide useful information via Level Zero.

2.  **Up-to-date Intel GPU Drivers:**
    *   Ensure you have the latest graphics drivers installed for your Intel GPU.
    *   Download from [Intel’s official Download Center](https://www.intel.com/content/www/us/en/download-center/home.html). Select the appropriate drivers for your operating system and GPU model.

3.  **Intel Level Zero Runtime:**
    *   This runtime is **usually installed automatically with the Intel graphics drivers**, especially recent ones for Arc GPUs.
    *   If you suspect it's missing (e.g., if PyZE installs but finds no devices), advanced users can look for standalone Level Zero runtime installers, often found on the [oneAPI Level Zero GitHub releases page](https://github.com/oneapi-src/level-zero/releases). However, ensuring your main graphics driver is up-to-date is the primary step.

4.  **Python Environment:**
    *   Python 3.8 or newer is generally recommended.
    *   A 64-bit Python interpreter is typically required.
    *   `pip` (Python package installer) must be available.

## Part 2: Installing PyZE

1.  **Install the `pyze-l0` Python Package:**
    *   Open your command prompt or terminal.
    *   Run the following command:
        ```bash
        pip install pyze-l0
        ```
    *   This will download and install the PyZE library and its dependencies into your current Python environment.

2.  **(Linux Only - Optional but Recommended) Enable System Management Interface:**
    *   For PyZE to access certain system-level GPU metrics on Linux, you might need to set an environment variable:
        ```bash
        export ZES_ENABLE_SYSMAN=1
        ```
    *   You can add this line to your shell profile (e.g., `~/.bashrc` or `~/.zshrc`) to make it persistent across sessions. Then, source the file (e.g., `source ~/.bashrc`) or open a new terminal.
    *   On Windows, this step is generally not required.

## Part 3: Verifying PyZE Installation

After installation, you can quickly test if PyZE can detect your Intel GPU(s):

1.  Open a Python interpreter or create a small Python script (`test_pyze.py`):
    ```python
    try:
        import pyze.api as pyze_api
        print("PyZE imported successfully.")

        if hasattr(pyze_api, 'zeInit') and pyze_api.zeInit(0) == pyze_api.ZE_RESULT_SUCCESS:
            print("Level Zero initialized successfully.")
            
            num_drivers_ptr = pyze_api.new_uint32_tp()
            pyze_api.zeDriverGet(num_drivers_ptr, None)
            num_drivers = pyze_api.uint32_tp_value(num_drivers_ptr)
            pyze_api.delete_uint32_tp(num_drivers_ptr)
            print(f"Found {num_drivers} Level Zero driver(s).")

            if num_drivers > 0:
                # Further checks for devices can be added here if desired
                print("PyZE seems to be working with the Level Zero runtime.")
            else:
                print("WARNING: No Level Zero drivers found, though PyZE and Level Zero initialized.")
        else:
            print("ERROR: Failed to initialize Level Zero via PyZE. Check drivers and runtime.")

    except ImportError:
        print("ERROR: PyZE (pyze-l0) library not found. Please install it using 'pip install pyze-l0'.")
    except Exception as e:
        print(f"ERROR: An issue occurred while testing PyZE: {e}")
    ```
2.  Run the script: `python test_pyze.py`
3.  If successful, you should see messages indicating PyZE imported, Level Zero initialized, and drivers/devices were found.

## How TensorTune Uses PyZE

*   TensorTune will automatically attempt to import and initialize PyZE.
*   If PyZE is successfully loaded and can find compatible Intel GPUs, TensorTune will use it to get GPU information (like VRAM size and name).
*   If PyZE is not installed, fails to load, or finds no compatible devices, TensorTune will show a related status message in its log and fall back to other methods (WMI on Windows, sysfs on Linux).

## Troubleshooting PyZE

*   **"PyZE (Intel Level Zero) library not found" (from TensorTune):**
    *   The `pyze-l0` package is not installed in the Python environment TensorTune is using. Run `pip install pyze-l0`.
*   **"PyZE (Intel Level Zero) zeInit call failed" / "No Level Zero devices found":**
    *   **Update Intel Drivers:** This is the most common cause. Ensure you have the latest official drivers for your Intel GPU.
    *   **Reboot:** After driver updates, a system reboot is often necessary.
    *   **GPU Support:** Verify that your specific Intel GPU model is supported by the Level Zero API. Generally, discrete Arc GPUs and newer integrated Xe graphics are.
    *   **(Linux) `ZES_ENABLE_SYSMAN=1`:** Ensure this environment variable is set if you're on Linux.
    *   **Level Zero Runtime:** Though usually bundled, in rare cases, the Level Zero runtime might be missing or corrupted. Consider reinstalling drivers or checking for standalone Level Zero runtimes from Intel's oneAPI GitHub.
*   **Other PyZE errors during TensorTune startup:**
    *   These might indicate a conflict with other installed software or an issue with the PyZE installation itself. You can try reinstalling PyZE: `pip uninstall pyze-l0` then `pip install pyze-l0`.

---

For most users with compatible Intel hardware, a simple `pip install pyze-l0` and up-to-date drivers are all that's needed for TensorTune to leverage PyZE.