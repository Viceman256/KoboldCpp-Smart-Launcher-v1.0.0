# TensorTune: Optional PyADLX Setup Guide (for AMD GPUs on Windows)

PyADLX is an optional Python library that TensorTune can use to access detailed information from AMD GPUs on Windows systems. This can sometimes provide more precise VRAM metrics than other methods.

**IMPORTANT:**
*   PyADLX is **NOT** required for TensorTune to function. If PyADLX is not found or not working, TensorTune will automatically fall back to using WMI (Windows Management Instrumentation) for AMD GPU information.
*   Setting up PyADLX is an **advanced, manual process** involving C++ compilation. It is not a simple `pip install`.
*   You only need to consider this if you have an AMD GPU on Windows AND you specifically want to enable the potentially more detailed metrics PyADLX offers through TensorTune.

## What is PyADLX?

PyADLX consists of Python bindings that you build for AMD's **ADLX (AMD Device Library X) SDK**. The ADLX SDK provides low-level access to AMD GPU hardware. TensorTune looks for a file typically named `pyadlx.pyd` or similar, which contains these bindings.

## Part 1: Prerequisites

Before attempting to build the PyADLX bindings, ensure you have the following:

1.  **AMD Adrenalin Software & ADLX SDK:**
    *   The full AMD Adrenalin software suite must be installed. This suite typically includes the necessary ADLX SDK components.
    *   Download and install the latest Adrenalin software for your specific AMD GPU from the [official AMD website](https://www.amd.com/en/support).
    *   A "Full Install" or "Standard Install" option is recommended over a "Minimal" or "Driver-only" install to ensure ADLX SDK components are present.

2.  **Visual Studio 2019 (or compatible):**
    *   Microsoft Visual Studio 2019 (Community Edition is free) with the "Desktop development with C++" workload installed. Other versions might work but 2019 is specified by AMD.
    *   Download from the [Visual Studio website](https://visualstudio.microsoft.com/downloads/).

3.  **Python 3.9:**
    *   AMD's guide specifies Python 3.9. While other 3.x versions *might* work with adjustments, it's safest to use the specified version for building these bindings.
    *   Download from the [official Python website](https://www.python.org/downloads/release/python-3913/) (or a similar 3.9.x version).
    *   Ensure Python 3.9 is added to your system's PATH during installation, or you know its installation location.

4.  **`pybind11` Library:**
    *   This is a C++ library for creating Python bindings. Install it for your Python 3.9 environment:
        ```bash
        C:\Path\To\Python39\python.exe -m pip install pybind11
        ```
        (Replace `C:\Path\To\Python39` with your actual Python 3.9 installation path).

5.  **`PYHOME` System Environment Variable:**
    *   This variable must point to your Python 3.9 installation directory.
    *   Search for "Edit the system environment variables" in Windows search.
    *   Click "Environment Variables..."
    *   Under "System variables" (or "User variables for <your_username>" if preferred), click "New..."
    *   Variable name: `PYHOME`
    *   Variable value: `C:\Path\To\Python39` (e.g., `C:\Python39` or `C:\Users\YourUser\AppData\Local\Programs\Python\Python39`)
    *   Click OK on all dialogs. You may need to restart your command prompt or Visual Studio for this change to take effect.

## Part 2: Building the PyADLX Python Bindings (`.pyd` file)

These steps are based on AMD's official guide.
https://gpuopen.com/manuals/adlx/adlx-page_guide_bindpy/

1.  **Create a Project Folder:**
    *   Create a folder for your Python ADLX binding wrapper project, for example, `C:\ADLXPybind`.

2.  **Create `bind.cpp`:**
    *   In your project folder (`C:\ADLXPybind`), create a new file named `bind.cpp`.
    *   You will need to populate this file with C++ code using `pybind11` to wrap the ADLX interfaces you need. AMD often provides samples for this, or you might find community examples.
    *   **Note:** This is the most complex part. AMD mentions: *"Edit `bind.cpp` and bind the interfaces you wish to wrap using `pybind11`. To create bindings for event notifications, inherit the respective listener interface. For example, inherit `IADLXDisplayListChangedListener`. Refer to the Python samples `bind.cpp` file (often included with the ADLX SDK or on GPUOpen) for an example of a Python binding."*
    *   TensorTune specifically uses `ADLXHelper` and related functions for GPU VRAM info. Your `bind.cpp` must expose these.

3.  **Create and Configure Visual Studio Project:**
    *   Open Visual Studio 2019.
    *   Create a new project: "Dynamic-Link Library (DLL)" (C++).
    *   Name the project `pyadlx` (or `ADLXPybind` as AMD suggests; `pyadlx` is what TensorTune tries to import by default). Let's assume you name the project `pyadlx`.
    *   Configure the project properties (Right-click on the project in Solution Explorer -> Properties):
        *   **Configuration:** All Configurations (or specifically Release)
        *   **Platform:** x64 (recommended for modern systems)
        *   **General > Target Name:** `pyadlx` (if your project is named `pyadlx`)
        *   **General > Target Extension:** `.pyd`
        *   **VC++ Directories > Include Directories:** Add the following (Edit and add new lines):
            *   `$(PYHOME)\include`
            *   `$(PYHOME)\Lib\site-packages\pybind11\include`
            *   The path to your ADLX SDK's `ADLXHelper` and general include files (e.g., `C:\AMD\ADLX\SDK\ADLXHelper\Windows\Cpp` and `C:\AMD\ADLX\SDK\Include` - *adjust these paths based on your ADLX SDK installation location*).
        *   **VC++ Directories > Library Directories:** Add:
            *   `$(PYHOME)\libs`
            *   The path to your ADLX SDK's library files if any are needed directly (usually not for header-only helpers or if ADLX is system-installed).
        *   **C/C++ > Precompiled Headers > Precompiled Header:** "Not Using Precompiled Headers"
        *   **Linker > Input > Additional Dependencies:** Add `python39.lib` (Edit and add to the list).

4.  **Add Source Files to Project:**
    *   Add your `bind.cpp` file to the Visual Studio project (Right-click "Source Files" -> Add -> Existing Item).
    *   Add the following files from your ADLX SDK installation to the project (you'll need to locate these in your ADLX SDK directory, often installed with Adrenalin, e.g., in a subfolder of `C:\Program Files\AMD\ADLX` or similar):
        *   `SDK/Platform/Windows/WinAPIs.cpp`
        *   `SDK/ADLXHelper/Windows/Cpp/ADLXHelper.h`
        *   `SDK/ADLXHelper/Windows/Cpp/ADLXHelper.cpp`

5.  **Build the Project:**
    *   In Visual Studio, select "Build" -> "Build Solution" (or F7).
    *   If successful, this will generate a `pyadlx.pyd` file (or `ADLXPybind.pyd` if you named the project that) in your solution's output directory (e.g., `C:\ADLXPybind\x64\Release\pyadlx.pyd`).

## Part 3: Installing the Built `pyadlx.pyd` for TensorTune

Once you have successfully built `pyadlx.pyd`:

1.  **Locate the `.pyd` file:** Find it in your Visual Studio project's output directory (e.g., `YourProjectFolder\x64\Release\pyadlx.pyd`).

2.  **Place it where Python can find it:** You have a few options:

    *   **Option A (Recommended for ease of use with TensorTune if distributed as a standalone app/script):**
        Place `pyadlx.pyd` in the **same directory as the TensorTune executable (`tensortune_gui.exe` or `tensortune_core.py`)**. Python usually checks the script's own directory first.

    *   **Option B (Python's `site-packages`):**
        Copy `pyadlx.pyd` to the `site-packages` directory of the Python environment that TensorTune will use.
        *   If TensorTune uses your system's Python 3.9: `C:\Path\To\Python39\Lib\site-packages\pyadlx.pyd`
        *   If TensorTune uses a virtual environment, place it in that environment's `site-packages` folder.

    *   **Option C (Adding to `PYTHONPATH`):**
        You can place `pyadlx.pyd` in a custom directory and add that directory to your `PYTHONPATH` environment variable. (Less common for single applications).

**After placing `pyadlx.pyd`, restart TensorTune.** It should now attempt to load the library.

## Troubleshooting

*   **Build Errors in Visual Studio:**
    *   Double-check all project path configurations for Python includes, pybind11 includes, and Python libs.
    *   Ensure the ADLX SDK files (`WinAPIs.cpp`, `ADLXHelper.h`, `ADLXHelper.cpp`) are correctly added and their paths are accessible.
    *   Make sure you're building for the correct platform (x64) that matches your Python 3.9 installation.

*   **TensorTune still says "PyADLX (AMD) library not found":**
    *   The `pyadlx.pyd` file is not in a location Python can find it (see "Part 3").
    *   The `.pyd` file might be built for a different Python version or architecture (e.g., 32-bit vs 64-bit). Ensure consistency.
    *   Filename mismatch: TensorTune might be looking for `pyadlx.pyd`, but you named your build output something else (like `ADLXPybind.pyd`). If so, rename your output file to `pyadlx.pyd`.

*   **TensorTune says "ADLX system helper failed to initialize" (or similar):**
    *   The `pyadlx.pyd` file was found and imported, but it couldn't properly connect to the underlying ADLX system services.
    *   This usually points to an issue with the AMD Adrenalin software installation, driver problems, or missing ADLX SDK runtime components.
    *   Try reinstalling AMD Adrenalin software (using the "Full Install" option).
    *   Ensure your AMD drivers are up to date.

*   **TensorTune says "ADLXHelper is missing":**
    *   The `pyadlx.pyd` file was found, but it seems to be an incomplete or incorrectly built version that doesn't expose the `ADLXHelper` class TensorTune expects. Review your `bind.cpp` and ensure `ADLXHelper` and its necessary methods are correctly bound.

---

**Disclaimer:** Building C++ Python bindings is a complex task. This guide provides an overview based on AMD's instructions. Success may depend on your specific system configuration, ADLX SDK version, and C++ development experience. For most users, the fallback WMI mechanism in TensorTune for AMD GPUs on Windows is sufficient.