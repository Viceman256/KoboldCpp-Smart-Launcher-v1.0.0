# TensorTune v1.1.0

TensorTune is a user-friendly toolkit for efficiently running KoboldCpp, intelligently managing VRAM usage through optimized tensor offload strategies to maximize performance on consumer GPUs. Available in both GUI and CLI versions.

![TensorTune GUI](https://raw.githubusercontent.com/Viceman256/TensorTune/main/screenshots/GUI1.png)
_(TensorTune v1.1.0 GUI showing the main tuning interface)_

## 🚀 Performance Examples with TensorTune

TensorTune helps you achieve significant performance gains by intelligently offloading parts of your model. Here's what's possible (examples use manually crafted commands that TensorTune helps you discover):

### Example 1: Qwen3-32B on RTX 3060 (12GB)
-   **Traditional:** ~3.98 t/s
-   **With Smart Tensor Offloading:** ~10.61 t/s (Approx. **+166%** speed)
    ```bash
    # Example command that TensorTune can help derive:
    python koboldcpp.py --model Qwen3-32B-Q4_K_M.gguf --gpulayers 65 --overridetensors "\.[13579]\.ffn_up|\.[1-3][13579]\.ffn_up=CPU" ...
    ```

### Example 2: Gemma3-27B on RTX 2060 (6GB)
-   **Traditional:** ~6.86 t/s
-   **With Smart Tensor Offloading:** ~10.4 t/s (Approx. **+52%** speed)
    ```bash
    # Example command:
    python koboldcpp.py --model gemma3-27b-IQ4_XS.gguf --gpulayers 99 --overridetensors "\.(5[3-9]|6[0-3])\.(ffn_*)=CPU" ...
    ```

### Example 3: Mixtral-8x7B MoE on RTX 4070 (12GB)
-   **Traditional:** ~15.3 t/s
-   **With Smart Tensor Offloading (MoE focused):** ~24.7 t/s (Approx. **+61%** speed)
    ```bash
    # Example command:
    python koboldcpp.py --model mixtral-8x7b-v0.1.Q5_K_M.gguf --gpulayers 32 --overridetensors "ffn_.*_exps=CPU" ...
    ```
*(Full example commands and more details can be found in `tensortune_examples.py`)*

## 📋 Features

-   **Intelligent Auto-Tuning:** Automatically finds optimal tensor offload strategies (OT Levels) based on your hardware and model.
-   **New in 1.1.0: "Set as Preferred" Strategy:** Mark your best-performing tuned configurations for easy recall.
-   **Multiple Launch Options:** Direct launch, best remembered config, or full auto-tuning.
-   **VRAM Monitoring:** Real-time display of GPU VRAM usage, with support for vendor-specific selection and manual override.
-   **Configuration Management:** Global, model-specific, and session-based settings. Configurable data/config paths using `appdirs` if available.
-   **Database-Backed History:** Remembers what worked best for each model.
-   **Process Management:** Easily start and stop KoboldCpp instances.
-   **Both CLI and GUI:** Choose the interface that suits your preference.
-   **Dynamically Adjusts GPU Layers:** Coordinates GPU layer counts with tensor offload levels for maximum performance.
-   **Export/Import Settings:** Backup and share your TensorTune configurations.
-   **GUI Enhancements:** Scrollable tuning page, improved layouts, clearer VRAM display, and more robust operation.

## 🔧 Installation of TensorTune

### Quick Start

1.  Ensure you have Python 3.8+ installed (Python 3.7 may work but 3.8+ is recommended for GUI stability).
2.  Download or clone this repository:
    ```bash
    git clone https://github.com/Viceman256/TensorTune.git
    cd TensorTune
    ```
3.  Run the installation script:
    ```bash
    python install.py 
    # On Linux/macOS, you might use: python3 install.py
    ```
    The script will guide you through:
    - Checking Python version.
    - Creating/updating `requirements.txt`.
    - Installing necessary Python packages (optionally in a virtual environment).
    - Checking for optional components and advising on their setup.
    - Creating convenient launch scripts (e.g., `launch_tensortune_gui.bat` or `./launch_tensortune_gui.sh`).

### Dependencies

Key dependencies are listed in `requirements.txt`. Core ones include:
-   `customtkinter` (for GUI)
-   `psutil` (system info)
-   `pynvml` (NVIDIA GPU info)
-   `rich` (enhanced CLI)
-   `appdirs` (config paths)

#### Dependencies for Advanced GPU Information (Optional)
TensorTune can leverage platform-specific libraries for more detailed GPU metrics. These are generally optional and TensorTune will show an informational message if they are relevant but not found.

-   **NVIDIA:** `pynvml` (install via `pip install pynvml`). Requires NVIDIA drivers.
-   **AMD (Windows - PyADLX):** For the most detailed AMD GPU info, TensorTune can use `PyADLX`.
    *   **Nature:** This is **not a standard pip-installable package**. It requires Python bindings to be manually built against AMD's ADLX SDK, which is usually part of the AMD Adrenalin Software.
    *   **Guidance:** Refer to [AMD's Official Python Binding Guide for ADLX](https://gpuopen.com/manuals/adlx/adlx-page_guide_bindpy/) for build instructions.
    *   **Fallback:** If PyADLX is not found, TensorTune uses WMI on Windows for AMD GPU info.
-   **Intel (PyZE):** For detailed Intel Arc GPU info, TensorTune uses `pyze-l0`.
    *   **Installation:** Try `pip install pyze-l0`.
    *   **Requirements:** This typically needs up-to-date Intel drivers and the Level Zero runtime.
-   **Windows Fallback (WMI):** For general GPU info on Windows, especially if vendor-specific libraries are unavailable.
    *   **Installation:** `pip install WMI`.

## 📚 Using TensorTune

TensorTune automatically creates and manages its configuration files (`tensortune_config.json`) and history database (`tensortune_history.db`) in your user-specific application directory (e.g., `~/.config/TensorTune` on Linux, `AppData/Roaming/TensorTune/Config` and `AppData/Local/TensorTune/Data` on Windows, managed by `appdirs` if available).

### GUI Version (`tensortune_gui.py` or launch script)
The GUI offers a comprehensive interface with tabs for "Tune & Launch," "Settings," and "History."
-   **Tune & Launch:** Select models, monitor VRAM, and initiate tuning sessions or direct launches.
-   **Auto-Tuning Session:** Interactively adjust Tensor Offload (OT) Levels, launch & monitor KoboldCpp, view live output, and save or mark preferred configurations. The tuning page is now scrollable for better usability.
-   **Settings:** Configure KoboldCpp path, global/model-specific arguments, GPU selection & VRAM override, UI theme, and manage config import/export.
-   **History:** Review past launches and their performance.

### CLI Version (`tensortune_cli.py` or launch script)
The CLI provides an interactive text-based experience for all core TensorTune functionalities, including model selection, tuning, launching, settings management, and history review.

## 📝 Understanding Tensor Offloading

Traditional layer offloading moves entire transformer layers between CPU and GPU. Smart Tensor Offloading, which TensorTune facilitates, is more granular. It allows specific, often large, tensors within layers (like parts of the Feed-Forward Network) to be moved to CPU RAM, while keeping the more compute-intensive tensors (like attention mechanisms) on the GPU. This often allows more layers to be primarily processed by the GPU, significantly boosting performance on VRAM-constrained hardware.

TensorTune helps by:
- Analyzing model characteristics (size, layers, MoE architecture).
- Systematically testing different offloading patterns (OT Levels).
- Monitoring VRAM usage and KoboldCpp load success/failure.
- Remembering what works best for each model and your hardware.

## ⚙️ Optional Dependencies & Advanced Setup

TensorTune can leverage additional Python libraries for more detailed hardware information, especially for specific GPU vendors. These are **optional**. If not installed, TensorTune will use fallback methods.

-   **NVIDIA GPUs:** `pynvml` (usually installed by `requirements.txt`) provides detailed VRAM and GPU stats.
-   **AMD GPUs (Windows):** `PyADLX` can provide enhanced details. This requires a manual C++ binding build. See `PYADLX_SETUP_GUIDE.md` for instructions.
-   **Intel GPUs (Arc/Xe):** `pyze-l0` (PyZE) offers specific Intel GPU metrics. See `PYZE_SETUP_GUIDE.md` for setup.
-   **Windows Fallback:** `WMI` Python package is used as a general fallback on Windows. See `WMI_SETUP_GUIDE.md` for details if issues arise.

## 🔍 Troubleshooting

-   **KoboldCpp Not Found:** Set the correct path in the Settings tab (GUI) or Launcher Settings (CLI).
-   **Python Errors:** Ensure you've installed all dependencies from `requirements.txt`.
-   **VRAM Info Not Displaying / Library Errors:**
    -   **NVIDIA:** Ensure `pynvml` is installed (`pip install pynvml`).
    -   **AMD (Windows):** If you see "PyADLX library not found" or related errors, this is often normal as PyADLX is optional and requires manual setup. TensorTune will use WMI as a fallback. If you wish to attempt PyADLX setup for potentially more detailed info, refer to **`PYADLX_SETUP_GUIDE.md`**.
    -   **Intel (Arc/Xe):** If you see "PyZE library not found" or initialization errors, ensure `pyze-l0` is installed (`pip install pyze-l0`) and your Intel drivers/Level Zero runtime are up-to-date. Refer to **`PYZE_SETUP_GUIDE.md`**.
    -   **WMI (Windows):** If you see WMI errors, ensure the `WMI` package is installed (`pip install WMI`) and your Windows WMI service is functioning. Refer to **`WMI_SETUP_GUIDE.md`**.
    -   Also, check GPU selection settings in TensorTune.
-   **Auto-Tuning Fails:** Try reducing context size or other memory-intensive settings. Manually adjust the OT Level towards more CPU offload.

## 🙏 Acknowledgments
This project leverages the powerful tensor offloading capabilities of KoboldCpp. TensorTune aims to make these advanced optimizations more accessible. Inspired by community discussions on maximizing LLM performance.

## 📜 License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## 📸 Screenshots

CLI Main Menu & Model Selection:
![TensorTune CLI Main Menu](https://raw.githubusercontent.com/Viceman256/TensorTune/main/screenshots/cli1.png)

CLI Tuning Session:
![TensorTune CLI Tuning Session](https://raw.githubusercontent.com/Viceman256/TensorTune/main/screenshots/cli2.png)

GUI Main Tab (Tune & Launch):
![TensorTune GUI Main Tab](https://raw.githubusercontent.com/Viceman256/TensorTune/main/screenshots/GUI1.png)

GUI Auto-Tuning Session (Scrollable):
![TensorTune GUI Tuning Session](https://raw.githubusercontent.com/Viceman256/TensorTune/main/screenshots/GUI2.png)

GUI Settings Tab:
![TensorTune GUI Settings Tab](https://raw.githubusercontent.com/Viceman256/TensorTune/main/screenshots/GUI4.png)

GUI History Tab:
![TensorTune GUI History Tab](https://raw.githubusercontent.com/Viceman256/TensorTune/main/screenshots/GUI6.png)


## Latest Release: TensorTune v1.1.0

Key changes in this version include:
-   New "Set as Preferred" tuning strategy feature in the GUI.
-   Scrollable auto-tuning session page in the GUI for better usability.
-   Enhanced GPU detection (SysFS fallbacks, PyADLX/PyZE info), error handling, and VRAM display logic.
-   Numerous bug fixes for both CLI and GUI leading to improved stability and user experience.
-   More robust KoboldCpp integration and configuration management.
-   Updated installation script and dependency guidance.

For a full list of changes, see the v1.1.0 Release Notes on GitHub or the `CHANGELOG.md` file.

## 🤝 Contributing
Contributions, bug reports, and feature suggestions are welcome! Please feel free to open an issue or submit a Pull Request.