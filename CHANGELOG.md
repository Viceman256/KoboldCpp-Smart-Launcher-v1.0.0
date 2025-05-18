# TensorTune v1.2.0 - Changelog

## Overview
Version 1.2.0 is a significant update focusing on enhanced user control over GPU layer allocation, a comprehensive rewrite of core model analysis and offloading logic, and numerous improvements to stability and user experience in both the GUI and CLI.

---

## Core Improvements (`tensortune_core.py`)
-   **Rewritten Model Analysis (`analyze_filename`):**
    -   Significantly improved model family recognition (Gemma, Llama, Mistral, Mixtral, Qwen, Phi, Dark Champion, etc.) for more accurate default layer counts.
    -   Enhanced heuristics for determining `num_layers` based on model size (e.g., 7B, 30B, 70B) and MoE characteristics.
    -   Added specific handling for MoE models like Dark Champion to ensure appropriate layer counts (e.g., 48 layers for 21B+ MoE).
    -   Improved fallback mechanisms for layer count determination when not explicitly found in the filename.
    -   Refined VRAM estimation logic.
-   **Rebuilt GPU Layer Allocation Logic (`get_gpu_layers_for_level`):**
    *   Completely redesigned to provide smoother and more granular transitions between CPU-biased and GPU-biased offload levels.
    *   Eliminated the "big jump" issue where layer counts would abruptly go from a low number to 999 (all layers).
    *   Implemented percentage-based scaling that respects the actual total layer count of different model architectures, providing more intuitive steps.
    *   Tailored allocation strategies for both MoE and Dense models, with different sensitivity curves for OT levels.
    *   Ensures a minimum of 1 layer is always assigned if GPU offload is active.
-   **More Informative Offload Descriptions (`get_offload_description`):**
    *   Updated to provide clearer, more qualitative insights into the current tensor offloading strategy based on the OT Level.
    *   Improved display of effective GPU layer counts versus total model layers.
-   **Historical Configuration Handling:**
    *   Improved logic in `find_best_historical_config` to better prioritize user-marked "preferred" configurations and successful runs that fit current VRAM.
    *   More robust parsing of historical arguments.
-   **Argument Building (`build_command`):**
    *   Now correctly incorporates manual GPU layer overrides, ensuring they take precedence over OT-level-derived layer counts.
    *   Refined logic for handling `auto`, `off`, and numeric values for `--gpulayers` in conjunction with `--nogpulayers`.

---

## GUI Improvements (`tensortune_gui.py`)
-   **New Feature: Manual GPU Layer Control:**
    *   Added an "Auto GPU Layers" checkbox in the tuning session view.
    *   When unchecked, users can enter a specific number of GPU layers in an accompanying entry field. This manual value will be used for the launch command, overriding the OT level's automatic calculation.
-   **Tuning Session Enhancements:**
    *   Manual GPU layer settings are now correctly applied and reflected in the "Proposed Command" display.
    *   Fixed an issue where "Set as Preferred" configurations might not be correctly prioritized when restarting a tuning session for the same model.
    *   The "Last Monitored Result" display in the tuning view is now more reliably updated from historical data.
    *   Improved UI state management for tuning buttons, especially after monitoring or changing settings.
-   **Stability & UX:**
    *   Fixed `TclError` that could occur when closing various dialogs (e.g., model-specific args, first-time setup) by improving focus management and widget destruction sequences.
    *   Refined handling of model-specific configuration dialogs, ensuring changes are correctly applied and UI is updated.
    *   Corrected an issue in `_handle_post_monitor_action` where `self.kcpp_process_obj` might be used after being set to `None`.

---

## CLI Improvements (`tensortune_cli.py`)
-   **Manual GPU Layer Control:**
    *   The CLI now implicitly supports manual GPU layer overrides if a user edits the `--gpulayers` value directly in the "Edit Session Args" or "Permanent Model Args" sections. The `build_command` function will respect this.
-   **Monitoring & Progress:**
    *   Fixed issues with Rich library progress bar display during live KCPP monitoring, ensuring it completes or hides correctly.
    *   The progress bar now shows "KCPP API Ready!" and auto-completes upon successful detection.
    *   Added a small delay after successful API detection to allow users to see the completed progress bar before further actions.
    *   Improved the reliability of the KCPP output monitoring thread and its interaction with user stop requests.
-   **Tuning Logic:**
    *   Ensured "preferred" historical configurations are better respected when initializing a new tuning session.
    *   Corrected logic for determining starting OT level based on historical data and current VRAM.


# TensorTune v1.1.1 - Changelog

### Added
- **Context-Aware Library Status Warnings (CLI & GUI):** Startup messages for optional GPU monitoring libraries (PyADLX, PyZE, WMI, Metal) are now more relevant to the user's detected hardware, significantly reducing irrelevant log noise.
- **CLI: Option to Suppress Optional Library Warnings:** Users can now toggle a setting in the CLI to hide non-critical optional library status messages after the first successful run. Critical warnings (e.g., for Psutil, or PyNVML if NVIDIA GPU is targeted) will still display.
- **Detailed Setup Guides for Optional Libraries:**
    - `PYADLX_SETUP_GUIDE.md`: For manual setup of PyADLX (AMD GPUs on Windows).
    - `PYZE_SETUP_GUIDE.md`: For PyZE/`pyze-l0` (Intel Arc/Xe GPUs).
    - `WMI_SETUP_GUIDE.md`: For the WMI Python package (Windows fallback).
- Core error messages for these libraries now direct users to these new guides.
- `appdirs` added to `requirements.txt` and default generation in `tensortune_install.py` for consistent config/data path management.

### Changed
- **Improved Startup Logging (CLI & GUI):** Enhanced clarity and structure of initial diagnostic messages.
- **Updated Documentation:** `README.md`, `tensortune_examples.py`, and `tensortune_install.py` updated to reflect version 1.1.1, new features, and guide locations.
- **Core Error Messages:** Refined error messages in `tensortune_core.py` for PyADLX, PyZE, and WMI to be more informative and link to setup guides.
- **GUI Initialization Order:** Fine-tuned the `__init__` sequence in `tensortune_gui.py` for more robust startup and data availability for logging.

# TensorTune v1.1.0 - Changelog

## Overview
Version 1.1.0 brings significant enhancements across the board, focusing on improved GPU management, user experience in both GUI and CLI, more robust KCPP integration, and a major new "Set as Preferred" feature for tuning. This release also includes numerous bug fixes for increased stability and smoother operation.

---

## Core Improvements (`tensortune_core.py`)
-   **Enhanced GPU Detection & Error Handling:**
    -   More precise error reporting when GPU libraries (e.g., PyNVML, PyADLX) fail to initialize.
    -   Helpful hints on AMD Linux systems to install ROCm if `rocm-smi` is not found.
    -   SysFS fallback support for AMD and Intel GPUs on Linux, enabling basic GPU listing and VRAM detection even when primary tools are unavailable.
-   **KoboldCpp Capabilities:**
    -   Capabilities (e.g., CUDA support) are now cached per executable path to avoid redundant detection calls, speeding up subsequent startups.
    -   Clarification: KCPP capabilities detection primarily relies on parsing `koboldcpp_executable --help`. If a KCPP build incorrectly lists flags it doesn't support, TensorTune might optimistically detect them.
-   **Process Management:**
    -   Fixed a `TypeError` in `kill_processes_by_name` that occurred when process command line information (`cmdline`) was `None` or not iterable. This improves the robustness of the "Stop ANY KCPP Processes (Sweep)" feature.

---

## CLI Improvements (`tensortune_cli.py`)
-   **Robust User Interaction:**
    -   Redesigned input prompts to gracefully handle empty inputs, defaults, and user cancellations (Ctrl+C).
    -   Fixed an issue where the GGUF file selection prompt could immediately cancel with empty input.
-   **KoboldCpp Integration:**
    -   Validates and resolves the KCPP executable path using the same robust logic as the GUI.
    -   Automatically re-detects KCPP capabilities on first-time setup and when changing settings, updating GPU backend flags accordingly.
    -   Added interactive options to view/redetect KCPP capabilities and manage model-specific or global default arguments.
-   **Tuning Session Enhancements:**
    -   Correctly propagates tuning levels into KCPP launch commands for consistent GPU layer settings.
    -   Ensures VRAM availability is captured accurately right before direct launches (“Skip Tune” or “Launch Best”).
    -   Reconstructs “best remembered” launches using historical tuning levels and VRAM data for improved accuracy.
    -   Improved user stop tracking (e.g., Ctrl+C) during live monitoring sessions for smoother cancellation.
-   **Output and Display:**
    -   Resolved Rich progress bar errors by managing progress context correctly.
    -   Improved KCPP live output handling to integrate cleanly with Rich progress bars for a tidier console view.
-   **Configuration & Code:**
    -   Ensured more consistent updates to global CLI settings after changes.
    -   Improved handling of Python script paths for KCPP executables (e.g., `.py`, `.pyw`).

---

## GUI Improvements (`tensortune_gui.py`)
-   **New Feature: "Set as Preferred" Tuning Strategy:**
    -   After a successful monitored launch (VRAM OK/Tight), users can now mark that configuration as "preferred" for the current model.
    -   This preference is saved to the launch history with a distinct outcome (`_USER_MARKED_AS_BEST_GUI`).
    -   The "Launch Best Remembered Config" feature now prioritizes these user-marked preferred configurations.
-   **New Feature: Scrollable Auto-Tuning Session Page:**
    -   The "Auto-Tuning Session" page is now fully scrollable, ensuring all controls and information (post-monitoring choices, KCPP output log) are accessible regardless of window size.
-   **Enhanced User Experience & Layout:**
    -   “Browse” buttons for GGUF models now correctly start in user-configured default or last-used directories.
    -   First-time setup dialog initializes the last-used GGUF directory for a smoother initial browsing experience.
    -   Settings Tab: Collapsible sections (Global Arguments, KCPP Capabilities) now default to collapsed for a cleaner initial view.
    -   Settings Tab: Improved GPU selection dropdown layout and repositioned "GPU Status" label to prevent UI breakage on smaller windows.
    -   VRAM Display: Progress bars expanded for clarity, and GPU info/VRAM text moved below for better readability. Prevented duplicate "Manual Budget" prefix in VRAM messages.
    -   VRAM Monitoring: The "Launch & Monitor Output" button in the tuning view is now disabled if no GPU is detected or if it has no VRAM information (with a tooltip explaining why).
-   **Improved Process Control:**
    -   Added “Stop Tracked KCPP Processes” button to selectively terminate only TensorTune-launched KoboldCpp instances.
    -   “Stop Current Monitoring” button now available during monitoring sessions to safely abort operations, logging user stops in history.
    -   Button Color Tweak: “Stop Tracked KCPP” button recolored to a darker orange/brown for better visual distinction.
-   **Bug Fixes & Stability:**
    -   Resolved `NameError` on GUI close by moving closing logic into a class method (`_on_gui_close_requested`) and correctly registering it.
    -   Corrected `if __name__ == "__main__":` block positioning in `tensortune_gui.py`.
    -   Added widget existence checks (`hasattr`, `winfo_exists`) and safe `TabView` access to prevent errors during view switching or if UI elements are not fully initialized.
    -   Fixed `AttributeError` issues with CTkOptionMenu/Button callbacks, particularly for Python 3.13 compatibility, by consistently using lambda functions.
    -   Resolved `TypeError` in VRAM calculation post-monitoring if GPU/VRAM info was unavailable.
    -   Fixed `ValueError` (e.g., invalid `pady` on CTkFrame) that caused UI misbehavior, particularly affecting tuning menu returns.
    -   Improved GPU ID dropdown selection logic for correct default choices when GPU lists or configurations change.
    -   Enhanced configuration load/save logic for booleans, empty inputs, and model-specific argument dialogs.
    -   More reliable VRAM monitoring and display updates linked to the current configuration.
    -   Improved history view stability and performance (widget existence checks, limited initial entries).
    -   Corrected exit confirmation dialog parenting.
    -   General code style improvements and `IndentationError` fixes.

---

## Repository Management
-   Added `__pycache__/` and `*.pyc` to `.gitignore` and removed existing `__pycache__` folders from Git tracking to prevent committing Python bytecode.

---

## Summary
TensorTune 1.1.0 significantly refines the user experience, enhances GPU management capabilities, and introduces valuable features like "Set as Preferred" tuning. With numerous bug fixes and stability improvements, this version offers a more reliable and intuitive way to tune and launch your KoboldCpp models.

Thank you for using TensorTune! We appreciate your feedback and encourage you to report any issues or suggestions on GitHub.

---


## TensorTune v1.0.0 - Initial Release

This marks the first official release of TensorTune, rebranded from the KoboldCpp Smart Launcher, now with enhanced stability, GPU management, and user experience features.

🌟 Key Features & Enhancements:

    Rebranded to TensorTune: New application name and identity.

        Configuration and history database files will now be stored under "TensorTune" specific paths (e.g., in ~/.config/TensorTune or AppData). Note: Existing settings from "KoboldCpp Smart Launcher" will not be automatically migrated.

    Improved Configuration Management:

        Automatic Updates: Configuration files are now automatically updated to the latest version, adding missing keys with default values, reducing startup errors after updates.

        Robust Loading: Enhanced handling of existing configuration files.

        (CLI & GUI): Export and Import full launcher configurations (settings only, not history DB).

        (CLI & GUI): Option to reset all settings to factory defaults with automatic backup of current config.

    Enhanced GPU Management & VRAM Reporting (Core, GUI, CLI):

        Target Specific GPUs: Users can now select a specific GPU vendor (NVIDIA, AMD, Intel, Apple) and then a specific GPU ID (if multiple are present) for VRAM monitoring.

        VRAM Budget Override: Manually set a total VRAM budget for the launcher to use in its calculations, useful if auto-detection is problematic or for testing scenarios.

        Accurate VRAM Logging: Database now correctly logs actual hardware free VRAM before launch and actual hardware KCPP VRAM usage after successful loads.

        Informed Tuning: "VRAM Tight" vs. "VRAM OK" decisions during monitored tuning are based on the (potentially overridden) budgeted free VRAM.

    Refined Auto-Tuning Heuristics (Core, GUI, CLI):

        The initial Tensor Offload (OT) level suggestion now better considers:

            Estimated model VRAM needs vs. the launcher's (potentially overridden) VRAM budget.

            Relevant successful or failed historical launch configurations, matched against actual current hardware VRAM.

    GUI Enhancements (v1.4.1 basis):

        New "GPU Management & VRAM Override" section in Settings.

        VRAM display on Main and Tuning tabs reflects GPU selection and VRAM override.

        Improved initial directory logic for "Browse" dialogs in First-Time Setup.

        Fixed History tab display issues (TclError).

        Corrected monitored KCPP output handling (byte stream decoding).

    CLI Enhancements (aligned with GUI features):

        New "Launcher Settings" menu for comprehensive configuration.

        Respects Tkinter dialog preference for model selection if set.

        Displays "Last Monitored Success" details during tuning.

        Resolved AttributeError with Rich Progress display.

    Core Stability:

        Merged stdout/stderr from KoboldCpp process during monitoring for more reliable OOM/success detection.

        Improved AMD GPU detection on Linux and Windows.

        NVML resources are properly cleaned up on exit.

🐛 Bug Fixes (from recent development):

    Fixed configuration version mismatch messages appearing repeatedly.

    Resolved various minor UI and logic errors in both GUI and CLI.

    Ensured KoboldCpp capabilities are re-detected correctly after executable path changes.