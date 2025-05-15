TensorTune v1.0.0 - Initial Release

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