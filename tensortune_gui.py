import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
import os
import threading
import time
import subprocess
import sys
import re
import json
import shutil
import tensortune_core  # Assuming tensortune_core.py is in the same directory or accessible
import platform
from pathlib import Path
import webbrowser
import traceback
from typing import Optional # Added for type hinting


# --- ToolTip Class (Helper for rich tooltips) ---
class ToolTip(ctk.CTkToplevel):
    def __init__(self, widget, text, delay=500, wraplength=350):
        super().__init__(widget)
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self._scheduled_show = None

        self.withdraw()
        self.overrideredirect(True)

        current_mode = ctk.get_appearance_mode()
        if current_mode == "Light":
            bg_color = "gray90"
            text_color = "black"
        else:  # Dark or System (assuming dark-like for tooltip)
            bg_color = "gray20"
            text_color = "white"

        self.attributes("-alpha", 0.92)
        self.attributes("-topmost", True)

        self.label = ctk.CTkLabel(self, text=self.text, fg_color=bg_color, text_color=text_color,
                                  corner_radius=4, padx=6, pady=4, wraplength=self.wraplength)
        self.label.pack(ipadx=1)

        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, event=None):
        self._cancel_scheduled_show()
        self._scheduled_show = self.widget.after(self.delay, self._show_tip)

    def _on_leave(self, event=None):
        self._cancel_scheduled_show()
        self.withdraw()

    def _cancel_scheduled_show(self):
        if self._scheduled_show:
            self.widget.after_cancel(self._scheduled_show)
            self._scheduled_show = None

    def _show_tip(self):
        if not self.widget.winfo_exists():
            return

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 1

        self.update_idletasks()
        tip_width = self.winfo_width()
        tip_height = self.winfo_height()
        screen_width = self.widget.winfo_screenwidth()
        screen_height = self.widget.winfo_screenheight()

        if x + tip_width > screen_width:
            x = screen_width - tip_width - 5
        if x < 0:
            x = 5
        if y + tip_height > screen_height:
            y = self.widget.winfo_rooty() - tip_height - 5
        if y < 0:
            y = 5

        self.geometry(f"+{x}+{y}")
        self.deiconify()


class KoboldLauncherGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("TensorTune (GUI Edition v1.2.0)") 
        self.geometry("950x880")
        self.minsize(900, 780)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Block 1: Core Initialization & Basic Config/Attribute Setup ---
        core_init_results = tensortune_core.initialize_launcher()
        self.config = core_init_results["config"]
        # Ensure all necessary self.config keys have defaults if not present
        self.config.setdefault("gpu_selection_mode", tensortune_core.DEFAULT_CONFIG_TEMPLATE["gpu_selection_mode"])
        self.config.setdefault("selected_gpu_index", tensortune_core.DEFAULT_CONFIG_TEMPLATE["selected_gpu_index"])
        self.config.setdefault("override_vram_budget", tensortune_core.DEFAULT_CONFIG_TEMPLATE["override_vram_budget"])
        self.config.setdefault("manual_vram_total_mb", tensortune_core.DEFAULT_CONFIG_TEMPLATE["manual_vram_total_mb"])
        self.config.setdefault("default_gguf_dir", tensortune_core.DEFAULT_CONFIG_TEMPLATE["default_gguf_dir"])
        self.config.setdefault("last_used_gguf_dir", tensortune_core.DEFAULT_CONFIG_TEMPLATE["last_used_gguf_dir"])
        self.config.setdefault("model_specific_args", {}) # Ensure this exists

        self.system_info = core_init_results["system_info"]
        self.gpu_info = core_init_results["gpu_info"] # self.gpu_info is set here
        self.koboldcpp_capabilities = core_init_results.get("koboldcpp_capabilities", {})

        self.settings_dirty = False
        self.current_model_path = None
        self.process_running = False
        self.model_analysis_info = {}
        self.last_process = None
        self.db_path = self.config["db_file"] # Use loaded config for db_path
        self.default_model_dir = self.config.get("default_gguf_dir", "")
        self.koboldcpp_executable = self.config.get("koboldcpp_executable", "") # Use loaded config

        self.tuning_in_progress = False
        self.current_tuning_attempt_level = 0
        # ... (rest of your tuning and KCPP monitor attribute initializations) ...
        self.current_tuning_min_level = 0
        self.current_tuning_max_level = 0
        self.current_tuning_session_base_args = {}
        self.current_tuning_model_analysis = {}
        self.current_tuning_model_path = None
        self.level_of_last_monitored_run = 0
        self.current_command_list_for_db = []
        self.vram_at_decision_for_db = None # Note: This is distinct from the CLI's handling
        self.user_requested_stop_monitoring = False
        self.kcpp_monitor_thread = None
        self.kcpp_process_obj = None
        self.kcpp_success_event = threading.Event()
        self.kcpp_oom_event = threading.Event()
        self.kcpp_output_lines_shared = []
        self.monitor_start_time = 0.0
        self.MAX_KCPP_CONSOLE_LINES = 1000
        self.kcpp_console_line_count = 0
        self.last_approx_vram_used_kcpp_mb = None
        self.last_free_vram_after_load_mb = None
        self.last_successful_monitored_run_details_gui = None

        # CTk Variables for UI elements
        self.gpu_selection_mode_var = ctk.StringVar(value=self.config.get("gpu_selection_mode", "auto"))
        self.selected_gpu_id_display_var = ctk.StringVar(value="N/A (Auto-Detect)") # Default display
        self.override_vram_var = ctk.BooleanVar(value=self.config.get("override_vram_budget", False))
        self.manual_gpu_layers_var = ctk.BooleanVar(value=True) # True means "Auto" is checked by default
        self.manual_gpu_layers_entry_var = ctk.StringVar(value="")
        self.effective_gpu_layers_for_command = ctk.StringVar(value="auto") # For display and command building
        # UI Appearance
        appearance_mode = self.config.get("color_mode", "dark").lower()
        if appearance_mode not in ["dark", "light", "system"]:
            appearance_mode = "dark" # Fallback
        ctk.set_appearance_mode(appearance_mode)
        ctk.set_default_color_theme("blue")

        # --- Block 2: Setup UI Tabs and Load Settings into them ---
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.tab_main = self.tabview.add("Tune & Launch")
        self.tab_settings = self.tabview.add("Settings")
        self.tab_history = self.tabview.add("History")

        for tab_widget in [self.tab_main, self.tab_settings, self.tab_history]: # Renamed 'tab' to 'tab_widget' to avoid conflict
            tab_widget.grid_columnconfigure(0, weight=1)
            tab_widget.grid_rowconfigure(0, weight=1) # Usually row 0 or a specific content row

        self.setup_settings_tab() # Creates widgets in settings tab
        self.load_settings_from_config() # Populates settings tab widgets using self.config

        self.setup_main_tab()    # Creates widgets in main tab
        self.setup_history_tab() # Creates widgets in history tab & loads initial history

        self.tabview.set("Tune & Launch") # Set initial visible tab

        # --- Block 3: Start Background Threads ---
        threading.Thread(target=self.monitor_vram, daemon=True).start()

        # --- Block 4: Initial GUI Log Messages (General Status & Refined Library Status) ---
        self.log_to_console(f"TensorTune GUI Initialized. Core: {self.config.get('launcher_core_version', 'N/A')}")
        self.log_to_console(f"Config loaded: {core_init_results['config_message']}")
        if not core_init_results["db_success"]:
            self.log_to_console(f"DB Warning: {core_init_results['db_message']}", level="WARNING")
        self.log_to_console(f"Using DB at: {self.db_path}")
        self.log_to_console(f"Initial GPU Info: {self.gpu_info.get('message', 'N/A')}") 

        kcpp_caps_info = self.koboldcpp_capabilities
        if "error" in kcpp_caps_info:
            self.log_to_console(f"KCPP Caps Error: {kcpp_caps_info['error']}", level="ERROR")
        else:
            cuda_support = kcpp_caps_info.get('cuda', False)
            rocm_support = kcpp_caps_info.get('rocm', False)
            flash_attn_support = kcpp_caps_info.get('flash_attn', False)
            self.log_to_console(
                f"KCPP Caps: CUDA:{cuda_support}, ROCm:{rocm_support}, FlashAttn:{flash_attn_support}"
            )
        
        # Refined Library Status Logging
        self.log_to_console("Checking status of optional support libraries...")
        detected_gpu_type_str_gui = ""
        if self.gpu_info and self.gpu_info.get("success"): # self.gpu_info is already set
            detected_gpu_type_str_gui = self.gpu_info.get("type", "").lower()

        any_warnings_logged_gui = False
        
        if tensortune_core.appdirs_load_error_reason:
            self.log_to_console(f"Library Status - Appdirs: {tensortune_core.appdirs_load_error_reason}", level="WARNING")
            any_warnings_logged_gui = True
        
        if tensortune_core.psutil_load_error_reason:
            self.log_to_console(f"Library Status - Psutil: {tensortune_core.psutil_load_error_reason} (Impacts auto-threads, process management)", level="ERROR")
            any_warnings_logged_gui = True

        if self.config.get("gpu_detection", {}).get("nvidia", True):
            if tensortune_core.pynvml_load_error_reason:
                self.log_to_console(f"Library Status - PyNVML (NVIDIA): {tensortune_core.pynvml_load_error_reason} (Required for NVIDIA VRAM monitoring)", level="ERROR")
                any_warnings_logged_gui = True
        
        if self.config.get("gpu_detection", {}).get("amd", True) and platform.system() == "win32":
            if tensortune_core.pyadlx_load_error_reason:
                if detected_gpu_type_str_gui == "amd" or not detected_gpu_type_str_gui or detected_gpu_type_str_gui == "unknown/none_auto":
                    self.log_to_console(f"Library Status - PyADLX (AMD): {tensortune_core.pyadlx_load_error_reason}", level="WARNING")
                    any_warnings_logged_gui = True
            if tensortune_core.wmi_load_error_reason:
                if detected_gpu_type_str_gui == "amd" or not detected_gpu_type_str_gui or detected_gpu_type_str_gui == "unknown/none_auto":
                    self.log_to_console(f"Library Status - WMI (Windows Fallback for AMD/Other): {tensortune_core.wmi_load_error_reason}", level="WARNING")
                    any_warnings_logged_gui = True
        
        if self.config.get("gpu_detection", {}).get("intel", True):
            if tensortune_core.pyze_load_error_reason:
                if "intel" in detected_gpu_type_str_gui or not detected_gpu_type_str_gui or detected_gpu_type_str_gui == "unknown/none_auto":
                    self.log_to_console(f"Library Status - PyZE (Intel): {tensortune_core.pyze_load_error_reason}", level="WARNING")
                    any_warnings_logged_gui = True

        if self.config.get("gpu_detection", {}).get("apple", True) and platform.system() == "darwin":
            if tensortune_core.metal_load_error_reason:
                if "apple" in detected_gpu_type_str_gui or "metal" in detected_gpu_type_str_gui or not detected_gpu_type_str_gui or detected_gpu_type_str_gui == "unknown/none_auto":
                    self.log_to_console(f"Library Status - Metal (Apple): {tensortune_core.metal_load_error_reason}", level="WARNING")
                    any_warnings_logged_gui = True
        
        if not any_warnings_logged_gui: # Only print this if no specific warnings were logged above
             self.log_to_console("All relevant optional and critical support libraries seem to be available or not applicable for your setup.", level="SUCCESS")
        
        # --- Block 5: Final UI Initialization Steps & Scheduled Calls ---
        self.check_koboldcpp_executable() 
        self._show_model_selection_view() # Ensures the correct initial view is displayed
        
        # These are scheduled to run after the main event loop starts, allowing the GUI to draw first
        self.after(100, self._run_first_time_setup_if_needed)
        self.after(200, self._populate_gpu_id_dropdown_on_startup) # Depends on settings being loaded

        self.update_save_button_state() # Initial state of save button

        # Register the window close handler
        self.protocol("WM_DELETE_WINDOW", self._on_gui_close_requested)

    def _safe_focus_set(self):
        """Safely set focus to the main window with error handling."""
        try:
            if self.winfo_exists():
                self.focus_force()  # Use focus_force instead of focus_set
        except Exception as e:
            print(f"Focus error (non-critical): {str(e)}")

    def _safe_widget_exists(self, widget):
        """Safely check if a widget exists."""
        try:
            return widget is not None and hasattr(widget, 'winfo_exists') and widget.winfo_exists()
        except Exception:
            return False

    def _safe_dialog_close(self, dialog):
        """Safely close a dialog window with proper error handling."""
        if dialog is None:
            return
            
        try:
            if hasattr(dialog, 'winfo_exists') and dialog.winfo_exists():
                if hasattr(dialog, 'grab_release'):
                    dialog.grab_release()
                dialog.destroy()
        except Exception as e:
            self.log_to_console(f"Dialog closing error (non-critical): {str(e)}") 



        # Set up global exception handling
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        # Log the error
        import traceback
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        self.log_to_console(f"UNCAUGHT EXCEPTION: {error_msg}", level="ERROR")
        
        # Continue with normal exception handling
        import sys
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    import sys
    sys.excepthook = global_exception_handler

    def _on_gui_close_requested(self):
        # Check if any KCPP process might be running
        kcpp_monitored_is_running = hasattr(self,'kcpp_process_obj') and \
                                    self.kcpp_process_obj and \
                                    self.kcpp_process_obj.poll() is None
        kcpp_direct_launch_is_running = hasattr(self, 'last_process') and \
                                        self.last_process and \
                                        self.last_process.poll() is None
        
        any_kcpp_potentially_running = (hasattr(self, 'process_running') and self.process_running) or \
                                       kcpp_monitored_is_running or \
                                       kcpp_direct_launch_is_running

        if any_kcpp_potentially_running:
            confirm_exit = messagebox.askyesno(
                "Exit Confirmation",
                "A KoboldCpp process launched or monitored by this tool might still be running.\n"
                "Do you want to stop all related KoboldCpp processes and exit the launcher?",
                icon=messagebox.QUESTION,
                parent=self # Ensure messagebox is child of self (the app window)
            )
            if confirm_exit:
                self.log_to_console("User chose to stop processes and exit.")
                self.stop_all_kcpp_processes_forcefully() # Attempt to clean up
                time.sleep(0.2) # Brief pause for processes to terminate
                self.destroy()
            else:
                self.log_to_console("User chose not to stop processes; exit cancelled.")
                # return # Do not destroy if user cancels
        else: # No known processes running
            self.destroy()

    def _create_collapsible_frame(self, parent, title_text, initially_collapsed=False):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.content_visible = not initially_collapsed

        header_frame = ctk.CTkFrame(frame)
        header_frame.pack(fill="x", pady=(0, 1))
        header_frame.grid_columnconfigure(1, weight=1)

        initial_toggle_text = "►" if initially_collapsed else "▼"
        toggle_button = ctk.CTkButton(header_frame, text=initial_toggle_text, width=25, height=25,
                                      command=lambda: self._toggle_collapsible_content(frame, toggle_button))
        toggle_button.grid(row=0, column=0, padx=5, pady=2)

        title_label = ctk.CTkLabel(header_frame, text=title_text, font=ctk.CTkFont(size=14, weight="bold"), anchor="w")
        title_label.grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        frame.content_frame = ctk.CTkFrame(frame, border_width=1)
        if not initially_collapsed:
            frame.content_frame.pack(fill="x", expand=True, padx=5, pady=(0, 5))
        return frame

    def _toggle_collapsible_content(self, main_collapsible_frame, button_widget):
        main_collapsible_frame.content_visible = not main_collapsible_frame.content_visible
        if main_collapsible_frame.content_visible:
            main_collapsible_frame.content_frame.pack(fill="x", expand=True, padx=5, pady=(0, 5))
            button_widget.configure(text="▼")
        else:
            main_collapsible_frame.content_frame.pack_forget()
            button_widget.configure(text="►")

    def browse_executable(self):
        current_path_in_entry = ""
        if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
            current_path_in_entry = self.exe_path_entry.get()

        initial_dir_browse = os.getcwd()
        if current_path_in_entry:
            dir_name = os.path.dirname(current_path_in_entry)
            if os.path.isdir(dir_name) and dir_name:
                initial_dir_browse = dir_name
            elif os.path.isdir(current_path_in_entry):
                initial_dir_browse = current_path_in_entry

        if platform.system() == "Windows":
            filetypes = [("Executables", "*.exe"), ("Python scripts", "*.py"), ("All files", "*.*")]
        elif platform.system() == "Darwin":
            filetypes = [("Applications/Scripts", "*.app *.sh *.py"), ("All files", "*.*")]
        else:
            filetypes = [("Scripts/Executables", "*.sh *.py"), ("Any Executable", "*"), ("All files", "*.*")]

        filepath_selected = filedialog.askopenfilename(
            title="Select KoboldCpp Executable or Script",
            initialdir=initial_dir_browse,
            filetypes=filetypes,
            parent=self
        )
        if filepath_selected:
            if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
                self.exe_path_entry.delete(0, "end")
                self.exe_path_entry.insert(0, filepath_selected)
                self.mark_settings_dirty()
                self.check_koboldcpp_executable()
                self.update_kcpp_capabilities_display(re_detect=True)
            self.log_to_console(f"KoboldCpp executable path selected: {filepath_selected}")

    def mark_settings_dirty(self, *args):
        if not self.settings_dirty:
            self.settings_dirty = True
            self.update_save_button_state()

    def update_save_button_state(self):
        if hasattr(self, 'btn_save_settings_tab') and self.btn_save_settings_tab.winfo_exists():
            new_text = "Save Settings*" if self.settings_dirty else "Save Settings"
            fg_color = "seagreen" if self.settings_dirty else ctk.ThemeManager.theme["CTkButton"]["fg_color"]
            self.btn_save_settings_tab.configure(text=new_text, fg_color=fg_color)

          
    def setup_main_tab(self):
            self.tab_main.grid_rowconfigure(0, weight=1)

            self.model_selection_frame = ctk.CTkFrame(self.tab_main, fg_color="transparent")
            self.model_selection_frame.grid_columnconfigure(0, weight=1)
            self.model_selection_frame.grid_rowconfigure(5, weight=1) # Console row uses remaining space

            title_label = ctk.CTkLabel(self.model_selection_frame, text="TensorTune Model Launcher", font=ctk.CTkFont(size=20, weight="bold"))
            title_label.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 15), sticky="n")

            model_controls_frame = ctk.CTkFrame(self.model_selection_frame)
            model_controls_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
            model_controls_frame.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(model_controls_frame, text="GGUF Model:").grid(row=0, column=0, padx=(10, 5), pady=10, sticky="w")
            self.model_path_entry = ctk.CTkEntry(model_controls_frame, width=400)
            self.model_path_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
            ToolTip(self.model_path_entry, "Path to the selected .gguf model file.")
            btn_browse_model = ctk.CTkButton(model_controls_frame, text="Browse", command=lambda: self.browse_model(), width=80)
            btn_browse_model.grid(row=0, column=2, padx=(5, 10), pady=10, sticky="e")
            ToolTip(btn_browse_model, "Select a .gguf model file to load.")

            self.model_info_label = ctk.CTkLabel(model_controls_frame, text="No model selected. Analysis includes: Type, Size, Quant, Layers, Est. VRAM.", justify="left", wraplength=650, font=("Segoe UI", 11))
            self.model_info_label.grid(row=1, column=0, columnspan=3, padx=10, pady=(0, 10), sticky="w")

            vram_frame = ctk.CTkFrame(self.model_selection_frame)
            vram_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
            vram_frame.grid_columnconfigure(0, weight=0)
            vram_frame.grid_columnconfigure(1, weight=1)
            vram_frame.grid_columnconfigure(2, weight=0)
            vram_frame.grid_rowconfigure(1, weight=0)

            ctk.CTkLabel(vram_frame, text="GPU Status:").grid(row=0, column=0, padx=(10, 5), pady=(5,0), sticky="w")
            self.vram_progress = ctk.CTkProgressBar(vram_frame, height=18)
            self.vram_progress.grid(row=0, column=1, padx=5, pady=(5,0), sticky="ew")
            self.vram_progress.set(0)

            btn_refresh_vram = ctk.CTkButton(vram_frame, text="Refresh", width=60, command=lambda: self.refresh_vram())
            btn_refresh_vram.grid(row=0, column=2, rowspan=2, padx=(5, 10), pady=5, sticky="ns")
            ToolTip(btn_refresh_vram, "Manually refresh GPU VRAM information based on current GPU selection settings.")

            self.vram_text = ctk.CTkLabel(vram_frame, text="Scanning...", font=("Segoe UI", 10), anchor="w")
            self.vram_text.grid(row=1, column=0, columnspan=2, padx=(10,5), pady=(0,5), sticky="ew")

            launch_buttons_frame = ctk.CTkFrame(self.model_selection_frame)
            launch_buttons_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
            launch_buttons_frame.grid_columnconfigure((0, 1, 2), weight=1)
            btn_start_tune = ctk.CTkButton(launch_buttons_frame, text="Start Auto-Tune / Use OT Strategy", command=lambda: self.start_tuning_session(), height=35, fg_color="seagreen", hover_color="darkgreen")
            btn_start_tune.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
            ToolTip(btn_start_tune, "Begin an interactive session to find optimal Tensor Offload (OT)\nsettings for the selected model and current VRAM.")
            btn_launch_best = ctk.CTkButton(launch_buttons_frame, text="Launch Best Remembered Config", command=lambda: self.launch_best_remembered(), height=35, fg_color="cornflowerblue", hover_color="royalblue")
            btn_launch_best.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            ToolTip(btn_launch_best, "Load the model using the most successful configuration\npreviously found in the launch history for this model and similar VRAM.")
            btn_launch_direct = ctk.CTkButton(launch_buttons_frame, text="Direct Launch (Settings Defaults)", command=lambda: self.launch_direct_defaults(), height=35)
            btn_launch_direct.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
            ToolTip(btn_launch_direct, "Launch the model directly using your current global default\nsettings combined with any model-specific settings, without auto-tuning.")

            stop_button_frame = ctk.CTkFrame(self.model_selection_frame)
            stop_button_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=(0, 5), sticky="ew")
            stop_button_frame.grid_columnconfigure((0, 1), weight=1)

            self.btn_stop_tracked_kcpp = ctk.CTkButton(stop_button_frame, text="Stop Tracked KCPP Process(es)", command=lambda: self.stop_tracked_kcpp_processes(), height=35, fg_color="saddlebrown", hover_color="sienna")
            self.btn_stop_tracked_kcpp.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
            ToolTip(self.btn_stop_tracked_kcpp, "Attempt to terminate KoboldCpp processes that were\nspecifically launched or monitored by this TensorTune session (Direct Launch or Tuning Monitor).")

            btn_stop_all = ctk.CTkButton(stop_button_frame, text="Stop ANY KCPP Processes (Sweep)", command=lambda: self.stop_all_kcpp_processes_forcefully(), height=35, fg_color="firebrick", hover_color="darkred")
            btn_stop_all.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            ToolTip(btn_stop_all, "Attempt to forcefully terminate any running KoboldCpp processes\nthat match the configured executable name or known Python script patterns.\nUSE WITH CAUTION.")

            console_frame_ms = ctk.CTkFrame(self.model_selection_frame)
            console_frame_ms.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
            console_frame_ms.grid_columnconfigure(0, weight=1)
            console_frame_ms.grid_rowconfigure(1, weight=1)
            ctk.CTkLabel(console_frame_ms, text="Launcher Log:").grid(row=0, column=0, padx=10, pady=(5, 0), sticky="w")
            self.console = ctk.CTkTextbox(console_frame_ms, height=120, wrap="word")
            self.console.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="nsew")
            self.console.configure(state="disabled")

            # --- Start of Tuning Mode View Setup ---
            self.tuning_view_container_frame = ctk.CTkFrame(self.tab_main, fg_color="transparent")
            self.tuning_view_container_frame.grid_columnconfigure(0, weight=1)
            self.tuning_view_container_frame.grid_rowconfigure(0, weight=1)

            self.tuning_mode_scrollable_content_frame = ctk.CTkScrollableFrame(self.tuning_view_container_frame)
            self.tuning_mode_scrollable_content_frame.pack(fill="both", expand=True, padx=0, pady=0)
            self.tuning_mode_scrollable_content_frame.grid_columnconfigure(0, weight=1)
            # Note: The row for kcpp_output_console_frame will be 12, so rowconfigure will be updated later.

            current_row_idx_tuning = 0

            tuning_title_label = ctk.CTkLabel(self.tuning_mode_scrollable_content_frame, text="Auto-Tuning Session", font=ctk.CTkFont(size=18, weight="bold"))
            tuning_title_label.grid(row=current_row_idx_tuning, column=0, padx=10, pady=(10, 0), sticky="n"); current_row_idx_tuning += 1
            
            self.tuning_model_name_label = ctk.CTkLabel(self.tuning_mode_scrollable_content_frame, text="Model: N/A", font=ctk.CTkFont(size=14))
            self.tuning_model_name_label.grid(row=current_row_idx_tuning, column=0, padx=10, pady=(0, 5), sticky="n"); current_row_idx_tuning += 1

            self.tuning_view_vram_frame = ctk.CTkFrame(self.tuning_mode_scrollable_content_frame)
            self.tuning_view_vram_frame.grid(row=current_row_idx_tuning, column=0, padx=10, pady=(0, 5), sticky="ew"); current_row_idx_tuning += 1
            self.tuning_view_vram_frame.grid_columnconfigure(0, weight=0)
            self.tuning_view_vram_frame.grid_columnconfigure(1, weight=1)
            self.tuning_view_vram_frame.grid_columnconfigure(2, weight=0)
            self.tuning_view_vram_frame.grid_rowconfigure(1, weight=0)
            ctk.CTkLabel(self.tuning_view_vram_frame, text="GPU Status:").grid(row=0, column=0, padx=(10, 5), pady=(5,0), sticky="w")
            self.tuning_view_vram_progress = ctk.CTkProgressBar(self.tuning_view_vram_frame, height=18)
            self.tuning_view_vram_progress.grid(row=0, column=1, padx=5, pady=(5,0), sticky="ew")
            self.tuning_view_vram_progress.set(0)
            btn_refresh_vram_tuning = ctk.CTkButton(self.tuning_view_vram_frame, text="Refresh", width=60, command=lambda: self.refresh_vram())
            btn_refresh_vram_tuning.grid(row=0, column=2, rowspan=2, padx=(5, 10), pady=5, sticky="ns")
            ToolTip(btn_refresh_vram_tuning, "Manually refresh GPU VRAM information.")
            self.tuning_view_vram_text = ctk.CTkLabel(self.tuning_view_vram_frame, text="Scanning...", font=("Segoe UI", 10), anchor="w")
            self.tuning_view_vram_text.grid(row=1, column=0, columnspan=2, padx=(10,5), pady=(0,5), sticky="ew")

            self.last_run_info_frame_tuning = ctk.CTkFrame(self.tuning_mode_scrollable_content_frame)
            self.last_run_info_frame_tuning.grid(row=current_row_idx_tuning, column=0, padx=10, pady=2, sticky="ew"); current_row_idx_tuning += 1
            self.last_run_info_frame_tuning.grid_columnconfigure(0, weight=1)
            self.tuning_last_run_info_label = ctk.CTkLabel(self.last_run_info_frame_tuning, text="Last Monitored Result: None yet in this session.", justify="left", font=ctk.CTkFont(size=11), text_color="gray", anchor="w")
            self.tuning_last_run_info_label.grid(row=0, column=0, padx=5, pady=2, sticky="ew")

            ot_strategy_display_frame = ctk.CTkFrame(self.tuning_mode_scrollable_content_frame)
            ot_strategy_display_frame.grid(row=current_row_idx_tuning, column=0, padx=10, pady=2, sticky="ew"); current_row_idx_tuning += 1
            ot_strategy_display_frame.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(ot_strategy_display_frame, text="Current OT Strategy:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=5, pady=2, sticky="w")
            self.tuning_ot_level_frame = ctk.CTkFrame(ot_strategy_display_frame, fg_color="transparent")
            self.tuning_ot_level_frame.grid(row=1, column=0, padx=0, pady=1, sticky="ew")
            self.tuning_ot_level_label = ctk.CTkLabel(self.tuning_ot_level_frame, text="Level: N/A", justify="left", anchor="w")
            self.tuning_ot_level_label.pack(side="left", padx=(10, 5))
            self.tuning_ot_qualitative_desc_label = ctk.CTkLabel(self.tuning_ot_level_frame, text="(GPU Biased/Balanced/CPU Biased)", justify="left", anchor="w", font=ctk.CTkFont(size=10, slant="italic"))
            self.tuning_ot_qualitative_desc_label.pack(side="left", padx=(0, 10))
            self.tuning_ot_range_label = ctk.CTkLabel(ot_strategy_display_frame, text="Range: N/A", justify="left", anchor="w")
            self.tuning_ot_range_label.grid(row=2, column=0, padx=10, pady=1, sticky="ew")
            self.tuning_ot_strategy_desc_label = ctk.CTkLabel(ot_strategy_display_frame, text="Strategy: N/A", justify="left", wraplength=650, anchor="w")
            self.tuning_ot_strategy_desc_label.grid(row=3, column=0, padx=10, pady=1, sticky="ew")
            self.tuning_ot_regex_label = ctk.CTkLabel(ot_strategy_display_frame, text="Regex: N/A", justify="left", wraplength=650, font=("Courier New", 10), anchor="w")
            self.tuning_ot_regex_label.grid(row=4, column=0, padx=10, pady=1, sticky="ew")
            self.tuning_gpu_layers_label = ctk.CTkLabel(ot_strategy_display_frame, text="GPU Layers: N/A", justify="left", anchor="w")
            self.tuning_gpu_layers_label.grid(row=5, column=0, padx=10, pady=1, sticky="ew")

            # --- Manual GPU Layers Control Frame ---
            self.manual_gpu_layers_control_frame = ctk.CTkFrame(self.tuning_mode_scrollable_content_frame, fg_color="transparent")
            self.manual_gpu_layers_control_frame.grid(row=current_row_idx_tuning, column=0, padx=10, pady=(1,2), sticky="ew")
            current_row_idx_tuning += 1
            
            self.auto_gpu_layers_checkbox = ctk.CTkCheckBox(self.manual_gpu_layers_control_frame,
                                                             text="Auto GPU Layers (Recommended)",
                                                             variable=self.manual_gpu_layers_var,
                                                             command=self._on_auto_gpu_layers_toggle)
            self.auto_gpu_layers_checkbox.pack(side="left", padx=(5,5), pady=5)
            ToolTip(self.auto_gpu_layers_checkbox, "Check to let TensorTune automatically determine GPU layers based on OT Level.\nUncheck to manually specify GPU layers.")
            
            self.manual_gpu_layers_label = ctk.CTkLabel(self.manual_gpu_layers_control_frame, text="Manual Layers:")
            self.manual_gpu_layers_label.pack(side="left", padx=(10,2), pady=5)
            
            self.manual_gpu_layers_entry = ctk.CTkEntry(self.manual_gpu_layers_control_frame,
                                                         textvariable=self.manual_gpu_layers_entry_var,
                                                         width=60)
            self.manual_gpu_layers_entry.pack(side="left", padx=(0,5), pady=5)
            ToolTip(self.manual_gpu_layers_entry, "Enter a specific number of GPU layers (e.g., 30, 999 for all possible),\nor 'auto' to revert to automatic calculation for this field if checkbox is also auto.")
            
            
            self._on_auto_gpu_layers_toggle()  # Call to set initial state of entry
            self._setup_manual_gpu_layers_controls()  # Set up the trace for real-time updates

            # --- Proposed Command Frame --- (Fix for Issue 4a starts here)
            proposed_command_frame = ctk.CTkFrame(self.tuning_mode_scrollable_content_frame)
            proposed_command_frame.grid(row=current_row_idx_tuning, column=0, padx=10, pady=2, sticky="ew"); current_row_idx_tuning += 1
            proposed_command_frame.grid_columnconfigure(0, weight=1) # For the content inside
            # proposed_command_frame.grid_rowconfigure(1, weight=1) # If textbox should expand vertically

            command_label_btn_frame = ctk.CTkFrame(proposed_command_frame) 
            command_label_btn_frame.grid(row=0, column=0, sticky="ew", pady=(0,2)) # Use grid
            # command_label_btn_frame.grid_columnconfigure(0, weight=1) # To make label expand vs button
            # For simple left/right packing within this small frame, pack is okay:
            ctk.CTkLabel(command_label_btn_frame, text="Proposed Command:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5, pady=0)
            self.btn_copy_command = ctk.CTkButton(command_label_btn_frame, text="Copy", width=60, height=24, command=lambda: self.copy_proposed_command())
            self.btn_copy_command.pack(side="right", padx=5, pady=0)
            ToolTip(self.btn_copy_command, "Copy the full proposed launch command to the clipboard.")
            
            self.tuning_proposed_command_text = ctk.CTkTextbox(proposed_command_frame, height=100, wrap="word", font=("Courier New", 10))
            self.tuning_proposed_command_text.grid(row=1, column=0, sticky="ew", padx=5, pady=0) # Use grid
            self.tuning_proposed_command_text.configure(state="disabled")
            # --- Proposed Command Frame --- (Fix for Issue 4a ends here)


            self.tuning_actions_primary_frame = ctk.CTkFrame(self.tuning_mode_scrollable_content_frame)
            self.tuning_actions_primary_frame.grid(row=current_row_idx_tuning, column=0, padx=10, pady=(5, 2), sticky="ew"); current_row_idx_tuning += 1
            self.tuning_actions_primary_frame.grid_columnconfigure((0, 1), weight=1)
            self.btn_tune_launch_monitor = ctk.CTkButton(self.tuning_actions_primary_frame, text="Launch & Monitor Output", command=lambda: self.launch_and_monitor_for_tuning(), height=35, fg_color="seagreen", hover_color="darkgreen")
            self.btn_tune_launch_monitor.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
            ToolTip(self.btn_tune_launch_monitor, "Launch KoboldCpp with the current OT strategy and monitor its output for success or errors (e.g., OOM).")
            self.btn_tune_skip_launch_direct = ctk.CTkButton(self.tuning_actions_primary_frame, text="Skip Tune & Launch This Config", command=lambda: self.skip_tune_and_launch_direct(), height=35)
            self.btn_tune_skip_launch_direct.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            ToolTip(self.btn_tune_skip_launch_direct, "Immediately launch KoboldCpp for use with the current OT strategy and base arguments,\nwithout further monitoring or tuning steps.")

            self.tuning_stop_monitor_frame = ctk.CTkFrame(self.tuning_mode_scrollable_content_frame, fg_color="transparent")
            self.tuning_stop_monitor_frame.grid(row=current_row_idx_tuning, column=0, padx=10, pady=(0, 2), sticky="ew"); current_row_idx_tuning += 1
            self.tuning_stop_monitor_frame.grid_columnconfigure(0, weight=1)
            self.btn_stop_monitoring = ctk.CTkButton(self.tuning_stop_monitor_frame, text="Stop Current Monitoring", command=lambda: self._stop_current_monitoring_action(), height=30, fg_color="firebrick", hover_color="darkred")
            ToolTip(self.btn_stop_monitoring, "Forcefully stop the KoboldCpp process currently being monitored for tuning.")
            self.tuning_stop_monitor_frame.grid_remove() # Initially hidden

            self.tuning_actions_secondary_frame = ctk.CTkFrame(self.tuning_mode_scrollable_content_frame)
            self.tuning_actions_secondary_frame.grid(row=current_row_idx_tuning, column=0, padx=10, pady=0, sticky="ew"); current_row_idx_tuning += 1
            self.tuning_actions_secondary_frame.grid_columnconfigure((0, 1), weight=1)
            self.btn_tune_more_gpu = ctk.CTkButton(self.tuning_actions_secondary_frame, text="More GPU (↓ Level)", command=lambda: self.adjust_ot_level(-1))
            self.btn_tune_more_gpu.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
            ToolTip(self.btn_tune_more_gpu, "Adjust the OT strategy to offload more layers/tensors to the GPU (decreases OT level).")
            self.btn_tune_more_cpu = ctk.CTkButton(self.tuning_actions_secondary_frame, text="More CPU (↑ Level)", command=lambda: self.adjust_ot_level(1))
            self.btn_tune_more_cpu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            ToolTip(self.btn_tune_more_cpu, "Adjust the OT strategy to offload more layers/tensors to the CPU (increases OT level),\nreducing VRAM usage.")

            self.tuning_edit_args_buttons_frame = ctk.CTkFrame(self.tuning_mode_scrollable_content_frame)
            self.tuning_edit_args_buttons_frame.grid(row=current_row_idx_tuning, column=0, padx=10, pady=2, sticky="ew"); current_row_idx_tuning += 1
            self.tuning_edit_args_buttons_frame.grid_columnconfigure((0, 1), weight=1)
            self.btn_tune_edit_args = ctk.CTkButton(self.tuning_edit_args_buttons_frame, text="Edit Base Args (This Session)", command=lambda: self.edit_base_args_for_tuning_session())
            self.btn_tune_edit_args.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
            ToolTip(self.btn_tune_edit_args, "Modify the base KoboldCpp arguments (e.g., context size, threads)\nfor this tuning session only.")
            self.btn_tune_edit_model_perm_args = ctk.CTkButton(self.tuning_edit_args_buttons_frame, text="Edit Base Args (Permanent for This Model)", command=lambda: self.edit_permanent_model_args())
            self.btn_tune_edit_model_perm_args.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            ToolTip(self.btn_tune_edit_model_perm_args, "Modify and save the base KoboldCpp arguments specifically for the current model.")

            self.tuning_actions_navigation_frame = ctk.CTkFrame(self.tuning_mode_scrollable_content_frame)
            self.tuning_actions_navigation_frame.grid(row=current_row_idx_tuning, column=0, padx=10, pady=2, sticky="ew"); current_row_idx_tuning += 1
            self.tuning_actions_navigation_frame.grid_columnconfigure((0, 1, 2), weight=1)
            self.btn_tune_new_gguf = ctk.CTkButton(self.tuning_actions_navigation_frame, text="New GGUF Model", command=lambda: self.select_new_gguf_during_tuning())
            self.btn_tune_new_gguf.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
            ToolTip(self.btn_tune_new_gguf, "End the current tuning session and return to the model selection screen.")
            self.btn_tune_history = ctk.CTkButton(self.tuning_actions_navigation_frame, text="View History (This Model)", command=lambda: self.view_history_for_current_model())
            self.btn_tune_history.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            ToolTip(self.btn_tune_history, "Open the History tab, filtered to show launch records for the current model.")
            self.btn_tune_quit_tuning = ctk.CTkButton(self.tuning_actions_navigation_frame, text="End Tuning Session", command=lambda: self.end_tuning_session(), fg_color="firebrick", hover_color="darkred")
            self.btn_tune_quit_tuning.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
            ToolTip(self.btn_tune_quit_tuning, "Stop the current tuning session and return to the main model selection view.")

            self.kcpp_output_console_frame = ctk.CTkFrame(self.tuning_mode_scrollable_content_frame)
            self.kcpp_output_console_frame.grid(row=current_row_idx_tuning, column=0, padx=10, pady=(5, 10), sticky="nsew"); current_row_idx_tuning += 1
            self.kcpp_output_console_frame.grid_columnconfigure(0, weight=1)
            self.kcpp_output_console_frame.grid_rowconfigure(1, weight=1) # This makes textbox expand
            ctk.CTkLabel(self.kcpp_output_console_frame, text="KoboldCpp Output (during monitoring):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
            self.kcpp_live_output_text = ctk.CTkTextbox(self.kcpp_output_console_frame, wrap="char", font=("Segoe UI", 10)) # Height will be determined by row weight
            self.kcpp_live_output_text.grid(row=1, column=0, padx=5, pady=2, sticky="nsew")
            self.kcpp_live_output_text.configure(state="disabled")

            # Configure the last row (kcpp_output_console_frame's row) to take remaining space
            self.tuning_mode_scrollable_content_frame.grid_rowconfigure(current_row_idx_tuning -1, weight=1)
            # --- End of Tuning Mode View Setup ---

    

    def copy_proposed_command(self):
        if hasattr(self, 'tuning_proposed_command_text') and self.tuning_proposed_command_text.winfo_exists():
            command_text = self.tuning_proposed_command_text.get("1.0", "end-1c").strip()
            if command_text:
                try:
                    self.clipboard_clear()
                    self.clipboard_append(command_text)
                    self.log_to_console("Proposed command copied to clipboard.")
                    if hasattr(self, 'btn_copy_command') and self.btn_copy_command.winfo_exists():
                        original_text = self.btn_copy_command.cget("text")
                        self.btn_copy_command.configure(text="Copied!")
                        self.btn_copy_command.after(2000, lambda: self.btn_copy_command.configure(text=original_text) if hasattr(self, 'btn_copy_command') and self.btn_copy_command.winfo_exists() else None)
                except Exception as e:
                    self.log_to_console(f"Error copying to clipboard: {e}")
                    messagebox.showerror("Clipboard Error", f"Could not copy to clipboard:\n{e}", parent=self)
            else:
                self.log_to_console("No command text to copy.")

    def _populate_gpu_id_dropdown_on_startup(self):
        if hasattr(self, 'gpu_id_optionmenu') and self.gpu_id_optionmenu.winfo_exists():
            self._populate_gpu_id_dropdown()
        else:
            self.after(500, self._populate_gpu_id_dropdown_on_startup)

    def setup_settings_tab(self):
        sf_parent = ctk.CTkFrame(self.tab_settings, fg_color="transparent")
        sf_parent.pack(fill="both", expand=True)
        sf = ctk.CTkScrollableFrame(sf_parent)
        sf.pack(fill="both", expand=True, padx=0, pady=0)

        ui_theme_frame = ctk.CTkFrame(sf)
        ui_theme_frame.pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkLabel(ui_theme_frame, text="UI Theme:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=(5, 10), pady=10, sticky="w")
        self.theme_var = ctk.StringVar(value=self.config.get("color_mode", "dark").lower())
        theme_option_menu = ctk.CTkOptionMenu(ui_theme_frame, values=["dark", "light", "system"], variable=self.theme_var, command=lambda selected_value: self.change_theme(selected_value))
        theme_option_menu.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.theme_var.trace_add("write", self.mark_settings_dirty)
        ToolTip(theme_option_menu, "Select the UI appearance mode (Dark, Light, or System default).")

        exe_path_frame = ctk.CTkFrame(sf)
        exe_path_frame.pack(fill="x", padx=10, pady=5)
        exe_path_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(exe_path_frame, text="KoboldCpp Executable:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=(5, 10), pady=10, sticky="w")
        self.exe_path_entry = ctk.CTkEntry(exe_path_frame, width=400)
        self.exe_path_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.exe_path_entry.bind("<KeyRelease>", self.mark_settings_dirty)
        ToolTip(self.exe_path_entry, "Full path to your KoboldCpp executable or Python script (e.g., koboldcpp.exe, ./koboldcpp, /path/to/koboldcpp.py).")
        btn_browse_exe = ctk.CTkButton(exe_path_frame, text="Browse", command=lambda: self.browse_executable(), width=80)
        btn_browse_exe.grid(row=0, column=2, padx=10, pady=10, sticky="e")
        ToolTip(btn_browse_exe, "Select your KoboldCpp executable file (e.g., koboldcpp.exe or a .py script).")

        kcpp_caps_collapsible_frame = self._create_collapsible_frame(sf, "Detected KoboldCpp Capabilities", initially_collapsed=True)
        kcpp_caps_collapsible_frame.pack(fill="x", expand=False, padx=10, pady=(10, 5))
        self.kcpp_caps_text_widget = ctk.CTkTextbox(kcpp_caps_collapsible_frame.content_frame, height=100, wrap="word", font=("Segoe UI", 11))
        self.kcpp_caps_text_widget.pack(fill="x", expand=True, padx=5, pady=(0, 5))
        self.kcpp_caps_text_widget.configure(state="disabled")
        self.btn_redetect_caps = ctk.CTkButton(kcpp_caps_collapsible_frame.content_frame, text="Re-detect Capabilities", command=lambda: self.update_kcpp_capabilities_display(re_detect=True))
        self.btn_redetect_caps.pack(pady=(0, 5), anchor="e", padx=5)
        ToolTip(self.btn_redetect_caps, "Queries the configured KoboldCpp executable with '--help'\nto determine its supported features and arguments.")

        launcher_behavior_frame = ctk.CTkFrame(sf)
        launcher_behavior_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(launcher_behavior_frame, text="Launcher Behavior:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=(5, 10), pady=5, sticky="w")
        self.auto_open_webui_var = ctk.BooleanVar(value=self.config.get("auto_open_webui", True))
        auto_open_checkbox = ctk.CTkCheckBox(launcher_behavior_frame, text="Auto-Open Web UI After Successful Launch", variable=self.auto_open_webui_var)
        auto_open_checkbox.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        self.auto_open_webui_var.trace_add("write", self.mark_settings_dirty)
        ToolTip(auto_open_checkbox, "If checked, automatically opens the KoboldCpp Web UI in your browser after a successful launch.")

        gpu_management_outer_frame = ctk.CTkFrame(sf)
        gpu_management_outer_frame.pack(fill="x", padx=10, pady=(20, 5))
        ctk.CTkLabel(gpu_management_outer_frame, text="GPU Management & VRAM Override", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", pady=(0, 10), padx=5)
        gpu_management_frame = ctk.CTkFrame(gpu_management_outer_frame)
        gpu_management_frame.pack(fill="x", padx=0, pady=0)

        gpu_select_subframe = ctk.CTkFrame(gpu_management_frame)
        gpu_select_subframe.pack(fill="x", pady=(0, 5))
        # GPU Type Selection (assuming you had this or plan to add it)
        ctk.CTkLabel(gpu_select_subframe, text="GPU Detection Mode:").grid(row=0, column=0, padx=(5,2), pady=(5,2), sticky="w")
        self.gpu_selection_mode_optionmenu = ctk.CTkOptionMenu(gpu_select_subframe, variable=self.gpu_selection_mode_var, 
                                                       values=["auto", "nvidia", "amd", "intel", "apple"], 
                                                       command=lambda selected_type: self._gpu_type_selection_changed(selected_type))
        self.gpu_selection_mode_optionmenu.grid(row=0, column=1, padx=(2,10), pady=(5,2), sticky="w")
        ToolTip(self.gpu_selection_mode_optionmenu, "Select which GPU type to target or use 'auto' for detection.")

        ctk.CTkLabel(gpu_select_subframe, text="Target GPU ID:").grid(row=0, column=2, padx=(10,2), pady=(5,2), sticky="w")
        self.gpu_id_optionmenu = ctk.CTkOptionMenu(gpu_select_subframe, variable=self.selected_gpu_id_display_var, values=["N/A (Auto-Detect)"], command=lambda selected_value: self._gpu_id_selection_changed(selected_value), width=200)
        self.gpu_id_optionmenu.grid(row=0, column=3, padx=2, pady=(5,2), sticky="ew")
        ToolTip(self.gpu_id_optionmenu, "Select specific GPU ID after choosing type and refreshing list.")

        self.btn_refresh_gpu_list = ctk.CTkButton(gpu_select_subframe, text="Refresh GPU List", command=lambda: self._populate_gpu_id_dropdown(), width=120)
        self.btn_refresh_gpu_list.grid(row=0, column=4, padx=(10,5), pady=(5,2), sticky="e")
        ToolTip(self.btn_refresh_gpu_list, "Scan for GPUs and update the ID list.")

        self.gpu_status_label = ctk.CTkLabel(gpu_select_subframe, text="", font=ctk.CTkFont(size=10))
        self.gpu_status_label.grid(row=1, column=0, columnspan=5, padx=5, pady=(0,5), sticky="w")

        vram_override_subframe = ctk.CTkFrame(gpu_management_frame)
        vram_override_subframe.pack(fill="x", pady=(5, 5))
        self.override_vram_checkbox = ctk.CTkCheckBox(vram_override_subframe, text="Override Detected Total VRAM for Launcher Calculations", variable=self.override_vram_var, command=lambda: self._toggle_manual_vram_entry_state())
        self.override_vram_checkbox.pack(side="left", padx=5, pady=5)
        self.override_vram_var.trace_add("write", self.mark_settings_dirty)
        ToolTip(self.override_vram_checkbox, "Manually set the total VRAM budget the launcher should assume.")
        ctk.CTkLabel(vram_override_subframe, text="Manual Total (MB):").pack(side="left", padx=(10, 2), pady=5)
        self.manual_vram_entry = ctk.CTkEntry(vram_override_subframe, width=100)
        self.manual_vram_entry.pack(side="left", padx=2, pady=5)
        self.manual_vram_entry.bind("<KeyRelease>", self.mark_settings_dirty)
        ToolTip(self.manual_vram_entry, "Enter total VRAM in MB for override.")

        global_args_collapsible_frame = self._create_collapsible_frame(sf, "Global KoboldCpp Default Arguments", initially_collapsed=True)
        global_args_collapsible_frame.pack(fill="x", expand=False, padx=10, pady=(20, 10))
        self.settings_widgets = {}
        grouped_args = {}
        category_order = ["Core", "Network", "Model Params", "Performance", "GPU Backend", "GPU Offload", "GPU Offload (Advanced)", "GPU Optimizations", "Memory", "Other"]
        for setting_def in tensortune_core.KOBOLDCPP_ARG_DEFINITIONS:
            category = setting_def.get("category", "Other")
            if category not in grouped_args:
                grouped_args[category] = []
            grouped_args[category].append(setting_def)

        for category_name in category_order:
            if category_name in grouped_args:
                category_frame = ctk.CTkFrame(global_args_collapsible_frame.content_frame)
                category_frame.pack(fill="x", padx=5, pady=5)
                ctk.CTkLabel(category_frame, text=category_name, font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=5, pady=(5, 2))
                for setting_def in grouped_args[category_name]:
                    param_key = setting_def["key"]
                    if param_key == "--model":
                        continue
                    core_default_value = tensortune_core.DEFAULT_CONFIG_TEMPLATE["default_args"].get(param_key)
                    type_hint = setting_def.get("type_hint", "str")
                    arg_frame = ctk.CTkFrame(category_frame)
                    arg_frame.pack(fill="x", padx=5, pady=2)
                    arg_frame.grid_columnconfigure(1, weight=0)
                    arg_frame.grid_columnconfigure(2, weight=1)

                    label_widget = ctk.CTkLabel(arg_frame, text=f"{setting_def['name']}:", width=180, anchor="w")
                    label_widget.grid(row=0, column=0, padx=(5, 0), pady=2, sticky="w")
                    ToolTip(label_widget, setting_def.get("help", "No description available."), wraplength=400)

                    widget_instance = None
                    if type_hint in ["bool", "bool_flag"]:
                        initial_bool_value = False
                        if isinstance(core_default_value, bool):
                            initial_bool_value = core_default_value
                        elif isinstance(core_default_value, str):
                            initial_bool_value = core_default_value.lower() == 'true'
                        var = ctk.BooleanVar(value=initial_bool_value)
                        widget_instance = ctk.CTkCheckBox(arg_frame, text="", variable=var, width=20)
                        widget_instance.grid(row=0, column=1, padx=(0, 5), pady=2, sticky="w")
                        widget_instance.variable = var
                        var.trace_add("write", self.mark_settings_dirty)
                    else:
                        widget_instance = ctk.CTkEntry(arg_frame, width=120)
                        if core_default_value is not None:
                            widget_instance.insert(0, str(core_default_value))
                        widget_instance.grid(row=0, column=1, padx=(0, 5), pady=2, sticky="w")
                        widget_instance.bind("<KeyRelease>", self.mark_settings_dirty)
                    self.settings_widgets[param_key] = {"widget": widget_instance, "type_hint": type_hint}

        model_specific_frame_outer = ctk.CTkFrame(sf)
        model_specific_frame_outer.pack(fill="x", padx=10, pady=(25, 5))
        ctk.CTkLabel(model_specific_frame_outer, text="Manage Model-Specific Configurations", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=5, pady=(0, 5))
        btn_add_model_specific = ctk.CTkButton(model_specific_frame_outer, text="Browse & Add GGUF for Specific Config", command=lambda: self.browse_and_add_model_specific_config())
        btn_add_model_specific.pack(anchor="w", padx=5, pady=(0, 10))
        ToolTip(btn_add_model_specific, "Select a GGUF file to create or edit its specific launch argument overrides.")

        self.model_specifics_list_frame = ctk.CTkScrollableFrame(model_specific_frame_outer, height=180)
        self.model_specifics_list_frame.pack(fill="x", expand=True, padx=0, pady=0)

        action_buttons_frame = ctk.CTkFrame(sf)
        action_buttons_frame.pack(fill="x", padx=10, pady=(20, 10), side="bottom")

        self.btn_save_settings_tab = ctk.CTkButton(action_buttons_frame, text="Save Settings", command=lambda: self.save_settings_action())
        self.btn_save_settings_tab.pack(side="left", padx=5, pady=5)
        ToolTip(self.btn_save_settings_tab, "Save all changes made in this Settings tab to the launcher's configuration file.")

        btn_reset_settings = ctk.CTkButton(action_buttons_frame, text="Reset All to Defaults", command=lambda: self.reset_config_action(), fg_color="#dc3545", hover_color="#c82333")
        btn_reset_settings.pack(side="left", padx=5, pady=5)
        ToolTip(btn_reset_settings, "WARNING: Resets all launcher settings to their original defaults.\nA backup of your current config will be attempted.")

        config_io_frame = ctk.CTkFrame(action_buttons_frame)
        config_io_frame.pack(side="left", padx=(10,0))

        self.btn_export_config = ctk.CTkButton(config_io_frame, text="Export Config", command=lambda: self.export_gui_config())
        self.btn_export_config.pack(side="left", padx=(0,5), pady=5)
        ToolTip(self.btn_export_config, "Save all current launcher settings (excluding history DB content)\nto a JSON file for backup or sharing.")

        self.btn_import_config = ctk.CTkButton(config_io_frame, text="Import Config", command=lambda: self.import_gui_config())
        self.btn_import_config.pack(side="left", padx=0, pady=5)
        ToolTip(self.btn_import_config, "Load all launcher settings from a previously exported JSON file.\nWARNING: This will overwrite current settings.")

    def _toggle_manual_vram_entry_state(self, *args):
        if hasattr(self, 'manual_vram_entry') and self.manual_vram_entry.winfo_exists():
            self.manual_vram_entry.configure(state="normal" if self.override_vram_var.get() else "disabled")
        self.mark_settings_dirty()
        self.refresh_vram() # Refresh VRAM display as override status changed

    def _gpu_type_selection_changed(self, selected_type: str):
        # This method is likely tied to a CTkOptionMenu's command for GPU type (e.g., auto, nvidia, amd)
        # It should update the config and refresh the GPU ID dropdown based on the new type
        self.config["gpu_selection_mode"] = selected_type # Update config immediately or mark dirty
        self._populate_gpu_id_dropdown()
        self.mark_settings_dirty() # Mark dirty as a selection was made
        self.refresh_vram() # Refresh VRAM display for new GPU context

    def _gpu_id_selection_changed(self, selected_id_display_str: str):
        actual_id_to_store = 0  # Default to 0 if parsing fails or N/A
        if selected_id_display_str and "N/A" not in selected_id_display_str and "No " not in selected_id_display_str:
            try:
                # Extract ID number, e.g., from "ID 0: GeForce RTX 3090"
                match = re.search(r"ID\s*(\d+)", selected_id_display_str)
                if match:
                    actual_id_to_store = int(match.group(1))
            except ValueError:
                self.log_to_console(f"Warning: Could not parse GPU ID from '{selected_id_display_str}'. Defaulting to 0.")
        
        if self.config.get("selected_gpu_index") != actual_id_to_store:
            self.config["selected_gpu_index"] = actual_id_to_store
            self.mark_settings_dirty()
        self.refresh_vram() # Refresh VRAM based on new selection


    def _populate_gpu_id_dropdown(self):
        if not (hasattr(self, 'gpu_id_optionmenu') and self.gpu_id_optionmenu.winfo_exists() and
                hasattr(self, 'gpu_status_label') and self.gpu_status_label.winfo_exists()):
            self.log_to_console("GPU ID dropdown or status label not ready for population.")
            return

        user_selected_type = self.gpu_selection_mode_var.get()
        gpu_list_from_core = []
        effective_type_for_listing = user_selected_type
        display_prefix = "" # For "Auto: NVIDIA" type prefixes

        # Set UI to "Scanning..." state
        self.selected_gpu_id_display_var.set("Scanning...")
        self.gpu_id_optionmenu.configure(values=["Scanning..."])
        self.gpu_status_label.configure(text="Status: Scanning...")
        self.update_idletasks() # Ensure UI updates before potentially long core calls

        if user_selected_type == "auto":
            self.log_to_console("GPU type is 'auto', determining actual detected GPU type for ID listing...")
            _, _, _, gpu_info_dict = tensortune_core.get_available_vram_mb(
                self.config, 
                target_gpu_type=None, # For auto, core function should detect
                target_gpu_index=self.config.get("selected_gpu_index", 0) # Pass current index preference
            )
            if gpu_info_dict and gpu_info_dict.get("success") and \
               gpu_info_dict.get("type") not in ["Unknown/None_Auto", "N/A", "Unknown/None", "INVALID_TARGET_PREFS"]:
                detected_gpu_vendor = gpu_info_dict.get("type", "").lower()
                # Normalize vendor string
                if "nvidia" in detected_gpu_vendor: detected_gpu_vendor = "nvidia"
                elif "amd" in detected_gpu_vendor: detected_gpu_vendor = "amd"
                elif "intel" in detected_gpu_vendor: detected_gpu_vendor = "intel"
                elif "apple_metal" in detected_gpu_vendor or "apple" in detected_gpu_vendor: detected_gpu_vendor = "apple"

                if detected_gpu_vendor in ["nvidia", "amd", "intel", "apple"]:
                    effective_type_for_listing = detected_gpu_vendor
                    display_prefix = f"(Auto: {detected_gpu_vendor.upper()}) "
                    self.log_to_console(f"Auto-detected GPU type as '{detected_gpu_vendor}' for ID listing.")
                else:
                    self.log_to_console(f"Auto-detection yielded an unusable type: '{detected_gpu_vendor}'. No IDs to list for 'auto'.")
                    self.gpu_id_optionmenu.configure(values=["N/A (Auto: No specific GPU type found)"])
                    self.selected_gpu_id_display_var.set("N/A (Auto: No specific GPU type found)")
                    self.gpu_status_label.configure(text="Status: Auto - No specific type.")
                    return # Stop if auto-detection doesn't yield a listable type
            else:
                msg = gpu_info_dict.get('message', 'N/A') if gpu_info_dict else "N/A (core func error)"
                self.log_to_console(f"Auto GPU detection failed or no specific GPU found. Message: {msg}")
                self.gpu_id_optionmenu.configure(values=["N/A (Auto: Detection Failed/No GPU)"])
                self.selected_gpu_id_display_var.set("N/A (Auto: Detection Failed/No GPU)")
                self.gpu_status_label.configure(text="Status: Auto - Detection failed.")
                return

        # Fetch GPU list based on effective type
        if effective_type_for_listing == "nvidia":
            gpu_list_from_core = tensortune_core.list_nvidia_gpus()
        elif effective_type_for_listing == "amd":
            if platform.system() == "Windows":
                gpu_list_from_core = tensortune_core.list_amd_gpus_windows()
            elif platform.system() == "Linux":
                gpu_list_from_core = tensortune_core.list_amd_gpus_linux()
            # Add macOS for AMD if supported by core
        elif effective_type_for_listing == "intel":
            gpu_list_from_core = tensortune_core.list_intel_gpus()
        elif effective_type_for_listing == "apple" and platform.system() == "darwin": # Explicitly check for macOS
             gpu_list_from_core = tensortune_core.list_apple_gpus()


        display_values = []
        if gpu_list_from_core:
            display_values = [f"{display_prefix}ID {gpu['id']}: {gpu['name']}" for gpu in gpu_list_from_core]
            self.gpu_status_label.configure(text=f"Status: {len(gpu_list_from_core)} {effective_type_for_listing.upper()} GPU(s) found.")
        else:
            status_msg_suffix = "No GPUs found for detected type" if user_selected_type == "auto" else f"No {effective_type_for_listing.upper()} GPUs Found"
            display_values = [f"N/A ({status_msg_suffix})"]
            self.gpu_status_label.configure(text=f"Status: {status_msg_suffix}.")

        self.gpu_id_optionmenu.configure(values=display_values)

        # Try to set the dropdown to the currently configured GPU index
        current_config_idx = self.config.get("selected_gpu_index", 0)
        newly_selected_id_str_for_var = None

        if display_values and "N/A" not in display_values[0] and "No " not in display_values[0] and "Error" not in display_values[0]:
            for option_str in display_values:
                match = re.search(r"ID\s*(\d+)", option_str)
                if match:
                    try:
                        if int(match.group(1)) == current_config_idx:
                            newly_selected_id_str_for_var = option_str
                            break
                    except ValueError:
                        continue # Should not happen if regex is correct
            if not newly_selected_id_str_for_var: # Configured ID not found (e.g. different GPU type selected)
                newly_selected_id_str_for_var = display_values[0] # Default to first in list
                # Update config to reflect this change if it's a valid ID
                match_first = re.search(r"ID\s*(\d+)", display_values[0])
                new_default_parsed_id = 0
                if match_first:
                    try: new_default_parsed_id = int(match_first.group(1))
                    except ValueError: pass # keep 0
                if self.config.get("selected_gpu_index") != new_default_parsed_id: # Only if changed
                    self.config["selected_gpu_index"] = new_default_parsed_id
                    # Do not mark_settings_dirty here as this is an auto-adjustment
                    self.log_to_console(f"Configured GPU index {current_config_idx} not in list for type '{effective_type_for_listing}'. Defaulted to index {new_default_parsed_id}.")
        else: # No GPUs found or error
            newly_selected_id_str_for_var = display_values[0] if display_values else "Error (No GPUs)"
            if self.config.get("selected_gpu_index") != 0: # Reset to 0 if no valid GPUs
                self.config["selected_gpu_index"] = 0

        self.selected_gpu_id_display_var.set(newly_selected_id_str_for_var)
        self.log_to_console(f"GPU ID list for user type '{user_selected_type}' (effective: '{effective_type_for_listing}') refreshed. Displaying: {newly_selected_id_str_for_var}")
        # No self.mark_settings_dirty() or self.refresh_vram() here, as this is called by them or on startup.


    def load_settings_from_config(self):
        if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
            self.exe_path_entry.delete(0, "end")
            self.exe_path_entry.insert(0, self.config.get("koboldcpp_executable", ""))
        
        if hasattr(self, 'theme_var') and self.theme_var: # Check if theme_var exists
            self.theme_var.set(self.config.get("color_mode", "dark").lower())
        
        if hasattr(self, 'auto_open_webui_var') and self.auto_open_webui_var: # Check if var exists
            self.auto_open_webui_var.set(self.config.get("auto_open_webui", True))

        self.gpu_selection_mode_var.set(self.config.get("gpu_selection_mode", "auto"))
        self.override_vram_var.set(self.config.get("override_vram_budget", False))
        
        if hasattr(self, 'manual_vram_entry') and self.manual_vram_entry.winfo_exists():
            self.manual_vram_entry.delete(0, "end")
            self.manual_vram_entry.insert(0, str(self.config.get("manual_vram_total_mb", 8192))) # Default to a string
        
        self._toggle_manual_vram_entry_state() # Ensure enabled/disabled state is correct

        # Load global default arguments
        global_default_args_from_config = self.config.get("default_args", {})
        for arg_definition in tensortune_core.KOBOLDCPP_ARG_DEFINITIONS:
            param_key = arg_definition["key"]
            if param_key not in self.settings_widgets:
                continue # Skip if no widget for this setting (e.g. --model)
            
            widget_info = self.settings_widgets[param_key]
            widget = widget_info["widget"]
            type_hint = arg_definition.get("type_hint", "str") # Use from definition

            value_from_config = global_default_args_from_config.get(param_key)
            core_template_default_value = tensortune_core.DEFAULT_CONFIG_TEMPLATE["default_args"].get(param_key)
            
            final_value_to_set = value_from_config if value_from_config is not None else core_template_default_value

            if type_hint in ["bool", "bool_flag"]:
                bool_value = False # Default for safety
                if isinstance(final_value_to_set, bool):
                    bool_value = final_value_to_set
                elif isinstance(final_value_to_set, str): # Handle "true"/"false" strings from config
                    bool_value = final_value_to_set.lower() == 'true'
                
                if hasattr(widget, 'variable'): # CTkCheckBox with BooleanVar
                    widget.variable.set(bool_value)
                elif isinstance(widget, ctk.CTkCheckBox): # Direct select/deselect if no separate var
                     if bool_value: widget.select()
                     else: widget.deselect()

            elif isinstance(widget, ctk.CTkEntry):
                widget.delete(0, "end")
                if final_value_to_set is not None:
                    widget.insert(0, str(final_value_to_set))
        
        if hasattr(self, 'populate_model_specifics_list_display'):
            self.populate_model_specifics_list_display()
        if hasattr(self, 'update_kcpp_capabilities_display'):
            self.update_kcpp_capabilities_display(re_detect=False) # Don't force re-detect on load

        self.settings_dirty = False # Reset dirty flag after loading
        self.update_save_button_state()
        self.log_to_console("Settings tab populated from current configuration.")

    def save_config(self):
        # Paths
        if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
            self.config["koboldcpp_executable"] = self.exe_path_entry.get().strip()
        self.config["default_gguf_dir"] = self.default_model_dir # Assumed to be up-to-date
        self.config["db_file"] = self.db_path # Assumed to be up-to-date

        # UI Settings
        if hasattr(self, 'theme_var') and self.theme_var:
            self.config["color_mode"] = self.theme_var.get().lower()
        else: # Fallback if theme_var somehow not set up
            self.config["color_mode"] = ctk.get_appearance_mode().lower()

        if hasattr(self, 'auto_open_webui_var') and self.auto_open_webui_var:
            self.config["auto_open_webui"] = self.auto_open_webui_var.get()

        # GPU Settings
        self.config["gpu_selection_mode"] = self.gpu_selection_mode_var.get()
        # selected_gpu_index is updated directly by _gpu_id_selection_changed
        self.config["override_vram_budget"] = self.override_vram_var.get()
        if hasattr(self, 'manual_vram_entry') and self.manual_vram_entry.winfo_exists():
            try:
                manual_vram_val_str = self.manual_vram_entry.get().strip()
                self.config["manual_vram_total_mb"] = int(manual_vram_val_str) if manual_vram_val_str and manual_vram_val_str.isdigit() else tensortune_core.DEFAULT_CONFIG_TEMPLATE["manual_vram_total_mb"]
            except ValueError:
                self.config["manual_vram_total_mb"] = tensortune_core.DEFAULT_CONFIG_TEMPLATE["manual_vram_total_mb"]
                self.log_to_console(f"Warning: Invalid manual VRAM total '{manual_vram_val_str}', defaulted.")
        else: # Fallback
            self.config["manual_vram_total_mb"] = tensortune_core.DEFAULT_CONFIG_TEMPLATE["manual_vram_total_mb"]


        # Global KoboldCpp Arguments
        current_global_defaults_in_config = self.config.get("default_args", {}).copy()
        for arg_definition in tensortune_core.KOBOLDCPP_ARG_DEFINITIONS:
            param_key = arg_definition["key"]
            if param_key not in self.settings_widgets:
                continue

            widget_info = self.settings_widgets[param_key]
            widget = widget_info["widget"]
            type_hint = arg_definition.get("type_hint", "str") # Use from definition
            new_value = None

            if type_hint in ["bool", "bool_flag"]:
                if hasattr(widget, 'variable'): # CTkCheckBox with BooleanVar
                    new_value = widget.variable.get()
                elif isinstance(widget, ctk.CTkCheckBox): # Direct get if no var
                    new_value = (widget.get() == 1) 
            elif isinstance(widget, ctk.CTkEntry):
                new_value_str = widget.get().strip()
                core_template_val = tensortune_core.DEFAULT_CONFIG_TEMPLATE["default_args"].get(param_key)

                if not new_value_str: # Empty string, treat as "use default"
                    new_value = core_template_val # Explicitly set to template default or None
                elif isinstance(core_template_val, int) and new_value_str.lower() != "auto": # preserve "auto" as str
                    try: new_value = int(new_value_str)
                    except ValueError: new_value = new_value_str # Keep as string if not parsable as int
                elif isinstance(core_template_val, float) and new_value_str.lower() != "auto":
                    try: new_value = float(new_value_str)
                    except ValueError: new_value = new_value_str # Keep as string if not parsable
                else:
                    new_value = new_value_str # Store as string (includes "auto")
            
            template_default = tensortune_core.DEFAULT_CONFIG_TEMPLATE["default_args"].get(param_key)
            # Logic to decide if we store the value or remove it (if it matches template default)
            if new_value is not None:
                if type_hint in ["bool", "bool_flag"]: # Booleans are always stored if different or explicitly set
                    current_global_defaults_in_config[param_key] = bool(new_value)
                elif str(new_value) != str(template_default): # Store if different from template
                    current_global_defaults_in_config[param_key] = new_value
                elif param_key in current_global_defaults_in_config: # Same as template, remove if exists
                    del current_global_defaults_in_config[param_key]
            # If new_value is None (e.g. empty entry for non-bool) and it was in config, it means "reset to default"
            # which is handled by it not being in current_global_defaults_in_config unless different from template.

        self.config["default_args"] = current_global_defaults_in_config
        
        # Model-specific args are managed by their own dialogs, ensure key exists
        if "model_specific_args" not in self.config:
            self.config["model_specific_args"] = {}

        success, message = tensortune_core.save_launcher_config(self.config)
        if success:
            self.log_to_console(message)
            # Update internal state that depends on saved config
            self.koboldcpp_executable = self.config["koboldcpp_executable"]
            self.default_model_dir = self.config.get("default_gguf_dir", "")
            self.db_path = self.config.get("db_file", self.db_path) # Ensure self.db_path is current
            self.settings_dirty = False
            self.update_save_button_state()
        else:
            self.log_to_console(f"Error saving configuration: {message}")
            messagebox.showerror("Configuration Save Error", f"Could not save configuration:\n{message}", parent=self)
        return success

    def monitor_vram(self):
        while True:
            try:
                # Use current config values for GPU selection
                selected_mode_from_cfg = self.config.get("gpu_selection_mode", "auto")
                selected_idx_from_cfg = self.config.get("selected_gpu_index", 0)
                
                target_type_for_query = selected_mode_from_cfg if selected_mode_from_cfg != "auto" else None
                target_idx_for_query = selected_idx_from_cfg

                # Call core function with current config context
                _, _, _, gpu_info_dict_core = tensortune_core.get_available_vram_mb(
                    current_config=self.config, # Pass the whole config
                    target_gpu_type=target_type_for_query,
                    target_gpu_index=target_idx_for_query
                )
                
                self.gpu_info = gpu_info_dict_core # Update internal state

                total_mb_for_display = float(gpu_info_dict_core.get("total_mb_budgeted", 0.0))
                free_mb_for_display = float(gpu_info_dict_core.get("free_mb_budgeted", 0.0))
                used_mb_for_display = 0.0
                if total_mb_for_display > 0: # Avoid division by zero
                     used_mb_for_display = total_mb_for_display - free_mb_for_display
                
                # Clamp used_mb to be between 0 and total_mb
                used_mb_for_display = max(0.0, min(used_mb_for_display, total_mb_for_display if total_mb_for_display > 0 else 0.0))
                final_message_for_display = gpu_info_dict_core.get("message", "N/A")

                if hasattr(self, 'after') and self.winfo_exists(): # Check if GUI still exists
                    self.after(0, lambda u=used_mb_for_display, t=total_mb_for_display, msg=final_message_for_display: self.update_vram_display(u, t, msg))
                else: # GUI closed, exit thread
                    break 
            except Exception as e_vram_mon:
                print(f"Error in VRAM monitor thread: {e_vram_mon}") # Log to console
                traceback.print_exc()
                if hasattr(self, 'after') and self.winfo_exists():
                    self.after(0, lambda: self.update_vram_display(0, 0, "VRAM monitor error"))
                else:
                    break # GUI closed
            time.sleep(5) # Check every 5 seconds

    def update_vram_display(self, used_mb: float, total_mb: float, message_from_core: str = ""):
        final_message_text = message_from_core if message_from_core else "VRAM N/A"

        # Prepend "Manual Budget" if override is active, trying to integrate it nicely
        if self.gpu_info and self.gpu_info.get("override_active", False):
            override_prefix = "Manual Budget"
            vendor_part_cleaned = "" 
            detail_part_after_vendor_cleaned = final_message_text

            known_vendors = ["NVIDIA", "AMD", "Intel", "Metal", "APPLE_METAL"] # Metal for Apple
            for v_name_check in known_vendors:
                match = re.match(rf"^({re.escape(v_name_check)})(\s*:|\s+|$)", final_message_text, re.IGNORECASE)
                if match:
                    vendor_part_cleaned = match.group(1) 
                    detail_part_after_vendor_cleaned = final_message_text[match.end(0):].strip()
                    break 
            
            if vendor_part_cleaned:
                if override_prefix not in detail_part_after_vendor_cleaned:
                    final_message_text = f"{vendor_part_cleaned}: {override_prefix} ({detail_part_after_vendor_cleaned})"
                else: 
                    final_message_text = f"{vendor_part_cleaned}: {detail_part_after_vendor_cleaned}"
            elif override_prefix not in final_message_text:
                final_message_text = f"{override_prefix} ({final_message_text})"

        # Update Main Tab VRAM display
        if hasattr(self, 'vram_progress') and self.vram_progress.winfo_exists():
            if total_mb > 0:
                percentage = min(1.0, max(0.0, used_mb / total_mb))
                self.vram_progress.set(percentage)
                progress_color = "#28a745" # Green
                if percentage > 0.9: progress_color = "#dc3545" # Red
                elif percentage > 0.7: progress_color = "#ffc107" # Yellow
                self.vram_progress.configure(progress_color=progress_color)
            else:
                self.vram_progress.set(0)
                self.vram_progress.configure(progress_color=ctk.ThemeManager.theme["CTkProgressBar"]["progress_color"]) # Default color
            
            if hasattr(self, 'vram_text') and self.vram_text.winfo_exists():
                self.vram_text.configure(text=final_message_text)

        # Update Tuning Tab VRAM display (if exists)
        if hasattr(self, 'tuning_view_vram_progress') and self.tuning_view_vram_progress.winfo_exists():
            if total_mb > 0:
                percentage_tuning = min(1.0, max(0.0, used_mb / total_mb))
                self.tuning_view_vram_progress.set(percentage_tuning)
                progress_color_tuning = "#28a745"
                if percentage_tuning > 0.9: progress_color_tuning = "#dc3545"
                elif percentage_tuning > 0.7: progress_color_tuning = "#ffc107"
                self.tuning_view_vram_progress.configure(progress_color=progress_color_tuning)
            else:
                self.tuning_view_vram_progress.set(0)
                self.tuning_view_vram_progress.configure(progress_color=ctk.ThemeManager.theme["CTkProgressBar"]["progress_color"])
            
            if hasattr(self, 'tuning_view_vram_text') and self.tuning_view_vram_text.winfo_exists():
                self.tuning_view_vram_text.configure(text=final_message_text)

    def refresh_vram(self):
        self.log_to_console("User requested VRAM refresh...")
        try:
            selected_mode_from_cfg = self.config.get("gpu_selection_mode", "auto")
            selected_idx_from_cfg = self.config.get("selected_gpu_index", 0)
            
            target_type_for_query = selected_mode_from_cfg if selected_mode_from_cfg != "auto" else None
            target_idx_for_query = selected_idx_from_cfg

            _, _, _, gpu_info_dict_core = tensortune_core.get_available_vram_mb(
                current_config=self.config,
                target_gpu_type=target_type_for_query,
                target_gpu_index=target_idx_for_query
            )
            self.gpu_info = gpu_info_dict_core

            total_mb_for_display = float(gpu_info_dict_core.get("total_mb_budgeted", 0.0))
            free_mb_for_display = float(gpu_info_dict_core.get("free_mb_budgeted", 0.0))
            used_mb_for_display = 0.0
            if total_mb_for_display > 0:
                 used_mb_for_display = total_mb_for_display - free_mb_for_display
            
            used_mb_for_display = max(0.0, min(used_mb_for_display, total_mb_for_display if total_mb_for_display > 0 else 0.0))
            final_message_for_display = gpu_info_dict_core.get("message", "N/A")
            
            self.update_vram_display(used_mb_for_display, total_mb_for_display, final_message_for_display)
            self.log_to_console(f"VRAM Refreshed: {final_message_for_display}")
        except Exception as e_vram_refresh:
            self.log_to_console(f"Error during manual VRAM refresh: {e_vram_refresh}")
            traceback.print_exc()
            self.update_vram_display(0, 0, "Error during VRAM refresh")

    def browse_and_add_model_specific_config(self):
        initial_dir_for_dialog = self.config.get("last_used_gguf_dir", self.default_model_dir)
        if not initial_dir_for_dialog or not os.path.isdir(initial_dir_for_dialog):
            initial_dir_for_dialog = os.getcwd() # Fallback
        
        filepath_selected = filedialog.askopenfilename(
            title="Select GGUF Model for Specific Configuration",
            filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")],
            initialdir=initial_dir_for_dialog,
            parent=self
        )
        if filepath_selected:
            abs_filepath = os.path.abspath(filepath_selected)
            self.log_to_console(f"GGUF selected for specific config: {abs_filepath}")
            
            if "model_specific_args" not in self.config: # Ensure key exists
                 self.config["model_specific_args"] = {}
            
            if abs_filepath not in self.config.get("model_specific_args", {}):
                self.config.setdefault("model_specific_args", {})[abs_filepath] = {} # Add empty dict if new
                self.mark_settings_dirty() # Config structure changed
                # Consider saving here or let user save via main button
            
            self.populate_model_specifics_list_display() # Refresh list
            self.open_model_specific_edit_dialog(abs_filepath)


    def populate_model_specifics_list_display(self):
        if not hasattr(self, 'model_specifics_list_frame') or not self.model_specifics_list_frame.winfo_exists():
            return

        for widget in self.model_specifics_list_frame.winfo_children():
            widget.destroy()

        model_configs = self.config.get("model_specific_args", {})
        if not model_configs:
            ctk.CTkLabel(self.model_specifics_list_frame, text="No model-specific configurations saved.\nUse 'Browse & Add GGUF...' to create one.").pack(pady=10, padx=10)
            return

        sorted_model_paths = sorted(model_configs.keys()) # Display alphabetically

        for model_path in sorted_model_paths:
            item_frame = ctk.CTkFrame(self.model_specifics_list_frame)
            item_frame.pack(fill="x", pady=(3, 0), padx=2) # Small vertical spacing

            model_display_name = os.path.basename(model_path)
            if len(model_display_name) > 55: # Truncate long names
                model_display_name = model_display_name[:26] + "..." + model_display_name[-26:]
            
            ctk.CTkLabel(item_frame, text=model_display_name, anchor="w", wraplength=400).pack(side="left", padx=(5, 10), pady=5, fill="x", expand=True)
            
            edit_button = ctk.CTkButton(item_frame, text="Edit", width=50, height=24, command=lambda mp=model_path: self.open_model_specific_edit_dialog(mp))
            edit_button.pack(side="left", padx=(0, 5), pady=5)
            ToolTip(edit_button, f"Edit specific settings for\n{os.path.basename(model_path)}")

            delete_button = ctk.CTkButton(item_frame, text="Delete", width=50, height=24, fg_color="firebrick", hover_color="darkred", command=lambda mp=model_path: self.delete_single_model_specific_config(mp))
            delete_button.pack(side="left", padx=(0, 5), pady=5)
            ToolTip(delete_button, f"Delete specific settings for\n{os.path.basename(model_path)}")

    def open_model_specific_edit_dialog(self, model_path_to_edit):
        """Create and manage a dialog for editing model-specific arguments."""
        if not model_path_to_edit:
            return

        try:
            dialog = ctk.CTkToplevel(self)
            dialog.title(f"Edit Specific Args: {os.path.basename(model_path_to_edit)}")
            dialog.geometry("800x700")
            dialog.transient(self)
            dialog.grab_set()
            dialog.attributes("-topmost", True)

            param_defs = self._get_param_definitions_for_dialog()
            
            args_for_display = tensortune_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
            args_for_display.update(self.config.get("default_args", {}))
            args_for_display.update(self.config.get("model_specific_args", {}).get(model_path_to_edit, {}))

            main_dialog_frame = ctk.CTkFrame(dialog)
            main_dialog_frame.pack(fill="both", expand=True)
            main_dialog_frame.grid_columnconfigure(0, weight=1)
            main_dialog_frame.grid_rowconfigure(0, weight=1) 

            content_frame, widgets_info_dialog = self._create_args_dialog_content_revised(main_dialog_frame, args_for_display, param_defs)
            content_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

            # Store any changes in a temporary dictionary
            temp_changes = {}

            # Define a function to collect widget values without risking focus issues
            def collect_widget_values():
                changes = {}
                try:
                    global_baseline_args = tensortune_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
                    global_baseline_args.update(self.config.get("default_args", {}))
                    
                    for param_key, info in widgets_info_dialog.items():
                        widget = info["widget"]
                        
                        # Skip if widget doesn't exist
                        if not (widget and hasattr(widget, 'winfo_exists') and widget.winfo_exists()):
                            continue
                            
                        type_hint = info["type_hint"]
                        new_value = None
                        is_empty = False

                        if type_hint in ["bool", "bool_flag"]:
                            if hasattr(widget, 'variable'):
                                new_value = widget.variable.get()
                        else:  # CTkEntry
                            try:
                                new_value_str = widget.get().strip()
                                if not new_value_str:
                                    is_empty = True
                                else:
                                    new_value = new_value_str
                            except:
                                continue
                        
                        if is_empty:
                            continue
                                
                        # Store only if value differs from base
                        base_value = global_baseline_args.get(param_key)
                        if type_hint in ["bool", "bool_flag"]:
                            if isinstance(base_value, str):
                                base_value = base_value.lower() == 'true'
                            elif not isinstance(base_value, bool):
                                base_value = False
                        else:
                            base_value = str(base_value) if base_value is not None else ""
                            new_value = str(new_value) if new_value is not None else ""
                            
                        if new_value != base_value:
                            changes[param_key] = new_value
                except Exception as e:
                    self.log_to_console(f"Error collecting values: {e}")
                    
                return changes

            # Define the save function
            def save_model_specifics_action():
                # Collect the values first
                collected_changes = collect_widget_values()
                
                # Store in temp dictionary
                temp_changes.update(collected_changes)
                
                # Close dialog - this avoids focus issues with destroyed widgets
                try:
                    dialog.grab_release()
                    dialog.destroy()
                except:
                    pass
                    
                # Use after to apply changes AFTER dialog is fully destroyed
                self.after(100, lambda: apply_changes_after_dialog_closed(collected_changes))
                    
            # Function to apply changes after dialog is closed
            def apply_changes_after_dialog_closed(changes):
                try:
                    if changes:
                        # Save the changes to the config
                        if changes:
                            self.config.setdefault("model_specific_args", {})[model_path_to_edit] = changes
                        elif model_path_to_edit in self.config.get("model_specific_args", {}):
                            del self.config["model_specific_args"][model_path_to_edit]
                            
                        if self.save_config():
                            self.log_to_console(f"Model-specific args updated for {os.path.basename(model_path_to_edit)}")
                            
                            # Refresh UI components
                            if hasattr(self, 'populate_model_specifics_list_display') and callable(self.populate_model_specifics_list_display):
                                self.populate_model_specifics_list_display()
                                
                            # Update tuning session if needed
                            if self.tuning_in_progress and self.current_tuning_model_path == model_path_to_edit:
                                if hasattr(self, '_reinitialize_session_base_args') and callable(self._reinitialize_session_base_args):
                                    self._reinitialize_session_base_args()
                                    
                                if hasattr(self, 'update_tuning_display') and callable(self.update_tuning_display):
                                    self.update_tuning_display()
                        else:
                            self.log_to_console("Failed to save config after editing model-specific args.")
                    else:
                        self.log_to_console(f"No changes detected for model-specific args of {os.path.basename(model_path_to_edit)}.")
                except Exception as e:
                    self.log_to_console(f"Error applying model-specific changes: {e}")

            # Define the cancel function
            def cancel_model_specifics_action():
                try:
                    dialog.grab_release()
                    dialog.destroy()
                except:
                    pass

            # Setup UI components
            button_frame_dialog = ctk.CTkFrame(main_dialog_frame)
            button_frame_dialog.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
            
            ctk.CTkButton(button_frame_dialog, text="Save Specifics for This Model", command=save_model_specifics_action).pack(side="left", padx=10)
            ctk.CTkButton(button_frame_dialog, text="Cancel", command=cancel_model_specifics_action).pack(side="right", padx=10)
            
            dialog.protocol("WM_DELETE_WINDOW", cancel_model_specifics_action)
            
        except Exception as e:
            self.log_to_console(f"Error setting up model-specific edit dialog: {e}")
            import traceback
            self.log_to_console(traceback.format_exc())
        
        def cancel_model_specifics_action():
            try:
                if dialog.winfo_exists():
                    dialog.focus_set()
                    dialog.grab_release()
                    dialog.destroy()
            except ctk.tumeurs.TclError:
                pass
            self.after(50, self.focus_set)

        ctk.CTkButton(button_frame_dialog, text="Cancel", command=cancel_model_specifics_action).pack(side="right", padx=10)
        
        dialog.wait_window()
        # self.after(50, self.focus_set) # Safeguard


    def _create_args_dialog_content_revised(self, parent_frame_for_scrollable, current_args_to_display, param_definitions_list):
        scrollable_content_frame = ctk.CTkScrollableFrame(parent_frame_for_scrollable, label_text="Model Arguments") # Label for context
        widgets_information = {}

        for setting_def in param_definitions_list:
            param_key = setting_def["key"]
            if param_key == "--model": # Skip --model as it's context, not an editable arg here
                continue

            current_value = current_args_to_display.get(param_key)
            type_hint = setting_def.get("type_hint", "str")
            
            row_frame = ctk.CTkFrame(scrollable_content_frame)
            row_frame.pack(fill="x", pady=2, padx=2)
            # No grid config needed for row_frame itself, pack handles children

            label_widget_dialog = ctk.CTkLabel(row_frame, text=f"{setting_def['name']}:", width=180, anchor="w")
            label_widget_dialog.pack(side="left", padx=(5,0)) # Add some padding
            ToolTip(label_widget_dialog, setting_def.get("help", "No description"), wraplength=400)

            widget_for_param = None
            if type_hint in ["bool", "bool_flag"]:
                bool_value_for_widget = False # Default
                if isinstance(current_value, bool):
                    bool_value_for_widget = current_value
                elif isinstance(current_value, str): # Handle "true"/"false" from config
                    bool_value_for_widget = current_value.lower() == 'true'
                
                var = ctk.BooleanVar(value=bool_value_for_widget)
                widget_for_param = ctk.CTkCheckBox(row_frame, text="", variable=var, width=20)
                widget_for_param.pack(side="left", padx=(0,5))
                widget_for_param.variable = var # Store var for easy access
            else:
                widget_for_param = ctk.CTkEntry(row_frame, width=150)
                if current_value is not None:
                    widget_for_param.insert(0, str(current_value))
                widget_for_param.pack(side="left", padx=(0,5))
            
            widgets_information[param_key] = {"widget": widget_for_param, "type_hint": type_hint}
        
        return scrollable_content_frame, widgets_information

    def delete_single_model_specific_config(self, model_path_to_delete):
        confirm = messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete the specific configuration for:\n{os.path.basename(model_path_to_delete)}?",
            parent=self,
            icon=messagebox.WARNING
        )
        if confirm:
            if "model_specific_args" in self.config and model_path_to_delete in self.config["model_specific_args"]:
                del self.config["model_specific_args"][model_path_to_delete]
                if self.save_config(): # Save the entire configuration
                    self.log_to_console(f"Deleted specific config for {os.path.basename(model_path_to_delete)}")
                    self.populate_model_specifics_list_display() # Refresh UI
                    # If tuning this model, re-init session args and update tuning display
                    if self.tuning_in_progress and self.current_tuning_model_path == model_path_to_delete:
                        self._reinitialize_session_base_args()
                        self.update_tuning_display()
                else:
                    self.log_to_console("Failed to save config after deleting model specific.")
            else:
                self.log_to_console(f"No specific config found to delete for {os.path.basename(model_path_to_delete)}")

    def export_gui_config(self):
        if self.settings_dirty:
            proceed = messagebox.askyesno(
                "Unsaved Settings",
                "You have unsaved changes in the Settings tab. Export the currently saved configuration (ignoring unsaved changes)?",
                detail="Choose 'Yes' to export last saved config, 'No' to cancel export.",
                parent=self,
                icon=messagebox.QUESTION
            )
            if not proceed:
                self.log_to_console("Export cancelled due to unsaved settings.")
                return

        export_filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Config files", "*.json"), ("All files", "*.*")],
            title="Export Launcher Configuration to...",
            initialdir=Path.home(), # Sensible default
            parent=self
        )
        if export_filepath:
            # Pass a copy of the config to avoid modifying the live one if core func does
            success, message = tensortune_core.export_config_to_file(self.config.copy(), export_filepath)
            if success:
                self.log_to_console(message)
                messagebox.showinfo("Export Successful", message, parent=self)
            else:
                self.log_to_console(f"Export failed: {message}")
                messagebox.showerror("Export Error", f"Could not export configuration:\n{message}", parent=self)

    def import_gui_config(self):
        import_filepath = filedialog.askopenfilename(
            filetypes=[("JSON Config files", "*.json"), ("All files", "*.*")],
            title="Import Launcher Configuration from...",
            initialdir=Path.home(),
            parent=self
        )
        if import_filepath:
            imported_config_data, message = tensortune_core.import_config_from_file(import_filepath)
            if imported_config_data:
                confirm_import = messagebox.askyesno(
                    "Confirm Import",
                    "This will OVERWRITE your current launcher settings with the content of the selected file.\n"
                    "A backup of your current settings will be attempted.\n\n"
                    "Do you want to proceed with the import?",
                    icon=messagebox.WARNING,
                    parent=self
                )
                if confirm_import:
                    try:
                        current_config_file_path = tensortune_core.CONFIG_FILE
                        if os.path.exists(current_config_file_path):
                            backup_path = current_config_file_path + f".backup_before_import_{time.strftime('%Y%m%d-%H%M%S')}.json"
                            shutil.copy2(current_config_file_path, backup_path)
                            self.log_to_console(f"Current configuration backed up to: {backup_path}")
                        
                        # Save the imported data directly to the config file
                        save_success, save_message = tensortune_core.save_launcher_config(imported_config_data)
                        if not save_success:
                            self.log_to_console(f"Failed to save imported config: {save_message}")
                            messagebox.showerror("Import Error", f"Failed to save imported configuration: {save_message}", parent=self)
                            return

                        # Re-initialize everything with the new config
                        self.log_to_console("Configuration data imported. Re-initializing...")
                        core_init_results = tensortune_core.initialize_launcher() # This will load from file
                        self.config = core_init_results["config"]
                        self.system_info = core_init_results["system_info"]
                        self.gpu_info = core_init_results["gpu_info"]
                        self.koboldcpp_capabilities = core_init_results.get("koboldcpp_capabilities", {})
                        self.db_path = self.config["db_file"] # Update internal db_path
                        self.default_model_dir = self.config.get("default_gguf_dir", os.getcwd())
                        self.koboldcpp_executable = self.config.get("koboldcpp_executable")

                        self.load_settings_from_config() # Populate UI from new config
                        self.check_koboldcpp_executable() # Validate new exe path
                        if hasattr(self, 'populate_model_specifics_list_display'):
                             self.populate_model_specifics_list_display()
                        if hasattr(self, 'update_kcpp_capabilities_display'):
                             self.update_kcpp_capabilities_display(re_detect=True) # Re-detect with new exe if changed
                        
                        self.after(100, self._populate_gpu_id_dropdown_on_startup) # Refresh GPU list
                        self.refresh_vram() # Refresh VRAM display
                        self.load_history() # Reload history, as DB path might have changed

                        if self.tuning_in_progress:
                            self.end_tuning_session(switch_to_model_selection=True) # End active tuning
                        else:
                            self._show_model_selection_view() # Ensure correct view

                        self.settings_dirty = False # Imported and saved, so not dirty
                        self.update_save_button_state()
                        self.log_to_console("Launcher re-initialized with imported settings.")
                        messagebox.showinfo("Import Successful", f"Configuration successfully imported from:\n{os.path.basename(import_filepath)}\n\nLauncher has been re-initialized.", parent=self)
                    
                    except Exception as e_apply:
                        error_msg_apply = f"Error applying imported configuration: {e_apply}\n{traceback.format_exc()}"
                        self.log_to_console(error_msg_apply)
                        messagebox.showerror("Import Application Error", error_msg_apply, parent=self)
            else:
                self.log_to_console(f"Import failed: {message}")
                messagebox.showerror("Import Error", f"Could not import configuration:\n{message}", parent=self)

    def update_kcpp_capabilities_display(self, re_detect=True):
        if not hasattr(self, 'kcpp_caps_text_widget') or not self.kcpp_caps_text_widget.winfo_exists():
            return

        if re_detect:
            self.log_to_console("Attempting to re-detect KoboldCpp capabilities...")
            if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
                self.koboldcpp_executable = self.exe_path_entry.get().strip()
                self.config["koboldcpp_executable"] = self.koboldcpp_executable # Update config
            
            self.check_koboldcpp_executable() # Validate and resolve the path
            self.koboldcpp_capabilities = tensortune_core.detect_koboldcpp_capabilities(self.koboldcpp_executable, force_redetect=True)
            
            if "error" not in self.koboldcpp_capabilities:
                self.log_to_console("KoboldCpp capabilities re-detected successfully.")
                self._update_gpu_backend_flags_in_config() # Potentially update backend flags
            else:
                self.log_to_console(f"Error re-detecting KCPP capabilities: {self.koboldcpp_capabilities['error']}")
        else: # Use cached if not re-detecting
            self.koboldcpp_capabilities = tensortune_core.detect_koboldcpp_capabilities(self.koboldcpp_executable, force_redetect=False)

        caps_str_parts = []
        if "error" in self.koboldcpp_capabilities:
            caps_str_parts.append(f"Error detecting capabilities:\n {self.koboldcpp_capabilities['error']}")
        else:
            display_order = ["cuda", "rocm", "opencl", "vulkan", "flash_attn", "auto_quantkv", "overridetensors"]
            friendly_names = {
                "cuda": "CUDA (cuBLAS)", "rocm": "ROCm (hipBLAS/rocBLAS)", 
                "opencl": "OpenCL (CLBlast)", "vulkan": "Vulkan Backend",
                "flash_attn": "FlashAttention Support", "auto_quantkv": "Auto QuantKV Option",
                "overridetensors": "Tensor Override Support"
            }
            for key in display_order:
                if key in self.koboldcpp_capabilities:
                    status = "Yes" if self.koboldcpp_capabilities.get(key) else "No"
                    caps_str_parts.append(f"{friendly_names.get(key, key.capitalize())}: {status}")
            
            # Add any other detected capabilities not in the primary list
            for key, value in self.koboldcpp_capabilities.items():
                if key not in display_order and key not in ["error", "available_args"]: # Exclude internal keys
                    status = "Yes" if value else "No"
                    caps_str_parts.append(f"{key.replace('_',' ').capitalize()}: {status}")

        caps_display_str = "\n".join(caps_str_parts) if caps_str_parts else "No capabilities data available or N/A."
        
        self.kcpp_caps_text_widget.configure(state="normal")
        self.kcpp_caps_text_widget.delete("1.0", "end")
        self.kcpp_caps_text_widget.insert("1.0", caps_display_str)
        self.kcpp_caps_text_widget.configure(state="disabled")

    def _update_gpu_backend_flags_in_config(self):
        if "error" in self.koboldcpp_capabilities: # Don't update if caps detection failed
            return
        if "default_args" not in self.config:
            self.config["default_args"] = {}

        original_cublas = self.config["default_args"].get("--usecublas", False)
        original_hipblas = self.config["default_args"].get("--usehipblas", False)

        # Temporarily set to False, then update based on detection
        self.config["default_args"]["--usecublas"] = False
        self.config["default_args"]["--usehipblas"] = False
        
        current_gpu_details = self.gpu_info # Use the latest fetched GPU info
        new_cublas, new_hipblas = False, False

        if current_gpu_details and current_gpu_details.get("type") == "NVIDIA" and self.config.get("gpu_detection", {}).get("nvidia", True):
            if self.koboldcpp_capabilities.get("cuda"):
                new_cublas = True
        elif current_gpu_details and current_gpu_details.get("type") == "AMD" and self.config.get("gpu_detection", {}).get("amd", True):
            if self.koboldcpp_capabilities.get("rocm"):
                new_hipblas = True
        
        if new_cublas != original_cublas or new_hipblas != original_hipblas:
            self.config["default_args"]["--usecublas"] = new_cublas
            self.config["default_args"]["--usehipblas"] = new_hipblas
            self.log_to_console(f"Auto-updated GPU backend flags: CUBLAS={new_cublas}, HIPBLAS={new_hipblas}. Refreshing settings UI.")
            self.load_settings_from_config() # Reload settings to reflect changes in UI
            self.mark_settings_dirty() # Changes were made

    def start_tuning_session(self):
        if not self.current_model_path:
            messagebox.showwarning("No Model Selected", "Please select a GGUF model first.", parent=self)
            return
        if not self.model_analysis_info or 'filepath' not in self.model_analysis_info:
            self.log_to_console("Model analysis info is missing. Re-analyzing...")
            self.analyze_model_action(self.current_model_path) # Analyze if not already done
            if not self.model_analysis_info or 'filepath' not in self.model_analysis_info: # Check again
                messagebox.showerror("Model Error", "Failed to analyze model. Cannot start tuning.", parent=self)
                return

        self.log_to_console(f"Starting tuning session for: {os.path.basename(self.current_model_path)}")
        self.tuning_in_progress = True
        self.current_tuning_model_path = self.current_model_path
        self.current_tuning_model_analysis = self.model_analysis_info.copy()
        
        self._reinitialize_session_base_args() # Load defaults for this model
        self.last_successful_monitored_run_details_gui = None # Reset session state
        self.user_requested_stop_monitoring = False

        if hasattr(self, 'tuning_model_name_label') and self.tuning_model_name_label.winfo_exists():
            self.tuning_model_name_label.configure(text=f"Model: {os.path.basename(self.current_tuning_model_path)}")
        
        # Get current VRAM for heuristics
        current_vram_budgeted, _, _, current_gpu_full_info = tensortune_core.get_available_vram_mb(
            self.config,
            target_gpu_type=self.config.get("gpu_selection_mode", "auto") if self.config.get("gpu_selection_mode", "auto") != "auto" else None,
            target_gpu_index=self.config.get("selected_gpu_index", 0)
        )
        current_actual_hw_vram_mb = current_gpu_full_info.get("free_mb", 0.0) # Actual free hardware VRAM

        is_moe = self.current_tuning_model_analysis.get('is_moe', False)
        estimated_vram_needed_gb = self.current_tuning_model_analysis.get('estimated_vram_gb_full_gpu', 0)
        estimated_vram_needed_mb = float(estimated_vram_needed_gb) * 1024 if estimated_vram_needed_gb else 0.0

        if is_moe:
            self.current_tuning_min_level, self.current_tuning_max_level, initial_heuristic_level = -25, 10, -10 # MoE range
        else: # Dense model
            self.current_tuning_min_level, self.current_tuning_max_level = -17, 9 # Dense range
            size_b_val = self.current_tuning_model_analysis.get('size_b', 0)
            size_b = 0.0
            if isinstance(size_b_val, (int, float)): size_b = float(size_b_val)
            elif isinstance(size_b_val, str):
                try: size_b = float(size_b_val)
                except ValueError: size_b = 0.0 # Default if parse fails

            if size_b >= 30: initial_heuristic_level = -3
            elif size_b >= 20: initial_heuristic_level = -5
            else: initial_heuristic_level = -7
        
        # VRAM-based heuristic adjustment
        safety_buffer_mb = float(self.config.get("vram_safety_buffer_mb", 768))
        min_free_after_load_mb = float(self.config.get("min_vram_free_after_load_success_mb", 512))
        effective_vram_budget_for_heuristic_mb = current_gpu_full_info.get("total_mb_budgeted", 0.0) - safety_buffer_mb - min_free_after_load_mb

        if estimated_vram_needed_mb > 0 and current_vram_budgeted > 0: # current_vram_budgeted is free_mb_budgeted
            if estimated_vram_needed_mb > effective_vram_budget_for_heuristic_mb * 1.1: # Needs more than available comfortably
                initial_heuristic_level = max(initial_heuristic_level, -3 if not is_moe else -6) # Shift towards CPU
                self.log_to_console("Heuristic: Est. VRAM > effective budget. Adjusting OT towards CPU.")
            elif estimated_vram_needed_mb < effective_vram_budget_for_heuristic_mb * 0.7: # Plenty of VRAM
                initial_heuristic_level = min(initial_heuristic_level, -12 if not is_moe else -18) # Shift towards GPU
                self.log_to_console("Heuristic: Ample VRAM budget. Adjusting OT towards GPU.")

        # Check history
        best_hist_config = tensortune_core.find_best_historical_config(
            self.db_path, self.current_tuning_model_analysis, current_actual_hw_vram_mb, self.config
        )
        if best_hist_config and "attempt_level" in best_hist_config:
                self.log_to_console(f"Found historical config. Level: {best_hist_config['attempt_level']}, Outcome: {best_hist_config['outcome']}")
        # Update last_successful_monitored_run_details_gui from history
        if best_hist_config and best_hist_config.get("outcome", "").startswith("SUCCESS"):
            self.last_successful_monitored_run_details_gui = {
                "level": best_hist_config['attempt_level'],
                "outcome": best_hist_config['outcome'],
                "vram_used_mb": f"{best_hist_config.get('approx_vram_used_kcpp_mb', 'N/A')}" 
            }
            # Update the info label with the historical result
            if hasattr(self, 'tuning_last_run_info_label') and self.tuning_last_run_info_label.winfo_exists():
                outcome_disp = best_hist_config.get('outcome', 'N/A')
                vram_used_disp = best_hist_config.get('approx_vram_used_kcpp_mb', 'N/A')
                level_disp = best_hist_config.get('attempt_level', 'N/A')
                self.tuning_last_run_info_label.configure(
                    text=f"Last Session Result (from history): Level {level_disp}, Outcome: {outcome_disp}, VRAM Used: {vram_used_disp}MB",
                    text_color="darkgreen"
                )            
            hist_level = best_hist_config['attempt_level']
            hist_outcome = best_hist_config.get('outcome', "")
            approx_hist_vram_used = best_hist_config.get('approx_vram_used_kcpp_mb')

            # Adjust starting level based on historical outcome and current VRAM
            # Handle starting level based on historical outcome and current VRAM - prioritize exact match for USER_MARKED_AS_BEST
            if hist_outcome.endswith("_USER_MARKED_AS_BEST_GUI") or hist_outcome.endswith("_USER_MARKED_AS_BEST_CLI"):
                # For preferred configurations, use exactly the same level
                initial_heuristic_level = hist_level
                self.log_to_console(f"Using exact level {hist_level} from preferred historical configuration")
            elif approx_hist_vram_used is not None and (float(approx_hist_vram_used) + safety_buffer_mb < current_actual_hw_vram_mb):
                # For other successful configs with room to spare, allow slight GPU adjustment
                initial_heuristic_level = max(self.current_tuning_min_level, hist_level - 1 if hist_level > self.current_tuning_min_level else hist_level)
                self.log_to_console(f"Historical success fits actual VRAM. Starting near: {initial_heuristic_level}")
            elif hist_outcome.startswith("SUCCESS_LOAD_VRAM_OK") or hist_outcome.startswith("SUCCESS_USER_CONFIRMED") or hist_outcome.endswith("_USER_SAVED_GOOD_GPU_GUI"):
                # For other successful configs, also use the exact level to be safe
                initial_heuristic_level = hist_level
                self.log_to_console(f"Using exact historical level {hist_level} for successful configuration")
            elif hist_outcome.endswith("_USER_AUTO_ADJUST_CPU_GUI") or hist_outcome.endswith("_USER_TRIED_CPU_FAIL_GUI") or "OOM" in hist_outcome.upper() or "TIGHT" in hist_outcome.upper():
                initial_heuristic_level = min(self.current_tuning_max_level, hist_level + 1 if hist_level < self.current_tuning_max_level else hist_level)
            else: # For other outcomes, start at the historical level
                initial_heuristic_level = hist_level
            
            # Apply remembered args from history if any
            remembered_args_list = best_hist_config.get("args_list", [])
            if remembered_args_list:
                remembered_args_dict = tensortune_core.args_list_to_dict(remembered_args_list)
                remembered_args_dict.pop("--model", None) # Don't override model path
                remembered_args_dict.pop("--overridetensors", None) # OT string is handled by level
                self.current_tuning_session_base_args.update(remembered_args_dict)
                self.log_to_console(f"Applied remembered arguments to current session base. OT Level target adjusted to: {initial_heuristic_level}")
        else:
            self.log_to_console(f"No suitable historical config found. Starting with heuristic OT Level: {initial_heuristic_level}")

        self.current_tuning_attempt_level = max(self.current_tuning_min_level, min(initial_heuristic_level, self.current_tuning_max_level))
        self.level_of_last_monitored_run = self.current_tuning_attempt_level # Store initial level
        # Check if this is our first update after loading a historical "best" config
        if hasattr(self, 'last_successful_monitored_run_details_gui') and self.last_successful_monitored_run_details_gui:
            historical_level = self.last_successful_monitored_run_details_gui.get('level')
            historical_outcome = self.last_successful_monitored_run_details_gui.get('outcome', '')
            
            # If we have a preferred/best historical config, prioritize using that exact level
            if historical_outcome and historical_level is not None and '_USER_MARKED_AS_BEST_' in historical_outcome:
                if self.current_tuning_attempt_level != historical_level:
                    self.log_to_console(f"Adjusting to exact historical preferred level: {historical_level}")
                    self.current_tuning_attempt_level = historical_level
        # Ensure level is within bounds
        self.current_tuning_attempt_level = max(self.current_tuning_min_level, min(self.current_tuning_attempt_level, self.current_tuning_max_level))    
        self.manual_gpu_layers_var.set(True)  # Default to Auto
        self.manual_gpu_layers_entry_var.set("")
        self._on_auto_gpu_layers_toggle() # Update entry state
        self.effective_gpu_layers_for_command.set("auto")
        
        self._show_tuning_mode_view()
        #self.update_tuning_display() # Called by _show_tuning_mode_view via _return_to_full_tuning_menu
        
        if hasattr(self, 'kcpp_live_output_text') and self.kcpp_live_output_text.winfo_exists():
            self.kcpp_live_output_text.configure(state="normal")
            self.kcpp_live_output_text.delete("1.0", "end") # Clear previous output
            self.kcpp_console_line_count = 0
            # No initial message here, update_tuning_display will populate command

    def get_qualitative_ot_level_desc(self, level: int, is_moe: bool) -> str:
        if is_moe: # MoE thresholds
            if level <= -18: return "Strong GPU Bias"
            elif level <= -10: return "GPU Biased"
            elif level <= -2: return "Balanced GPU/CPU"
            elif level <= 5: return "CPU Biased"
            else: return "Strong CPU Bias"
        else: # Dense model thresholds
            if level <= -11: return "Strong GPU Bias"
            elif level <= -5: return "GPU Biased"
            elif level <= 0: return "Balanced GPU/CPU"
            elif level <= 5: return "CPU Biased"
            else: return "Strong CPU Bias"

    def update_tuning_display(self):
        if not self.tuning_in_progress:
            return
        
        # Ensure level is within bounds
        self.current_tuning_attempt_level = max(self.current_tuning_min_level, min(self.current_tuning_attempt_level, self.current_tuning_max_level))

        if hasattr(self, 'tuning_model_name_label') and self.tuning_model_name_label.winfo_exists() and self.current_tuning_model_path:
            self.tuning_model_name_label.configure(text=f"Model: {os.path.basename(self.current_tuning_model_path)}")
        
        # Update "Last Monitored Result" label
        if hasattr(self, 'tuning_last_run_info_label') and self.tuning_last_run_info_label.winfo_exists():
            # [this section remains unchanged]
            pass  # Keep existing code for this section

        ot_string = tensortune_core.generate_overridetensors(self.current_tuning_model_analysis, self.current_tuning_attempt_level)
        description = tensortune_core.get_offload_description(self.current_tuning_model_analysis, self.current_tuning_attempt_level, ot_string)
        
        # KEY FIX: Properly handle manual GPU layers
        effective_gpu_layers_value = "auto"  # Default to auto
        if hasattr(self, 'manual_gpu_layers_var') and not self.manual_gpu_layers_var.get():  # If "Auto GPU Layers" is UNCHECKED
            manual_entry_val = self.manual_gpu_layers_entry_var.get().strip()
            if manual_entry_val.isdigit():
                effective_gpu_layers_value = int(manual_entry_val)
            elif manual_entry_val.lower() == "auto":  # User typed "auto" in manual box
                effective_gpu_layers_value = "auto"
        
        # Store for command building
        self.effective_gpu_layers_for_command.set(str(effective_gpu_layers_value))

        # Determine display and actual layers for OT string calculation if needed
        gpu_layers_for_ot_calc_display = 0
        if isinstance(effective_gpu_layers_value, int):
            gpu_layers_for_ot_calc_display = effective_gpu_layers_value
        else:  # It's "auto" or invalid, use level-based for OT and display
            gpu_layers_for_ot_calc_display = tensortune_core.get_gpu_layers_for_level(
                self.current_tuning_model_analysis, self.current_tuning_attempt_level
            )
        total_layers = self.current_tuning_model_analysis.get('num_layers', 32)  # Default if not found

        # Update UI elements
        if hasattr(self, 'tuning_ot_level_label'):
            self.tuning_ot_level_label.configure(text=f"Level: {self.current_tuning_attempt_level}")
        if hasattr(self, 'tuning_ot_qualitative_desc_label'):
            qual_desc = self.get_qualitative_ot_level_desc(self.current_tuning_attempt_level, self.current_tuning_model_analysis.get('is_moe', False))
            self.tuning_ot_qualitative_desc_label.configure(text=f"({qual_desc})")
        
        range_text_suffix = 'SuperMaxCPU' if self.current_tuning_model_analysis.get('is_moe') else 'MaxCPU'
        range_text = f"{self.current_tuning_min_level}=MaxGPU ... {self.current_tuning_max_level}={range_text_suffix}"
        if hasattr(self, 'tuning_ot_range_label'):
            self.tuning_ot_range_label.configure(text=f"Range: {range_text}")
        if hasattr(self, 'tuning_ot_strategy_desc_label'):
            self.tuning_ot_strategy_desc_label.configure(text=f"Strategy: {description}")
        
        regex_display = ot_string if ot_string else "None (GPU layers set by level)"
        if hasattr(self, 'tuning_ot_regex_label'):
            self.tuning_ot_regex_label.configure(text=f"Regex: {regex_display}")
        
        if hasattr(self, 'tuning_gpu_layers_label'):
            display_gpu_layers_text = str(self.effective_gpu_layers_for_command.get())
            if display_gpu_layers_text.lower() == "auto":
                auto_calculated_layers = tensortune_core.get_gpu_layers_for_level(
                    self.current_tuning_model_analysis, self.current_tuning_attempt_level
                )
                display_gpu_layers_text = f"Auto ({auto_calculated_layers})"
            self.tuning_gpu_layers_label.configure(text=f"Effective GPU Layers: {display_gpu_layers_text}/{total_layers}")

        # Build and display proposed command - IMPORTANT FIX HERE
        manual_gpu_override_for_command = None
        effective_gpu_setting = self.effective_gpu_layers_for_command.get()
        
        # Only convert to int if it's a digit string
        if effective_gpu_setting and effective_gpu_setting.isdigit():
            manual_gpu_override_for_command = int(effective_gpu_setting)
        
        args_for_kcpp_list = tensortune_core.build_command(
            self.current_tuning_model_path,
            ot_string,
            self.current_tuning_model_analysis,
            self.current_tuning_session_base_args,
            current_attempt_level_for_tuning=self.current_tuning_attempt_level,
            manual_gpu_layers_override=manual_gpu_override_for_command 
        )
        
        full_command_list = tensortune_core.get_command_to_run(self.koboldcpp_executable, args_for_kcpp_list)
        display_command_str = tensortune_core.format_command_for_display(full_command_list)

        if hasattr(self, 'tuning_proposed_command_text'):
            self.tuning_proposed_command_text.configure(state="normal")
            self.tuning_proposed_command_text.delete("1.0", "end")
            self.tuning_proposed_command_text.insert("1.0", display_command_str)
            self.tuning_proposed_command_text.configure(state="disabled")
        
        self._set_tuning_buttons_state("normal")  # Enable buttons now that display is updated

    def _setup_manual_gpu_layers_controls(self):
        # This should be called during setup_main_tab or similar initialization
        
        # Set up the handler for changes to manual_gpu_layers_entry
        def on_manual_gpu_layers_changed(*args):
            if not self.manual_gpu_layers_var.get():  # Only update if Auto is unchecked
                self.update_tuning_display()
        
        # Bind to the StringVar for real-time updates
        self.manual_gpu_layers_entry_var.trace_add("write", on_manual_gpu_layers_changed)

    # Replace the existing _on_auto_gpu_layers_toggle function
    def _on_auto_gpu_layers_toggle(self):
        if hasattr(self, 'manual_gpu_layers_entry') and self.manual_gpu_layers_entry.winfo_exists():
            if self.manual_gpu_layers_var.get():  # Auto is checked
                self.manual_gpu_layers_entry.configure(state="disabled")
            else:  # Manual is selected
                self.manual_gpu_layers_entry.configure(state="normal")
        
        if self.tuning_in_progress:
            self.update_tuning_display()
    
    def _handle_monitoring_completion(self, initial_outcome_key: str):
        self.log_to_console(f"Monitoring completed. Initial Outcome: {initial_outcome_key}")
        self._log_to_kcpp_live_output(f"\n--- Monitoring Finished: {initial_outcome_key} ---\n")
        self.user_requested_stop_monitoring = False # Reset flag

        if initial_outcome_key in ["TIMEOUT_NO_SIGNAL_GUI", "OOM_CRASH_DETECTED_GUI", "PREMATURE_EXIT_GUI", "USER_STOPPED_MONITORING_GUI"] \
           or "OOM" in initial_outcome_key.upper() or "CRASH" in initial_outcome_key.upper():
            if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
                self.log_to_console("Terminating KCPP process due to unfavorable outcome or user stop...")
                tensortune_core.kill_process(self.kcpp_process_obj.pid, force=True)
            self.kcpp_process_obj = None

        self.last_approx_vram_used_kcpp_mb = None
        self.last_free_vram_after_load_mb = None
        final_db_outcome = initial_outcome_key

        if initial_outcome_key == "SUCCESS_LOAD_DETECTED_GUI":
            self._log_to_kcpp_live_output("API detected. Waiting for VRAM to stabilize...\n")
            stabilization_wait_s = float(self.config.get("vram_stabilization_wait_s", 3.0))
            time.sleep(max(2.0, stabilization_wait_s))

            selected_gpu_type_for_vram = self.config.get("gpu_selection_mode", "auto")
            target_type = selected_gpu_type_for_vram if selected_gpu_type_for_vram != "auto" else None
            target_idx = self.config.get("selected_gpu_index", 0)
            _, _, _, gpu_info_after_load = tensortune_core.get_available_vram_mb(self.config, target_type, target_idx)

            budgeted_free_after_load_raw = gpu_info_after_load.get("free_mb_budgeted")
            actual_hw_free_after_load_raw = gpu_info_after_load.get("free_mb")
            actual_hw_total_raw = gpu_info_after_load.get("total_mb")
            vram_at_decision_raw = self.vram_at_decision_for_db

            # Convert to float, defaulting to 0.0 if None or not convertible
            self.last_free_vram_after_load_mb = float(budgeted_free_after_load_raw) if isinstance(budgeted_free_after_load_raw, (int, float)) else 0.0
            actual_hw_free_after_load_num = float(actual_hw_free_after_load_raw) if isinstance(actual_hw_free_after_load_raw, (int, float)) else 0.0
            actual_hw_total_num = float(actual_hw_total_raw) if isinstance(actual_hw_total_raw, (int, float)) else 0.0
            vram_at_decision_num = float(vram_at_decision_raw) if isinstance(vram_at_decision_raw, (int, float)) else 0.0
            
            min_vram_free_target = float(self.config.get("min_vram_free_after_load_success_mb", 512))

            if actual_hw_total_num > 0 and self.vram_at_decision_for_db is not None and actual_hw_free_after_load_raw is not None:
                vram_used_by_kcpp_hw = vram_at_decision_num - actual_hw_free_after_load_num
                self.last_approx_vram_used_kcpp_mb = max(0, min(vram_used_by_kcpp_hw, actual_hw_total_num))
                
                self._log_to_kcpp_live_output(
                    f"VRAM After Load (Budgeted Free): {self.last_free_vram_after_load_mb:.0f}MB. "
                    f"Approx Actual KCPP VRAM Used: {self.last_approx_vram_used_kcpp_mb:.0f}MB\n"
                )
                
                if self.last_free_vram_after_load_mb < min_vram_free_target:
                    self._log_to_kcpp_live_output(f"WARNING: Budgeted VRAM tight! {self.last_free_vram_after_load_mb:.0f}MB free < {min_vram_free_target}MB target.\n")
                    final_db_outcome = "SUCCESS_LOAD_VRAM_TIGHT_GUI"
                else:
                    self._log_to_kcpp_live_output("Budgeted VRAM usage OK.\n")
                    final_db_outcome = "SUCCESS_LOAD_VRAM_OK_GUI"

                if gpu_info_after_load.get("override_active", False) and self.last_approx_vram_used_kcpp_mb > gpu_info_after_load.get("total_mb_budgeted", 0):
                     self._log_to_kcpp_live_output(
                         f"NOTE: Actual KCPP VRAM usage ({self.last_approx_vram_used_kcpp_mb:.0f}MB) "
                         f"exceeded the manual VRAM budget ({gpu_info_after_load.get('total_mb_budgeted', 0):.0f}MB).\n"
                     )
            else: # VRAM info insufficient for detailed check
                self._log_to_kcpp_live_output("VRAM check after load skipped (total VRAM is zero, or other values missing for calculation).\n")
                final_db_outcome = "SUCCESS_LOAD_NO_VRAM_CHECK_GUI" 
            
            self.last_successful_monitored_run_details_gui = {
                "level": self.level_of_last_monitored_run,
                "outcome": final_db_outcome,
                "vram_used_mb": f"{self.last_approx_vram_used_kcpp_mb:.0f}" if self.last_approx_vram_used_kcpp_mb is not None else "N/A"
            }
            self.update_tuning_display()

        tensortune_core.save_config_to_db(
            self.db_path,
            self.current_tuning_model_path,
            self.current_tuning_model_analysis,
            self.vram_at_decision_for_db,
            self.current_command_list_for_db,
            self.level_of_last_monitored_run,
            final_db_outcome,
            self.last_approx_vram_used_kcpp_mb
        )
        self.load_history()
        self._present_post_monitoring_choices(final_db_outcome)


        # Save to DB
        tensortune_core.save_config_to_db(
            self.db_path,
            self.current_tuning_model_path,
            self.current_tuning_model_analysis,
            self.vram_at_decision_for_db, # Actual free VRAM before this monitored launch
            self.current_command_list_for_db, # The command that was run
            self.level_of_last_monitored_run, # The OT level for this run
            final_db_outcome,
            self.last_approx_vram_used_kcpp_mb # Approx VRAM used by KCPP for this run
        )
        self.load_history() # Refresh history tab
        self._present_post_monitoring_choices(final_db_outcome)

    def _present_post_monitoring_choices(self, outcome: str):
        # Hide primary tuning action frames
        frames_to_hide_names = [
            'tuning_actions_primary_frame', 'tuning_stop_monitor_frame',
            'tuning_actions_secondary_frame', 'tuning_edit_args_buttons_frame',
            'tuning_actions_navigation_frame'
        ]
        for frame_name in frames_to_hide_names:
            frame_widget = getattr(self, frame_name, None)
            if frame_widget and hasattr(frame_widget, 'winfo_exists') and frame_widget.winfo_exists():
                frame_widget.grid_remove()

        # Setup or reuse the post_monitor_choices_frame
        if not hasattr(self, 'post_monitor_choices_frame') or not self.post_monitor_choices_frame.winfo_exists():
            self.post_monitor_choices_frame = ctk.CTkFrame(self.tuning_mode_scrollable_content_frame)
            self.post_monitor_choices_frame.grid_columnconfigure(0, weight=1)
        
        # Fix for Issue 4b: Grid post_monitor_choices_frame at row 7 (was 6)
        self.post_monitor_choices_frame.grid(row=7, column=0, rowspan=5, padx=10, pady=5, sticky="news")


        for widget in self.post_monitor_choices_frame.winfo_children():
            widget.destroy()

        ctk.CTkLabel(self.post_monitor_choices_frame, text=f"Outcome: {outcome}", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 2), anchor="w", padx=5)
        
        if self.last_free_vram_after_load_mb is not None:
            ctk.CTkLabel(self.post_monitor_choices_frame, text=f"Budgeted VRAM After Load: {self.last_free_vram_after_load_mb:.0f} MB free").pack(pady=1, anchor="w", padx=5)
        if self.last_approx_vram_used_kcpp_mb is not None:
            ctk.CTkLabel(self.post_monitor_choices_frame, text=f"Approx. Actual KCPP VRAM Used: {self.last_approx_vram_used_kcpp_mb:.0f} MB").pack(pady=1, anchor="w", padx=5)

        vram_status_message = "VRAM Status: Check KCPP Log for details."
        if "SUCCESS_LOAD_VRAM_OK" in outcome: vram_status_message = "Budgeted VRAM Usage: OK"
        elif "SUCCESS_LOAD_VRAM_TIGHT" in outcome: vram_status_message = "Budgeted VRAM Usage: TIGHT"
        elif "SUCCESS_LOAD_NO_VRAM_CHECK" in outcome: vram_status_message = "Budgeted VRAM Usage: Could not be determined by launcher."
        ctk.CTkLabel(self.post_monitor_choices_frame, text=vram_status_message).pack(pady=(1,10), anchor="w", padx=5)

        if "SUCCESS_LOAD_VRAM_OK" in outcome:
            btn_accept = ctk.CTkButton(self.post_monitor_choices_frame, text="✅ Accept & Use this KCPP instance", command=lambda: self._handle_post_monitor_action("launch_for_use", outcome))
            btn_accept.pack(fill="x", pady=3, padx=5)
            ToolTip(btn_accept, "Keep the currently running (monitored) KoboldCpp instance and use it.\nTuning session will end.")

            btn_set_preferred = ctk.CTkButton(self.post_monitor_choices_frame, text="⭐ Set as Preferred & Continue Tuning", command=lambda: self._handle_post_monitor_action("set_as_preferred", outcome))
            btn_set_preferred.pack(fill="x", pady=3, padx=5)
            ToolTip(btn_set_preferred, "Mark this configuration as the preferred one for this model in history.\nThen, continue tuning.")

            btn_save_good_gpu = ctk.CTkButton(self.post_monitor_choices_frame, text="💾 Save as Good, Auto-Adjust for More GPU & Continue Tuning", command=lambda: self._handle_post_monitor_action("save_good_more_gpu", outcome))
            btn_save_good_gpu.pack(fill="x", pady=3, padx=5)
            ToolTip(btn_save_good_gpu, "Mark this configuration as successful, try to use even more GPU,\nand continue the tuning session.")
            
            btn_manual_gpu = ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Manually Try More GPU (This Session) & Continue Tuning", command=lambda: self._handle_post_monitor_action("more_gpu_now", outcome))
            btn_manual_gpu.pack(fill="x", pady=3, padx=5)
            ToolTip(btn_manual_gpu, "Manually decrease the OT level to use more GPU and continue tuning.")

            btn_manual_cpu = ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Manually Try More CPU (This Session) & Continue Tuning", command=lambda: self._handle_post_monitor_action("more_cpu_now", outcome))
            btn_manual_cpu.pack(fill="x", pady=3, padx=5)
            ToolTip(btn_manual_cpu, "Manually increase the OT level to use more CPU and continue tuning.")

        elif "SUCCESS_LOAD_VRAM_TIGHT" in outcome:
            btn_auto_cpu_tight = ctk.CTkButton(self.post_monitor_choices_frame, text="⚠️ Auto-Adjust for More CPU & Continue Tuning", command=lambda: self._handle_post_monitor_action("auto_adjust_cpu", outcome))
            btn_auto_cpu_tight.pack(fill="x", pady=3, padx=5)
            ToolTip(btn_auto_cpu_tight, "VRAM is tight. Automatically increase OT level (more CPU) and continue tuning.")

            btn_set_preferred_tight = ctk.CTkButton(self.post_monitor_choices_frame, text="⭐ Set as Preferred (Tight) & Continue Tuning", command=lambda: self._handle_post_monitor_action("set_as_preferred", outcome))
            btn_set_preferred_tight.pack(fill="x", pady=3, padx=5)
            ToolTip(btn_set_preferred_tight, "Mark this (tight but successful) configuration as preferred for this model in history.\nThen, continue tuning.")

            btn_launch_risky = ctk.CTkButton(self.post_monitor_choices_frame, text="🚀 Launch This Config Anyway (Risky)", command=lambda: self._handle_post_monitor_action("launch_for_use_risky", outcome))
            btn_launch_risky.pack(fill="x", pady=3, padx=5)
            ToolTip(btn_launch_risky, "Launch the current (monitored) KoboldCpp instance for use, despite tight VRAM.\nThis might lead to instability.")

        elif "OOM" in outcome or "CRASH" in outcome or "PREMATURE_EXIT" in outcome or "USER_STOPPED" in outcome:
            btn_auto_cpu_fail = ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Auto-Adjust for More CPU & Continue Tuning", command=lambda: self._handle_post_monitor_action("more_cpu_after_fail", outcome))
            btn_auto_cpu_fail.pack(fill="x", pady=3, padx=5)
            ToolTip(btn_auto_cpu_fail, "Launch failed (OOM/Crash/User Stop). Automatically increase OT level (more CPU) and continue tuning.")
        
        elif "TIMEOUT" in outcome:
            ctk.CTkLabel(self.post_monitor_choices_frame, text="KCPP launch timed out without a clear success/OOM signal.").pack(pady=3, anchor="w", padx=5)
            btn_auto_cpu_timeout = ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Auto-Adjust for More CPU (Assume OOM) & Continue Tuning", command=lambda: self._handle_post_monitor_action("more_cpu_after_fail", outcome))
            btn_auto_cpu_timeout.pack(fill="x", pady=3, padx=5)
            ToolTip(btn_auto_cpu_timeout, "Launch timed out. Assume it was an OOM or similar issue, increase OT level (more CPU), and continue tuning.")
        
        else: # Generic outcome, or SUCCESS_LOAD_NO_VRAM_CHECK_GUI
            if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
                btn_accept_generic = ctk.CTkButton(self.post_monitor_choices_frame, text="✅ Keep KCPP Running & Use", command=lambda: self._handle_post_monitor_action("launch_for_use", outcome))
                btn_accept_generic.pack(fill="x", pady=3, padx=5)
                ToolTip(btn_accept_generic, "The monitored KCPP instance is still running. Accept and use it.")

        btn_return_menu = ctk.CTkButton(self.post_monitor_choices_frame, text="↩️ Save Outcome & Return to Tuning Menu (Manual Adjust)", command=lambda: self._handle_post_monitor_action("return_to_tuning_menu", outcome))
        btn_return_menu.pack(fill="x", pady=3, padx=5)
        ToolTip(btn_return_menu, "Save the outcome of this monitored run and return to the main tuning strategy screen\nfor manual adjustments or to try other actions.")
        
        self.log_to_console("Presented post-monitoring choices to user.")


          
    def _return_to_full_tuning_menu(self):
        if hasattr(self, 'post_monitor_choices_frame') and self.post_monitor_choices_frame.winfo_exists():
            self.post_monitor_choices_frame.grid_remove() # Hide choices frame

        # Fix for Issue 4c: Restore frames to their original row indices
        # Original row indices during setup_main_tab:
        # tuning_actions_primary_frame: row 7
        # tuning_stop_monitor_frame: row 8 (handled by _set_tuning_buttons_state)
        # tuning_actions_secondary_frame: row 9
        # tuning_edit_args_buttons_frame: row 10
        # tuning_actions_navigation_frame: row 11
        frames_config = [
            (self.tuning_actions_primary_frame, 7, (5, 2)),
            (self.tuning_actions_secondary_frame, 9, 0), 
            (self.tuning_edit_args_buttons_frame, 10, 2),
            (self.tuning_actions_navigation_frame, 11, 2)
        ]
        for frame_widget, row_idx, pady_val in frames_config:
            if frame_widget and hasattr(frame_widget, 'winfo_exists') and frame_widget.winfo_exists():
                frame_widget.grid(row=row_idx, column=0, padx=10, pady=pady_val, sticky="ew")
        
        self._set_tuning_buttons_state("normal", monitoring_active=False) 
        self.update_tuning_display() 

    def _run_first_time_setup_if_needed(self):
        if not self.config.get("first_run_completed", False):
            self.log_to_console("Performing first-time setup...")
            dialog = FirstTimeSetupDialog(self, self.config, self.koboldcpp_executable)
            self.wait_window(dialog) # Blocks until dialog is closed
            self.focus_set()

            if dialog.saved_config:
                self.config = dialog.saved_config # Update app's config with dialog's result
                self.koboldcpp_executable = self.config["koboldcpp_executable"]
                self.default_model_dir = self.config.get("default_gguf_dir", "")
                self.db_path = self.config["db_file"] # Ensure this is updated
                
                # Set last_used_gguf_dir if default_gguf_dir was set
                if self.config.get("default_gguf_dir"):
                    self.config["last_used_gguf_dir"] = self.config["default_gguf_dir"]
                
                self.load_settings_from_config() # Reload UI with new settings
                self.log_to_console("First-time setup complete. Configuration updated.")
                self.update_kcpp_capabilities_display(re_detect=True) # Detect caps for new exe
                self.after(100, self._populate_gpu_id_dropdown_on_startup) # Refresh GPU list
            else:
                self.log_to_console("First-time setup cancelled or not completed.")
                messagebox.showwarning("Setup Incomplete", "First-time setup was not completed. Please configure the KoboldCpp executable in Settings.", parent=self)


    def _get_merged_args_for_model(self, model_path):
        # Start with core defaults
        merged_args = tensortune_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
        # Overlay global defaults from config
        merged_args.update(self.config.get("default_args", {}))
        # Overlay model-specific args if path is provided and config exists
        if model_path:
            merged_args.update(self.config.get("model_specific_args", {}).get(model_path, {}))
        return merged_args

    def _reinitialize_session_base_args(self):
        # Get args considering global and model-specific settings for the current tuning model
        if self.current_tuning_model_path:
            effective_args_base = self._get_merged_args_for_model(self.current_tuning_model_path)
        else: # Fallback if no model path (should not happen in normal tuning)
            effective_args_base = {
                **tensortune_core.DEFAULT_CONFIG_TEMPLATE["default_args"], 
                **self.config.get("default_args", {})
            }
        
        self.current_tuning_session_base_args = {} # Clear previous session overrides
        # Populate with properly typed values
        for k, v_raw in effective_args_base.items():
            v = v_raw
            arg_def = next((d for d in tensortune_core.KOBOLDCPP_ARG_DEFINITIONS if d["key"] == k), None)
            is_bool_type = arg_def and arg_def.get("type_hint") in ["bool", "bool_flag"]

            if is_bool_type:
                if isinstance(v_raw, bool): v = v_raw
                elif isinstance(v_raw, str): v = v_raw.lower() == 'true'
                else: v = False # Default to false if type is unexpected
                self.current_tuning_session_base_args[k] = v
            elif v is not None: # For other types, store as is, or string-stripped
                self.current_tuning_session_base_args[k] = str(v).strip() if isinstance(v, str) else v
        
        model_name_display = os.path.basename(self.current_tuning_model_path) if self.current_tuning_model_path else "No Model"
        self.log_to_console(f"Session base arguments reinitialized for '{model_name_display}'.")

    def check_koboldcpp_executable(self):
        current_exe_path_from_gui = ""
        config_needs_update_due_to_autocorrect = False

        if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
            current_exe_path_from_gui = self.exe_path_entry.get().strip()

        current_path_to_validate = current_exe_path_from_gui if current_exe_path_from_gui else self.koboldcpp_executable
        original_configured_path = self.koboldcpp_executable # Store for comparison

        resolved_path, message = tensortune_core.validate_and_resolve_koboldcpp_exe_path(current_path_to_validate)
        self.log_to_console(f"KCPP Exe Validation: {message}")

        if resolved_path:
            self.koboldcpp_executable = resolved_path # Update internal variable
            # If resolved path is different from original config or GUI entry, update GUI and mark for save
            if self.koboldcpp_executable != original_configured_path or \
               (hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists() and self.exe_path_entry.get() != self.koboldcpp_executable):
                config_needs_update_due_to_autocorrect = True
                if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
                    self.exe_path_entry.delete(0, "end")
                    self.exe_path_entry.insert(0, self.koboldcpp_executable)
        else: # Validation failed, use the path user provided (or current internal if GUI empty)
            self.koboldcpp_executable = current_path_to_validate
        
        # Update config if internal koboldcpp_executable has changed
        if self.config.get("koboldcpp_executable") != self.koboldcpp_executable:
            self.config["koboldcpp_executable"] = self.koboldcpp_executable
            if config_needs_update_due_to_autocorrect: # Auto-correction implies save
                self.log_to_console("KoboldCpp exe path updated in config.")
                self.save_config() # Save the config if path was auto-corrected
            else: # Path changed due to user input in settings, mark dirty for manual save
                self.mark_settings_dirty()
          
    def _show_model_selection_view(self):
        if hasattr(self, 'tuning_view_container_frame') and self.tuning_view_container_frame.winfo_exists():
            self.tuning_view_container_frame.grid_remove()
        
        if hasattr(self, 'model_selection_frame') and self.model_selection_frame.winfo_exists():
            self.model_selection_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        else: 
            self.setup_main_tab() 
            if hasattr(self, 'model_selection_frame'): 
                self.model_selection_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.log_to_console("Switched to Model Selection view.")
        if hasattr(self, 'tabview'):
            self.tabview.set("Tune & Launch") 

    

          
    def _show_tuning_mode_view(self):
        if hasattr(self, 'model_selection_frame') and self.model_selection_frame.winfo_exists():
            self.model_selection_frame.grid_remove()
            
        if hasattr(self, 'tuning_view_container_frame') and self.tuning_view_container_frame.winfo_exists():
            self.tuning_view_container_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        else: 
            self.setup_main_tab()
            if hasattr(self, 'tuning_view_container_frame'): 
                 self.tuning_view_container_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.log_to_console("Switched to Tuning Mode view.")
        if hasattr(self, 'tabview'):
            self.tabview.set("Tune & Launch")
        self._return_to_full_tuning_menu() 
    
    def log_gpu_layers_mapping(self):
        """Debug helper to log how OT levels map to GPU layer counts for current model"""
        if not self.tuning_in_progress or not self.current_tuning_model_analysis:
            self.log_to_console("No active tuning session - cannot display GPU layer mapping")
            return
        
        model_layers = self.current_tuning_model_analysis.get('num_layers', 32)
        is_moe = self.current_tuning_model_analysis.get('is_moe', False)
        
        self.log_to_console(f"GPU Layer mapping for current model ({model_layers} total layers, {'MoE' if is_moe else 'Dense'}):")
        
        # Create a mapping table for a reasonable range of levels
        level_range = range(self.current_tuning_min_level, self.current_tuning_max_level + 1)
        
        result_table = []
        last_layers = None
        
        for level in level_range:
            layers = tensortune_core.get_gpu_layers_for_level(
                self.current_tuning_model_analysis, level
            )
            
            # Mark significant jumps
            jump_marker = ""
            if last_layers is not None:
                if layers == 999 and last_layers < 100:
                    jump_marker = "⚠️ BIG JUMP"
                elif layers > last_layers * 2:
                    jump_marker = "⬆️ Large increase"
            
            result_table.append(f"Level {level}: {layers} GPU layers {jump_marker}")
            last_layers = layers
        
        # Log in batches to avoid flooding console
        batch_size = 5
        for i in range(0, len(result_table), batch_size):
            batch = result_table[i:i+batch_size]
            self.log_to_console("  " + " | ".join(batch))

    def adjust_ot_level(self, delta):
        if not self.tuning_in_progress:
            return
        
        # Get current GPU layers for context
        current_model_layers = self.current_tuning_model_analysis.get('num_layers', 32)
        is_moe = self.current_tuning_model_analysis.get('is_moe', False)
        current_gpu_layers = tensortune_core.get_gpu_layers_for_level(
            self.current_tuning_model_analysis, self.current_tuning_attempt_level
        )
        
        # Store original level for logging
        original_level = self.current_tuning_attempt_level
        
        # Check if we're about to make a jump to 999 layers (common threshold)
        next_level = self.current_tuning_attempt_level + delta
        
        # Always allow More CPU even if at level boundary
        if delta > 0:  # More CPU
            self.current_tuning_attempt_level = min(self.current_tuning_attempt_level + delta, self.current_tuning_max_level)
            self.update_tuning_display()
            return
        
        # For More GPU (delta < 0)
        if next_level < self.current_tuning_min_level:
            # Already at min level, can't go further
            self.log_to_console(f"Already at maximum GPU level ({self.current_tuning_min_level})")
            return
            
        # Check what would happen at the next level
        next_gpu_layers = tensortune_core.get_gpu_layers_for_level(
            self.current_tuning_model_analysis, next_level
        )
        
        # Special handling for the big jump to 999
        if delta < 0 and next_gpu_layers == 999 and current_gpu_layers < 100:
            # Calculate an intermediate value (approximately 30% more layers)
            target_increment = min(10, max(3, int(current_model_layers * 0.1)))  # At least 3, at most 10 more layers
            target_layers = min(current_gpu_layers + target_increment, current_model_layers - 1)
            
            self.log_to_console(f"Adding intermediate step between {current_gpu_layers} and 999 layers: using {target_layers} layers")
            
            # Switch to manual mode and set the value
            self.manual_gpu_layers_var.set(False)  # Uncheck Auto
            self.manual_gpu_layers_entry_var.set(str(int(target_layers)))
            
            # Update UI state
            if hasattr(self, 'manual_gpu_layers_entry') and self.manual_gpu_layers_entry.winfo_exists():
                self.manual_gpu_layers_entry.configure(state="normal")
            
            # Also update the level anyway - this prevents getting stuck
            self.current_tuning_attempt_level = next_level
        else:
            # Standard adjustment
            self.current_tuning_attempt_level = next_level
            
            # If manual mode was on, switch back to auto
            if not self.manual_gpu_layers_var.get():
                self.manual_gpu_layers_var.set(True)
                if hasattr(self, 'manual_gpu_layers_entry') and self.manual_gpu_layers_entry.winfo_exists():
                    self.manual_gpu_layers_entry.configure(state="disabled")
        
        # Ensure we stay within bounds (redundant check)
        self.current_tuning_attempt_level = max(self.current_tuning_min_level, 
                                               min(self.current_tuning_attempt_level, 
                                                   self.current_tuning_max_level))
        
        # Get new GPU layers count for logging
        if self.manual_gpu_layers_var.get():  # Auto mode
            new_gpu_layers = tensortune_core.get_gpu_layers_for_level(
                self.current_tuning_model_analysis, self.current_tuning_attempt_level
            )
        else:  # Manual mode
            try:
                new_gpu_layers = int(self.manual_gpu_layers_entry_var.get())
            except (ValueError, TypeError):
                new_gpu_layers = "manual (invalid)"
        
        # Log the change for user awareness
        self.log_to_console(f"OT Level adjusted: {original_level} → {self.current_tuning_attempt_level}. GPU layers: {current_gpu_layers} → {new_gpu_layers}/{current_model_layers}")
        
        # Update display with new level
        self.update_tuning_display()

    def edit_base_args_for_tuning_session(self):
        """Edit base arguments for the current tuning session with robust error handling."""
        if not self.tuning_in_progress:
            messagebox.showwarning("Not Tuning", "Tuning session is not active.", parent=self)
            return
        if not self.current_tuning_model_path: 
            messagebox.showerror("Error", "No model associated with current tuning session.", parent=self)
            return

        try:
            dialog = ctk.CTkToplevel(self)
            dialog.title("Edit Base Args (Current Tuning Session)")
            dialog.geometry("800x700")
            dialog.transient(self)
            dialog.grab_set()
            dialog.attributes("-topmost", True)

            param_defs = self._get_param_definitions_for_dialog()
            
            effective_base_for_session = self._get_merged_args_for_model(self.current_tuning_model_path)
            current_display_args = effective_base_for_session.copy()
            current_display_args.update(self.current_tuning_session_base_args) 

            dialog_main_frame = ctk.CTkFrame(dialog)
            dialog_main_frame.pack(fill="both", expand=True)
            dialog_main_frame.grid_columnconfigure(0, weight=1)
            dialog_main_frame.grid_rowconfigure(0, weight=1)

            content_frame, widgets_info = self._create_args_dialog_content_revised(dialog_main_frame, current_display_args, param_defs)
            content_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

            # Store any changes in a temporary dictionary
            temp_changes = {}

            # Define a function to collect widget values without risking focus issues
            def collect_widget_values():
                changes = {}
                try:
                    for param_key, info in widgets_info.items():
                        widget = info["widget"]
                        
                        # Skip if widget doesn't exist
                        if not (widget and hasattr(widget, 'winfo_exists') and widget.winfo_exists()):
                            continue
                            
                        type_hint = info["type_hint"]
                        new_value = None

                        if type_hint in ["bool", "bool_flag"]:
                            if hasattr(widget, 'variable'):
                                new_value = widget.variable.get()
                        else:  # CTkEntry
                            try:
                                new_value_str = widget.get().strip()
                                if not new_value_str:
                                    continue
                                new_value = new_value_str
                            except:
                                continue
                                
                        # Store only if value differs from base
                        base_value = effective_base_for_session.get(param_key)
                        if type_hint in ["bool", "bool_flag"]:
                            if isinstance(base_value, str):
                                base_value = base_value.lower() == 'true'
                            elif not isinstance(base_value, bool):
                                base_value = False
                        else:
                            base_value = str(base_value) if base_value is not None else ""
                            new_value = str(new_value) if new_value is not None else ""
                            
                        if new_value != base_value:
                            changes[param_key] = new_value
                except Exception as e:
                    self.log_to_console(f"Error collecting values: {e}")
                    
                return changes

            # Define the save function
            def save_session_args_action():
                # Collect the values first
                collected_changes = collect_widget_values()
                
                # Store in temp dictionary
                temp_changes.update(collected_changes)
                
                # Close dialog - this avoids focus issues with destroyed widgets
                try:
                    dialog.grab_release()
                    dialog.destroy()
                except:
                    pass
                    
                # Use after to apply changes AFTER dialog is fully destroyed
                self.after(100, lambda: apply_changes_after_dialog_closed(collected_changes))
                    
            # Function to apply changes after dialog is closed
            def apply_changes_after_dialog_closed(changes):
                try:
                    if not changes:
                        self.log_to_console("No changes to apply to session args.")
                        return
                        
                    # Apply each collected change to the session args
                    changes_applied = False
                    for param_key, new_value in changes.items():
                        current_value = self.current_tuning_session_base_args.get(param_key)
                        if current_value != new_value:
                            self.current_tuning_session_base_args[param_key] = new_value
                            changes_applied = True
                    
                    # Remove params that should be reset to defaults
                    # (those in base args but not in our changes and different from base)
                    for param_key in list(self.current_tuning_session_base_args.keys()):
                        if param_key not in changes:
                            base_value = effective_base_for_session.get(param_key)
                            session_value = self.current_tuning_session_base_args.get(param_key)
                            if base_value != session_value:
                                del self.current_tuning_session_base_args[param_key]
                                changes_applied = True
                    
                    if changes_applied:
                        self.log_to_console("Session base arguments updated for current tuning.")
                        # Update UI safely
                        if hasattr(self, 'update_tuning_display') and callable(self.update_tuning_display):
                            self.update_tuning_display()
                except Exception as e:
                    self.log_to_console(f"Error applying changes: {e}")

            # Define the cancel function
            def cancel_action():
                try:
                    dialog.grab_release()
                    dialog.destroy()
                except:
                    pass

            # Setup UI components
            button_frame = ctk.CTkFrame(dialog_main_frame)
            button_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
            ctk.CTkButton(button_frame, text="Apply Session Args", command=save_session_args_action).pack(side="left", padx=10)
            ctk.CTkButton(button_frame, text="Cancel", command=cancel_action).pack(side="right", padx=10)
            
            dialog.protocol("WM_DELETE_WINDOW", cancel_action)
            
        except Exception as e:
            self.log_to_console(f"Error setting up edit args dialog: {e}")
            import traceback
            self.log_to_console(traceback.format_exc())

    def edit_permanent_model_args(self):
        if not self.tuning_in_progress or not self.current_tuning_model_path:
            messagebox.showwarning("Not Available", "This option is available only during an active tuning session for a selected model.", parent=self)
            return
        self.open_model_specific_edit_dialog(self.current_tuning_model_path)

    def select_new_gguf_during_tuning(self):
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None: # Monitored process running
            confirm = messagebox.askyesno("Process Running", "A KoboldCpp monitoring process might be running. Stop it and select a new model?", parent=self)
            if not confirm:
                return
            self.log_to_console("Stopping monitored KoboldCpp process...")
            tensortune_core.kill_process(self.kcpp_process_obj.pid, force=True)
            self.kcpp_process_obj = None
        
        self.end_tuning_session(switch_to_model_selection=True) # Ends session and switches view

    def end_tuning_session(self, switch_to_model_selection=True):
        if not self.tuning_in_progress:
            return
        self.log_to_console("Ending tuning session.")
        
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
            self.log_to_console(f"Stopping any active monitored KCPP process (PID: {self.kcpp_process_obj.pid}) from tuning session...")
            tensortune_core.kill_process(self.kcpp_process_obj.pid, force=True)
            self.kcpp_process_obj = None
        
        self.tuning_in_progress = False
        self.current_tuning_session_base_args = {} # Clear session overrides
        self.last_free_vram_after_load_mb = None
        self.last_approx_vram_used_kcpp_mb = None
        self.last_successful_monitored_run_details_gui = None
        self.user_requested_stop_monitoring = False

        self._return_to_full_tuning_menu() # Reset tuning view buttons/layout

        if switch_to_model_selection:
            self._show_model_selection_view()

    def launch_and_monitor_for_tuning(self):
        if not self.tuning_in_progress:
            messagebox.showerror("Error", "Tuning session not active.", parent=self)
            return
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
            messagebox.showwarning("Process Running", "A KoboldCpp process is already being monitored. Please wait or stop it.", parent=self)
            return

        self.log_to_console(f"Tuning: Launch & Monitor OT Level {self.current_tuning_attempt_level}")
        self.user_requested_stop_monitoring = False  # Reset for new monitor

        if hasattr(self, 'kcpp_live_output_text') and self.kcpp_live_output_text.winfo_exists():
            self.kcpp_live_output_text.configure(state="normal")
            self.kcpp_live_output_text.delete("1.0", "end")  # Clear old output
            self.kcpp_console_line_count = 0
            self.kcpp_live_output_text.insert("1.0", f"Preparing to launch KoboldCpp with OT Level {self.current_tuning_attempt_level}...\n")
            self.kcpp_console_line_count += 1
            self.kcpp_live_output_text.configure(state="disabled")
        
        self._set_tuning_buttons_state("disabled", monitoring_active=True)
        if hasattr(self, 'post_monitor_choices_frame') and self.post_monitor_choices_frame.winfo_exists():
            self.post_monitor_choices_frame.grid_remove()  # Hide choices frame during monitoring

        self.kcpp_success_event.clear()
        self.kcpp_oom_event.clear()
        self.kcpp_output_lines_shared.clear()
        self.last_free_vram_after_load_mb = None  # Reset for this run
        self.last_approx_vram_used_kcpp_mb = None
        self.level_of_last_monitored_run = self.current_tuning_attempt_level

        ot_string = tensortune_core.generate_overridetensors(self.current_tuning_model_analysis, self.current_tuning_attempt_level)
        
        # Key fix: Properly process manual GPU layers
        manual_gpu_override_for_command = None
        effective_gpu_setting = self.effective_gpu_layers_for_command.get()
        
        if effective_gpu_setting and effective_gpu_setting.isdigit():
            manual_gpu_override_for_command = int(effective_gpu_setting)

        args_list = tensortune_core.build_command(
            self.current_tuning_model_path, 
            ot_string, 
            self.current_tuning_model_analysis,
            self.current_tuning_session_base_args, 
            current_attempt_level_for_tuning=self.current_tuning_attempt_level,
            manual_gpu_layers_override=manual_gpu_override_for_command 
        )
        
        self.current_command_list_for_db = tensortune_core.get_command_to_run(self.koboldcpp_executable, args_list)
        
        _, _, _, gpu_info_before_launch = tensortune_core.get_available_vram_mb(
            self.config,
            target_gpu_type=self.config.get("gpu_selection_mode", "auto") if self.config.get("gpu_selection_mode", "auto") != "auto" else None,
            target_gpu_index=self.config.get("selected_gpu_index", 0)
        )
        self.vram_at_decision_for_db = gpu_info_before_launch.get("free_mb") 

        self.kcpp_process_obj, launch_error_msg = tensortune_core.launch_process(
            self.current_command_list_for_db, capture_output=True, new_console=False, use_text_mode=False 
        )

        if launch_error_msg or not self.kcpp_process_obj:
            self.log_to_console(f"Failed to launch KoboldCpp for monitoring: {launch_error_msg or 'Unknown error'}")
            self._log_to_kcpp_live_output(f"LAUNCH ERROR: {launch_error_msg or 'Unknown error'}\n")
            tensortune_core.save_config_to_db( 
                self.db_path, self.current_tuning_model_path, self.current_tuning_model_analysis,
                self.vram_at_decision_for_db, self.current_command_list_for_db,
                self.level_of_last_monitored_run, "LAUNCH_FAILED_SETUP_GUI", None
            )
            self._set_tuning_buttons_state("normal", monitoring_active=False) 
            return

        self._log_to_kcpp_live_output(f"KoboldCpp process started (PID: {self.kcpp_process_obj.pid}). Monitoring output...\n")

        effective_args_for_port_check = {**self.config.get("default_args", {}), **self.current_tuning_session_base_args}
        target_port_str_for_success = effective_args_for_port_check.get("--port", "5000") 
        success_pattern_regex = self.config.get("kobold_success_pattern", tensortune_core.DEFAULT_CONFIG_TEMPLATE["kobold_success_pattern"])
        oom_keywords_list = self.config.get("oom_error_keywords", tensortune_core.DEFAULT_CONFIG_TEMPLATE["oom_error_keywords"])

        self.kcpp_monitor_thread = threading.Thread(
            target=self._monitor_kcpp_output_thread_target,
            args=(self.kcpp_process_obj, success_pattern_regex, oom_keywords_list, target_port_str_for_success),
            daemon=True
        )
        self.kcpp_monitor_thread.start()
        self.monitor_start_time = time.monotonic()
        self._poll_monitor_status() 

    def _log_to_kcpp_live_output(self, text_line: str):
        def _update():
            if hasattr(self, 'kcpp_live_output_text') and self.kcpp_live_output_text.winfo_exists():
                self.kcpp_live_output_text.configure(state="normal")
                if self.kcpp_console_line_count >= self.MAX_KCPP_CONSOLE_LINES:
                    num_lines_to_delete = self.kcpp_console_line_count - self.MAX_KCPP_CONSOLE_LINES + text_line.count('\n') + 1
                    self.kcpp_live_output_text.delete("1.0", f"{num_lines_to_delete}.0")
                    self.kcpp_console_line_count -= (num_lines_to_delete -1) 

                self.kcpp_live_output_text.insert("end", text_line)
                self.kcpp_console_line_count += text_line.count('\n') 
                self.kcpp_live_output_text.see("end") 
                self.kcpp_live_output_text.configure(state="disabled")
        
        if hasattr(self, 'after'): 
            self.after(0, _update) 

    def _monitor_kcpp_output_thread_target(self, process, success_regex, oom_keywords, target_port):
        try:
            for line_bytes in iter(process.stdout.readline, b''): 
                if not line_bytes or self.user_requested_stop_monitoring: 
                    break
                
                try: 
                    line_decoded = line_bytes.decode('utf-8', errors='replace')
                except UnicodeDecodeError:
                    line_decoded = line_bytes.decode('latin-1', errors='replace') 

                self._log_to_kcpp_live_output(line_decoded) 
                line_strip_lower = line_decoded.strip().lower()

                if line_strip_lower: 
                    self.kcpp_output_lines_shared.append(line_decoded.strip()) 
                    
                    if not self.kcpp_success_event.is_set() and not self.kcpp_oom_event.is_set():
                        success_match = re.search(success_regex, line_decoded.strip(), re.IGNORECASE)
                        if success_match:
                            launched_port_from_log = target_port 
                            try: 
                                launched_port_from_log = success_match.group(1)
                            except IndexError:
                                pass 
                            
                            if str(launched_port_from_log) == str(target_port):
                                self.kcpp_success_event.set()
                        
                        if not self.kcpp_success_event.is_set(): 
                            for oom_keyword in oom_keywords:
                                if oom_keyword.lower() in line_strip_lower:
                                    self.kcpp_oom_event.set()
                                    break 
                
                if self.kcpp_success_event.is_set() or self.kcpp_oom_event.is_set():
                    break 
        except Exception as e_monitor:
            self._log_to_kcpp_live_output(f"\nError in KCPP output monitor thread: {type(e_monitor).__name__}: {e_monitor}\n")
            traceback.print_exc() 
        finally:
            if process.stdout and not process.stdout.closed:
                try: process.stdout.close()
                except: pass # noqa, Best effort to close
            if self.user_requested_stop_monitoring:
                 self.log_to_console("KCPP output monitor thread exiting due to user stop request.")


    def _poll_monitor_status(self):
        loading_timeout_seconds = float(self.config.get("loading_timeout_seconds", 60))
        elapsed_time = time.monotonic() - self.monitor_start_time
        process_has_exited = False

        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is not None: # Process terminated
            if not self.kcpp_success_event.is_set() and not self.kcpp_oom_event.is_set(): # And no clear signal
                process_has_exited = True

        if self.user_requested_stop_monitoring:
            self._handle_monitoring_completion("USER_STOPPED_MONITORING_GUI")
        elif self.kcpp_success_event.is_set():
            self._handle_monitoring_completion("SUCCESS_LOAD_DETECTED_GUI")
        elif self.kcpp_oom_event.is_set():
            self._handle_monitoring_completion("OOM_CRASH_DETECTED_GUI")
        elif process_has_exited:
            self._handle_monitoring_completion("PREMATURE_EXIT_GUI")
        elif elapsed_time > loading_timeout_seconds:
            self._handle_monitoring_completion("TIMEOUT_NO_SIGNAL_GUI")
        else: # Continue polling
            self.after(250, self._poll_monitor_status)

    def _set_tuning_buttons_state(self, state="normal", monitoring_active=False):
            # Primary launch buttons
            for btn_attr in ['btn_tune_launch_monitor', 'btn_tune_skip_launch_direct']:
                btn = getattr(self, btn_attr, None)
                if btn and hasattr(btn, 'winfo_exists') and btn.winfo_exists():
                    btn.configure(state="disabled" if monitoring_active else state)

            # Logic to disable monitor button if no GPU for monitoring
            if hasattr(self, 'btn_tune_launch_monitor') and self.btn_tune_launch_monitor.winfo_exists():
                if not monitoring_active and state == "normal": # Only modify if it would otherwise be normal
                    no_gpu_for_monitoring = not (self.gpu_info and self.gpu_info.get("success") and self.gpu_info.get("total_mb_budgeted", 0.0) > 0)
                    if no_gpu_for_monitoring:
                        self.btn_tune_launch_monitor.configure(state="disabled")
                        ToolTip(self.btn_tune_launch_monitor, "Monitoring is less useful without a detected GPU for VRAM measurement.\nLaunch directly or fix GPU detection in Settings.")
                    else: # Ensure original tooltip is restored if re-enabled
                        ToolTip(self.btn_tune_launch_monitor, "Launch KoboldCpp with the current OT strategy and monitor its output for success or errors (e.g., OOM).")


            # Stop monitoring button visibility and state
            if hasattr(self, 'tuning_stop_monitor_frame') and self.tuning_stop_monitor_frame.winfo_exists():
                if monitoring_active:
                    self.tuning_stop_monitor_frame.grid() # Show frame
                    if hasattr(self, 'btn_stop_monitoring') and self.btn_stop_monitoring.winfo_exists():
                        self.btn_stop_monitoring.pack(fill="x", padx=5, pady=5) # Ensure packed
                        self.btn_stop_monitoring.configure(state="normal")
                else: # Not monitoring
                    if hasattr(self, 'btn_stop_monitoring') and self.btn_stop_monitoring.winfo_exists():
                        self.btn_stop_monitoring.pack_forget() # Hide button
                    self.tuning_stop_monitor_frame.grid_remove() # Hide frame

            # Secondary and navigation buttons
            secondary_nav_buttons = [
                'btn_tune_more_gpu', 'btn_tune_more_cpu', 'btn_tune_edit_args',
                'btn_tune_edit_model_perm_args', 'btn_tune_new_gguf',
                'btn_tune_history', 'btn_tune_quit_tuning'
            ]
            for btn_attr in secondary_nav_buttons:
                btn = getattr(self, btn_attr, None)
                if btn and hasattr(btn, 'winfo_exists') and btn.winfo_exists():
                    btn.configure(state="disabled" if monitoring_active else state)

            # Fine-tune More GPU/CPU buttons based on current level if not monitoring
            if not monitoring_active and state == "normal" and self.tuning_in_progress:
                if hasattr(self, 'btn_tune_more_gpu') and self.btn_tune_more_gpu.winfo_exists():
                    self.btn_tune_more_gpu.configure(state="normal" if self.current_tuning_attempt_level > self.current_tuning_min_level else "disabled")
                if hasattr(self, 'btn_tune_more_cpu') and self.btn_tune_more_cpu.winfo_exists():
                    self.btn_tune_more_cpu.configure(state="normal" if self.current_tuning_attempt_level < self.current_tuning_max_level else "disabled")
                if hasattr(self, 'btn_tune_edit_model_perm_args') and self.btn_tune_edit_model_perm_args.winfo_exists():
                     self.btn_tune_edit_model_perm_args.configure(state="normal" if self.current_tuning_model_path else "disabled")


    def _stop_current_monitoring_action(self):
        if self.tuning_in_progress and self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
            self.log_to_console("User requested to stop current KCPP monitoring.")
            self.user_requested_stop_monitoring = True # Signal the polling loop/monitor thread
        else:
            self.log_to_console("No active KCPP monitoring process to stop.")

          
    # Fix 1: Update the _handle_post_monitor_action function to safely handle focus
    def _handle_post_monitor_action(self, action_key: str, original_outcome: str):
        try:
            self.log_to_console(f"User selected post-monitoring action: '{action_key}' for outcome '{original_outcome}'")
            should_stop_monitored_kcpp = True # Default to stopping
            self.user_requested_stop_monitoring = False # Reset this flag

            # Decide if the monitored KCPP instance should be kept running
            if action_key == "launch_for_use" or action_key == "launch_for_use_risky":
                if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
                    should_stop_monitored_kcpp = False # Keep it running for these actions
                else: # KCPP already stopped or was never started properly
                    self.log_to_console("KCPP instance for 'launch_for_use' is not running. Will attempt a new launch.")
                    should_stop_monitored_kcpp = True # Ensure it's considered stopped
            
            if action_key == "set_as_preferred":
                 should_stop_monitored_kcpp = True

            if should_stop_monitored_kcpp and self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
                self.log_to_console(f"Stopping monitored KCPP instance (PID: {self.kcpp_process_obj.pid}) based on user action '{action_key}'.")
                tensortune_core.kill_process(self.kcpp_process_obj.pid, force=True)
            
            if should_stop_monitored_kcpp: 
                self.kcpp_process_obj = None 

            command_that_led_to_outcome = self.current_command_list_for_db 
            db_outcome_suffix = "_GUI" 
            if action_key == "launch_for_use": db_outcome_suffix = "_USER_ACCEPTED_TUNED_GUI"
            elif action_key == "launch_for_use_risky": db_outcome_suffix = "_USER_LAUNCHED_RISKY_GUI"
            elif action_key == "save_good_more_gpu": db_outcome_suffix = "_USER_SAVED_GOOD_GPU_GUI"
            elif action_key == "more_gpu_now": db_outcome_suffix = "_USER_WANTS_MORE_GPU_GUI"
            elif action_key == "auto_adjust_cpu": db_outcome_suffix = "_USER_AUTO_ADJUST_CPU_GUI"
            elif action_key == "more_cpu_after_fail": db_outcome_suffix = "_USER_TRIED_CPU_FAIL_GUI"
            elif action_key == "more_cpu_now": db_outcome_suffix = "_USER_WANTS_MORE_CPU_GUI"
            elif action_key == "return_to_tuning_menu": db_outcome_suffix = "_USER_RETURNED_MENU_GUI"
            elif action_key == "set_as_preferred": db_outcome_suffix = "_USER_MARKED_AS_BEST_GUI" # New outcome for "Set as Preferred"
            
            final_db_outcome_for_this_run = original_outcome + db_outcome_suffix
            tensortune_core.save_config_to_db(
                self.db_path, self.current_tuning_model_path, self.current_tuning_model_analysis,
                self.vram_at_decision_for_db, command_that_led_to_outcome,
                self.level_of_last_monitored_run, final_db_outcome_for_this_run,
                self.last_approx_vram_used_kcpp_mb
            )
            self.load_history() # Refresh history tab

            # Perform the chosen action
            if action_key == "launch_for_use":
                if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None: # If kept running
                    self.log_to_console("Transferring control of monitored KCPP instance for use.")
                    self.last_process = self.kcpp_process_obj # This is now the main launched process
                    self.process_running = True
                    self.kcpp_process_obj = None # Clear from monitored slot
                    
                    if self.config.get("auto_open_webui", True):
                        effective_args_for_webui = {**self.config.get("default_args", {}), **self.current_tuning_session_base_args}
                        port_to_open = effective_args_for_webui.get("--port", "5000")
                        webbrowser.open(f"http://localhost:{port_to_open}")
                    
                    self.end_tuning_session(switch_to_model_selection=True)
                    return
                else: # Monitored process was stopped or failed, so relaunch
                    self.log_to_console("Monitored KCPP not running, re-launching for use with accepted config.")
                    self._launch_final_koboldcpp(command_that_led_to_outcome, final_db_outcome_for_this_run, self.level_of_last_monitored_run)
                    self.end_tuning_session(switch_to_model_selection=True)
                    return

            elif action_key == "launch_for_use_risky":
                self.log_to_console("Attempting to launch risky configuration for use...")
                self._launch_final_koboldcpp(command_that_led_to_outcome, final_db_outcome_for_this_run, self.level_of_last_monitored_run)
                self.end_tuning_session(switch_to_model_selection=True)
                return
            
            # Adjust OT level for continued tuning
            if action_key in ["save_good_more_gpu", "more_gpu_now"]: # More GPU
                if self.current_tuning_attempt_level > self.current_tuning_min_level:
                    self.current_tuning_attempt_level -= 1
                else: self.log_to_console("Already at Max GPU (Min Level). Cannot decrease further.")
            elif action_key in ["auto_adjust_cpu", "more_cpu_after_fail", "more_cpu_now"]: # More CPU
                if self.current_tuning_attempt_level < self.current_tuning_max_level:
                    self.current_tuning_attempt_level += 1
                else: self.log_to_console("Already at Max CPU (Max Level). Cannot increase further.")

            # For "set_as_preferred", "return_to_tuning_menu", or any other action that continues tuning:
            self._return_to_full_tuning_menu() # This also calls update_tuning_display
        except Exception as e:
            self.log_to_console(f"Error in post-monitor action: {str(e)}")
            import traceback
            self.log_to_console(traceback.format_exc())
            # Attempt to recover by returning to tuning menu
            if self.winfo_exists():
                self._return_to_full_tuning_menu()

    def _launch_final_koboldcpp(self, command_list_to_run: list, db_outcome_reason: str, attempt_level_for_db: int):
        self.check_koboldcpp_executable() # Ensure path is current and valid
        if not self.koboldcpp_executable or not (os.path.exists(self.koboldcpp_executable) or shutil.which(self.koboldcpp_executable)):
            messagebox.showerror("KoboldCpp Not Found", f"The KoboldCpp executable path is not valid or not found:\n{self.koboldcpp_executable}", parent=self)
            self.log_to_console(f"Launch Aborted: KCPP executable '{self.koboldcpp_executable}' not found.")
            return

        self.log_to_console(f"Preparing to launch KoboldCpp for use. DB Outcome Reason: {db_outcome_reason}")

        # Stop any previously launched KCPP (from a direct launch, not monitored one)
        if self.last_process and self.last_process.poll() is None:
            self.log_to_console(f"Stopping previously launched KCPP (PID: {self.last_process.pid})...")
            tensortune_core.kill_process(self.last_process.pid, force=True)
            self.last_process = None
        self.process_running = False # Reset flag

        # Get VRAM at this specific launch decision point for DB
        _, _, _, gpu_info_at_final_launch = tensortune_core.get_available_vram_mb(
            self.config,
            target_gpu_type=self.config.get("gpu_selection_mode", "auto") if self.config.get("gpu_selection_mode", "auto") != "auto" else None,
            target_gpu_index=self.config.get("selected_gpu_index", 0)
        )
        vram_at_this_launch_decision = gpu_info_at_final_launch.get("free_mb") # Actual HW free VRAM

        model_path_for_log = self.current_tuning_model_path if self.tuning_in_progress else self.current_model_path
        model_analysis_for_log = self.current_tuning_model_analysis if self.tuning_in_progress else self.model_analysis_info
        if not model_path_for_log or not model_analysis_for_log:
             self.log_to_console("Warning: Model path or analysis missing for DB logging during final launch.")
             # Fallback if tuning info is somehow cleared but model path is still there
             if self.current_model_path: model_path_for_log = self.current_model_path
             if self.model_analysis_info: model_analysis_for_log = self.model_analysis_info


        cmd_final = command_list_to_run[:] # Make a copy to modify
        if cmd_final:
            # Ensure the executable path in the command is the current one from config
            if cmd_final[0].lower() == sys.executable.lower() and len(cmd_final) > 1 and cmd_final[1].lower().endswith(".py"):
                # Python script launch, update script path if different
                if self.koboldcpp_executable.lower().endswith(".py"):
                     cmd_final[1] = self.koboldcpp_executable
            else: # Direct executable launch
                cmd_final[0] = self.koboldcpp_executable
        else:
            self.log_to_console("Error: Empty command list for final launch.")
            messagebox.showerror("Launch Error", "Cannot launch: Command list is empty.", parent=self)
            return

        # Save to DB before launching
        tensortune_core.save_config_to_db(
            self.db_path, model_path_for_log, model_analysis_for_log,
            vram_at_this_launch_decision, cmd_final,
            attempt_level_for_db, db_outcome_reason,
            self.last_approx_vram_used_kcpp_mb # Carry over from monitoring if available
        )
        self.load_history() # Refresh history tab

        # Launch KCPP in a new console, not capturing output
        launched_kcpp_process, launch_err_msg = tensortune_core.launch_process(
            cmd_final, capture_output=False, new_console=True
        )

        if launch_err_msg or not launched_kcpp_process:
            self.log_to_console(f"Failed to launch KoboldCpp: {launch_err_msg or 'Unknown error'}")
            messagebox.showerror("Launch Error", f"Failed to launch KoboldCpp:\n{launch_err_msg or 'Unknown error'}", parent=self)
            # Update DB record to indicate launch failure if it was a success type initially
            tensortune_core.save_config_to_db(
                self.db_path, model_path_for_log, model_analysis_for_log,
                vram_at_this_launch_decision, cmd_final,
                attempt_level_for_db, "LAUNCH_FOR_USE_FAILED_GUI", # Specific outcome
                self.last_approx_vram_used_kcpp_mb
            )
            self.load_history()
        else:
            self.log_to_console(f"KoboldCpp launched successfully for use (PID: {launched_kcpp_process.pid}). New console window opened.")
            self.last_process = launched_kcpp_process
            self.process_running = True

            if self.config.get("auto_open_webui", True):
                # Determine port from effective arguments
                args_dict_from_cmd = tensortune_core.args_list_to_dict(
                    cmd_final[2:] if cmd_final[0].lower() == sys.executable.lower() and len(cmd_final) > 1 and cmd_final[1].lower().endswith(".py") else cmd_final[1:]
                )
                port_to_open_webui = "5000" # Default
                base_args_for_launch = self._get_merged_args_for_model(model_path_for_log)
                effective_launch_args = base_args_for_launch.copy()
                effective_launch_args.update(args_dict_from_cmd) # Command line args take precedence

                if "--port" in effective_launch_args:
                    port_to_open_webui = effective_launch_args["--port"]
                
                try:
                    port_num = int(port_to_open_webui)
                    if 1 <= port_num <= 65535:
                        url_to_open = f"http://localhost:{port_num}"
                        self.log_to_console(f"Auto-opening Web UI at {url_to_open} in a few seconds...")
                        threading.Timer(3.0, lambda: webbrowser.open(url_to_open)).start() # Delay slightly
                    else:
                        self.log_to_console(f"Invalid port number '{port_to_open_webui}' for auto-opening Web UI.")
                except ValueError:
                     self.log_to_console(f"Invalid port value '{port_to_open_webui}' configured; cannot auto-open Web UI.")
            else:
                self.log_to_console("Auto-open Web UI is disabled in settings.")

    def skip_tune_and_launch_direct(self):
        if not self.tuning_in_progress:
            messagebox.showwarning("Not Tuning", "Tuning session not active. Cannot skip.", parent=self)
            return
        self.log_to_console("User chose to skip further tuning and launch current configuration directly.")

        _, _, _, gpu_info_at_direct_launch = tensortune_core.get_available_vram_mb(
            self.config,
            target_gpu_type=self.config.get("gpu_selection_mode", "auto") if self.config.get("gpu_selection_mode", "auto") != "auto" else None,
            target_gpu_index=self.config.get("selected_gpu_index", 0)
        )
        if self.vram_at_decision_for_db is None:  # If not set by a prior monitoring
            self.vram_at_decision_for_db = gpu_info_at_direct_launch.get("free_mb")

        ot_string_for_launch = tensortune_core.generate_overridetensors(self.current_tuning_model_analysis, self.current_tuning_attempt_level)
        
        # Key fix: Properly handle manual GPU layers
        manual_gpu_override_for_command = None
        effective_gpu_setting = self.effective_gpu_layers_for_command.get()
        
        if effective_gpu_setting and effective_gpu_setting.isdigit():
            manual_gpu_override_for_command = int(effective_gpu_setting)

        args_list_for_launch = tensortune_core.build_command(
            self.current_tuning_model_path, 
            ot_string_for_launch,
            self.current_tuning_model_analysis, 
            self.current_tuning_session_base_args,
            current_attempt_level_for_tuning=self.current_tuning_attempt_level,
            manual_gpu_layers_override=manual_gpu_override_for_command
        )
        
        command_to_run_final = tensortune_core.get_command_to_run(self.koboldcpp_executable, args_list_for_launch)
        
        self._launch_final_koboldcpp(command_to_run_final, "SUCCESS_USER_DIRECT_LAUNCH_GUI", self.current_tuning_attempt_level)
        self.end_tuning_session(switch_to_model_selection=True)  # End session and go back

    def save_settings_action(self):
        if self.save_config(): # save_config handles logging and dirty flag
            messagebox.showinfo("Settings Saved", "Global launcher settings have been saved!", parent=self)
            # No explicit error message here as save_config shows one if needed

    def reset_config_action(self):
        confirm_reset = messagebox.askyesno(
            "Reset Configuration",
            "Are you sure you want to reset ALL launcher settings to their original defaults?\n"
            "This will overwrite your current configuration file.\n\n"
            "A backup of your current settings will be attempted.",
            icon=messagebox.WARNING,
            parent=self
        )
        if confirm_reset:
            try:
                core_config_filepath = tensortune_core.CONFIG_FILE
                if os.path.exists(core_config_filepath):
                    backup_filepath = core_config_filepath + f".backup_reset_{time.strftime('%Y%m%d-%H%M%S')}.json"
                    shutil.copy2(core_config_filepath, backup_filepath)
                    self.log_to_console(f"Current configuration backed up to: {backup_filepath}")
                
                if os.path.exists(core_config_filepath): # Delete current config to force re-creation
                    os.remove(core_config_filepath)

                core_init_results_after_reset = tensortune_core.initialize_launcher() # This creates new default
                self.config = core_init_results_after_reset["config"]
                self.system_info = core_init_results_after_reset["system_info"]
                self.gpu_info = core_init_results_after_reset["gpu_info"]
                self.koboldcpp_capabilities = core_init_results_after_reset.get("koboldcpp_capabilities", {})
                
                self.koboldcpp_executable = self.config.get("koboldcpp_executable")
                self.default_model_dir = self.config.get("default_gguf_dir", "")
                self.db_path = self.config["db_file"]

                self.load_settings_from_config() # Populate UI with new defaults
                self.check_koboldcpp_executable()
                if hasattr(self, 'populate_model_specifics_list_display'):
                     self.populate_model_specifics_list_display() # Should be empty
                if hasattr(self, 'update_kcpp_capabilities_display'):
                     self.update_kcpp_capabilities_display(re_detect=True)
                
                self.after(100, self._populate_gpu_id_dropdown_on_startup) # Refresh GPU list
                self.log_to_console(f"Configuration reset to defaults. DB path: {self.db_path}")
                messagebox.showinfo("Configuration Reset", "Launcher settings have been reset to defaults.\nYou might need to go through the first-time setup again if paths are incorrect.", parent=self)
                self.after(100, self._run_first_time_setup_if_needed) # Trigger FTS if needed

            except Exception as e_reset:
                error_msg_reset = f"An error occurred while resetting configuration: {e_reset}\n{traceback.format_exc()}"
                self.log_to_console(error_msg_reset)
                messagebox.showerror("Reset Error", error_msg_reset, parent=self)

    def launch_direct_defaults(self):
        if not self.current_model_path:
            messagebox.showwarning("No Model Selected", "Please select a GGUF model file first.", parent=self)
            return
        
        self.check_koboldcpp_executable() # Ensure exe path is valid
        if not self.koboldcpp_executable or not (os.path.exists(self.koboldcpp_executable) or shutil.which(self.koboldcpp_executable)):
            messagebox.showerror("KoboldCpp Not Found", f"The KoboldCpp executable path is not valid or not found:\n{self.koboldcpp_executable}", parent=self)
            return
        
        self.log_to_console(f"Attempting direct launch for: {os.path.basename(self.current_model_path)} using default settings.")
        
        effective_args_for_launch = self._get_merged_args_for_model(self.current_model_path)
        current_model_analysis = self.model_analysis_info
        
        # Ensure model is analyzed
        if not current_model_analysis or current_model_analysis.get('filepath') != self.current_model_path:
            self.log_to_console("Re-analyzing model for direct launch...")
            self.analyze_model_action(self.current_model_path)
            current_model_analysis = self.model_analysis_info
        
        if not current_model_analysis or 'filepath' not in current_model_analysis: # Check again
            messagebox.showerror("Analysis Error", "Failed to analyze model. Cannot proceed with direct launch.", parent=self)
            self.log_to_console("Direct launch aborted: Model analysis failed.")
            return
            
        # Build command without specific OT string (uses defaults from args)
        args_list_for_kcpp = tensortune_core.build_command(
            self.current_model_path, None, current_model_analysis, effective_args_for_launch
        )
        command_list_final = tensortune_core.get_command_to_run(self.koboldcpp_executable, args_list_for_kcpp)

        _, _, _, gpu_info_at_direct_launch = tensortune_core.get_available_vram_mb(
            self.config,
            target_gpu_type=self.config.get("gpu_selection_mode", "auto") if self.config.get("gpu_selection_mode", "auto") != "auto" else None,
            target_gpu_index=self.config.get("selected_gpu_index", 0)
        )
        self.vram_at_decision_for_db = gpu_info_at_direct_launch.get("free_mb") # Actual HW free
        self.last_approx_vram_used_kcpp_mb = None # Not monitored

        self._launch_final_koboldcpp(command_list_final, "SUCCESS_USER_DIRECT_SETTINGS_GUI", 0) # 0 for N/A OT level

    def launch_best_remembered(self):
        if not self.current_model_path:
            messagebox.showwarning("No Model Selected", "Please select a GGUF model file first.", parent=self)
            return
        
        current_model_analysis = self.model_analysis_info
        if not current_model_analysis or current_model_analysis.get('filepath') != self.current_model_path:
            self.log_to_console("Re-analyzing model for launching best remembered...")
            self.analyze_model_action(self.current_model_path)
            current_model_analysis = self.model_analysis_info
        
        if not current_model_analysis or 'filepath' not in current_model_analysis:
            messagebox.showerror("Analysis Error", "Failed to analyze model. Cannot find best remembered config.", parent=self)
            self.log_to_console("Launch best remembered aborted: Model analysis failed.")
            return

        self.check_koboldcpp_executable()
        if not self.koboldcpp_executable or not (os.path.exists(self.koboldcpp_executable) or shutil.which(self.koboldcpp_executable)):
            messagebox.showerror("KoboldCpp Not Found", f"The KoboldCpp executable path is not valid or not found:\n{self.koboldcpp_executable}", parent=self)
            return

        self.log_to_console("Attempting to launch using the best remembered configuration...")
        _, _, _, current_gpu_full_info = tensortune_core.get_available_vram_mb(
            self.config,
            target_gpu_type=self.config.get("gpu_selection_mode", "auto") if self.config.get("gpu_selection_mode", "auto") != "auto" else None,
            target_gpu_index=self.config.get("selected_gpu_index", 0)
        )
        current_actual_hw_vram_mb = current_gpu_full_info.get("free_mb", 0.0) # Actual HW free VRAM

        best_historical_config_found = tensortune_core.find_best_historical_config(
            self.db_path, current_model_analysis, current_actual_hw_vram_mb, self.config
        )

        if best_historical_config_found and best_historical_config_found.get("args_list"):
            self.log_to_console(f"Found best remembered config - Level: {best_historical_config_found['attempt_level']}, Outcome: {best_historical_config_found['outcome']}")
            remembered_args_list = best_historical_config_found["args_list"]
            remembered_args_dict = tensortune_core.args_list_to_dict(remembered_args_list)
            
            historical_ot_string = remembered_args_dict.pop("--overridetensors", None) # Get OT string if present
            
            # Base args for this launch: current global + model-specific defaults
            base_args_for_this_launch = self._get_merged_args_for_model(self.current_model_path)
            final_effective_args_dict = base_args_for_this_launch.copy()
            remembered_args_dict.pop("--model", None) # Don't let historical override model
            final_effective_args_dict.update(remembered_args_dict) # Overlay historical args

            historical_attempt_level = best_historical_config_found.get("attempt_level", 0)

            final_command_args_list = tensortune_core.build_command(
                self.current_model_path, historical_ot_string, current_model_analysis,
                final_effective_args_dict, current_attempt_level_for_tuning=historical_attempt_level
            )
            command_list_to_execute = tensortune_core.get_command_to_run(self.koboldcpp_executable, final_command_args_list)
            
            self.vram_at_decision_for_db = current_actual_hw_vram_mb
            self.last_approx_vram_used_kcpp_mb = best_historical_config_found.get("approx_vram_used_kcpp_mb")

            self._launch_final_koboldcpp(command_list_to_execute, "SUCCESS_USER_LAUNCHED_BEST_REMEMBERED_GUI", historical_attempt_level)
        else:
            self.log_to_console("No suitable remembered configuration found in history. Falling back to direct launch with default settings.")
            messagebox.showinfo("Not Found", "No suitable remembered configuration was found in the launch history for this model and current VRAM.\n\nLaunching with default settings instead.", parent=self)
            self.launch_direct_defaults() # Fallback

    def change_theme(self, theme_name: str):
        ctk.set_appearance_mode(theme_name.lower())
        self.mark_settings_dirty() # Theme change is a setting
        self.log_to_console(f"UI Theme changed to: {theme_name}.")

    def log_to_console(self, text_message: str, level: str = "INFO"): # ADDED 'level' parameter
        def _perform_log():
            if hasattr(self, 'console') and self.console.winfo_exists():
                self.console.configure(state="normal")
                timestamp = time.strftime("%H:%M:%S") # Make sure 'time' module is imported
                
                level_upper = level.upper()
                # For simplicity, just prepending the level string.
                # For actual colors in CTkTextbox, you'd use its tag system which is more involved.
                prefix = f"[{timestamp}] [{level_upper}] "
                
                self.console.insert("end", f"{prefix}{text_message}\n")
                self.console.see("end")
                self.console.configure(state="disabled")
        
        if hasattr(self, 'after') and self.winfo_exists(): # Check if GUI window still exists
            try:
                self.after(0, _perform_log)
            except RuntimeError: # Can happen if window is being destroyed
                print(f"LOG [{level.upper()}]: {text_message}") # Fallback
        else:
            # This case handles if self.after is not available (e.g., called very early or from non-GUI thread)
            # or if self.winfo_exists() is false (window destroyed)
            print(f"LOG [{level.upper()}]: {text_message}")

    def debug_show_detailed_layer_mapping(self):
        """
        Display a comprehensive mapping between OT levels and GPU layers.
        This helps visualize how the core function is behaving with your specific model.
        """
        if not self.tuning_in_progress or not self.current_tuning_model_analysis:
            self.log_to_console("No active tuning session - cannot display mapping")
            return
        
        model_layers = self.current_tuning_model_analysis.get('num_layers', 32)
        is_moe = self.current_tuning_model_analysis.get('is_moe', False)
        model_type = "MoE" if is_moe else "Dense"
        
        self.log_to_console(f"GPU Layer mapping for current model ({model_layers} total layers, {model_type}):")
        
        # Show the full range of possible values
        level_range = range(self.current_tuning_min_level, self.current_tuning_max_level + 1)
        
        # Record values to look for patterns
        values_list = []
        last_value = None
        
        for level in level_range:
            layers = tensortune_core.get_gpu_layers_for_level(
                self.current_tuning_model_analysis, level
            )
            
            # Track unique values and changes
            if layers != last_value:
                values_list.append((level, layers))
                last_value = layers
        
        # Display only the level transitions for clarity
        self.log_to_console(f"Found {len(values_list)} unique layer values across {len(list(level_range))} levels:")
        
        # Log all the transition points
        for i, (level, layers) in enumerate(values_list):
            percent = (layers / model_layers) * 100 if model_layers > 0 else 0
            if i < len(values_list) - 1:
                next_level = values_list[i+1][0]
                range_txt = f"{level} to {next_level-1}"
            else:
                range_txt = f"{level}+"
                
            self.log_to_console(f"  Level {range_txt}: {layers} layers ({percent:.1f}% of model)")
        
        # Summary information
        min_layers = min([layers for _, layers in values_list]) if values_list else 0
        max_layers = max([layers for _, layers in values_list]) if values_list else 0
        self.log_to_console(f"Layer range: {min_layers} to {max_layers} (model has {model_layers} total layers)")
        
        # Provide advice if there are issues
        if len(values_list) < 5:
            self.log_to_console("WARNING: Very few unique layer values detected. This could indicate an issue with layer calculation.")
        if max_layers < model_layers * 0.9 and values_list:
            self.log_to_console(f"WARNING: Maximum layer count ({max_layers}) is significantly less than total layers ({model_layers}).")
    
    def browse_model(self):
        initial_dir_for_dialog = self.config.get("last_used_gguf_dir") or \
                                 self.config.get("default_gguf_dir") or \
                                 os.getcwd()
        if not os.path.isdir(initial_dir_for_dialog): # Ensure it's a valid dir
            initial_dir_for_dialog = self.config.get("default_gguf_dir") or os.getcwd()
        if not os.path.isdir(initial_dir_for_dialog):
             initial_dir_for_dialog = os.getcwd() # Ultimate fallback


        filepath_selected = filedialog.askopenfilename(
            title="Select GGUF Model File",
            filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")],
            initialdir=initial_dir_for_dialog,
            parent=self
        )
        if filepath_selected:
            if hasattr(self, 'model_path_entry') and self.model_path_entry.winfo_exists():
                self.model_path_entry.delete(0, "end")
                self.model_path_entry.insert(0, filepath_selected)
            
            self.current_model_path = os.path.abspath(filepath_selected)
            new_last_dir = os.path.dirname(self.current_model_path)

            if self.config.get("last_used_gguf_dir") != new_last_dir:
                self.config["last_used_gguf_dir"] = new_last_dir
                self.save_config() 
            
            self.analyze_model_action(self.current_model_path)

    def analyze_model_action(self, model_filepath_to_analyze: str):
        self.log_to_console(f"Analyzing model: {os.path.basename(model_filepath_to_analyze)}")
        self.model_analysis_info = tensortune_core.analyze_filename(model_filepath_to_analyze)
        
        moe_str = 'MoE' if self.model_analysis_info.get('is_moe') else 'Dense'
        size_b_str = self.model_analysis_info.get('size_b', "N/A")
        quant_str = self.model_analysis_info.get('quant', "N/A")
        num_layers_str = self.model_analysis_info.get('num_layers', "N/A")
        est_vram_str = self.model_analysis_info.get('estimated_vram_gb_full_gpu', "N/A")
        
        info_text = (
            f"Type: {moe_str}, Size: ~{size_b_str}B, Quant: {quant_str}, "
            f"Layers: {num_layers_str}, Est. Full VRAM: {est_vram_str}GB"
        )
        if hasattr(self, 'model_info_label') and self.model_info_label.winfo_exists():
            self.model_info_label.configure(text=info_text)
        self.log_to_console(f"Model Analysis Complete - {info_text}")
        self.on_model_analyzed()
            
        # If in tuning mode and this is the model being tuned, update tuning analysis
        if self.tuning_in_progress and self.current_tuning_model_path == model_filepath_to_analyze:
            self.current_tuning_model_analysis = self.model_analysis_info.copy()
            self.update_tuning_display() # Refresh tuning UI with new analysis
        
    def on_model_analyzed(self):
        """Called after a model is analyzed to handle special cases"""
        if not self.model_analysis_info:
            return
            
        # Special handling for MoE models to ensure they get enough layers
        if self.model_analysis_info.get('is_moe', False):
            # Get current layer count
            num_layers = self.model_analysis_info.get('num_layers', 32)
            
            # For large 21B+ MoE models like Dark Champion, ensure we have reasonable layer count
            model_name = os.path.basename(self.current_model_path or "").lower()
            if ('dark-champion' in model_name or 'inst-21b' in model_name) and num_layers < 32:
                # Override with a more reasonable layer count for this model family
                self.model_analysis_info['num_layers'] = 48
                self.log_to_console(f"Applied MoE-specific enhancement: Layer count adjusted to 48 for {model_name}")
                
            # Check for very low layer counts that don't make sense
            if num_layers < 10 and self.model_analysis_info.get('size_b', 0) > 10:
                # This is likely incorrect - apply a correction based on model size
                size_b = self.model_analysis_info.get('size_b', 0)
                corrected_layers = 48 if size_b > 20 else 32
                self.model_analysis_info['num_layers'] = corrected_layers
                self.log_to_console(f"Corrected suspicious layer count: Adjusted from {num_layers} to {corrected_layers} based on model size")    

    
    def view_history_for_current_model(self):
        model_path_for_history = None
        if self.tuning_in_progress and self.current_tuning_model_path:
            model_path_for_history = self.current_tuning_model_path
            self.log_to_console(f"Viewing history for current tuning model: {os.path.basename(model_path_for_history)}")
        elif self.current_model_path: # Model selected in main view but not tuning
            model_path_for_history = self.current_model_path
            self.log_to_console(f"Viewing history for selected model: {os.path.basename(model_path_for_history)}")
        
        if model_path_for_history:
            self.load_history(model_filter=model_path_for_history)
            if hasattr(self, 'tabview'):
                self.tabview.set("History")
        else:
            messagebox.showwarning("No Model Context", "No model is currently selected or active in a tuning session to show specific history.", parent=self)

    def setup_history_tab(self):
        self.history_tab_content_frame = ctk.CTkFrame(self.tab_history, fg_color="transparent")
        self.history_tab_content_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.history_tab_content_frame.grid_columnconfigure(0, weight=1)
        self.history_tab_content_frame.grid_rowconfigure(1, weight=1)

        history_controls_frame = ctk.CTkFrame(self.history_tab_content_frame)
        history_controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.btn_refresh_history = ctk.CTkButton(history_controls_frame, text="Refresh History", command=lambda: self.load_history(model_filter=None))
        self.btn_refresh_history.pack(side="left", padx=10, pady=5)
        ToolTip(self.btn_refresh_history, "Reload and display all launch history records from the database.")
        
        self.history_title_label = ctk.CTkLabel(history_controls_frame, text="Loading history...", justify="left", anchor="w")
        self.history_title_label.pack(side="left", padx=10, pady=5, fill="x", expand=True)

        self.history_scrollable_frame = ctk.CTkScrollableFrame(self.history_tab_content_frame)
        self.history_scrollable_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self.history_scrollable_frame.grid_columnconfigure(0, weight=1)

        self.load_history()

    def load_history(self, model_filter=None):
        if not hasattr(self, 'history_scrollable_frame') or not self.history_scrollable_frame.winfo_exists():
            self.log_to_console("History scrollable frame not ready.")
            return
        if not hasattr(self, 'history_title_label') or not self.history_title_label.winfo_exists():
            self.log_to_console("History title label not ready.")
            return

        for widget in self.history_scrollable_frame.winfo_children():
            if widget and widget.winfo_exists(): # Check existence before destroy
                 widget.destroy()
        
        self.history_title_label.configure(text="Fetching history...")
        try:
            limit = 100 # Max entries to fetch initially for performance
            all_entries = tensortune_core.get_history_entries(self.db_path, limit=limit)
            display_title_text = ""
            entries_to_show = []

            if not all_entries:
                display_title_text = "No launch history found."
            elif model_filter:
                entries_to_show = [entry for entry in all_entries if entry[0] == model_filter]
                if not entries_to_show:
                    display_title_text = f"No history found for model: {os.path.basename(model_filter)}"
                else:
                    # Show most recent N for this model
                    display_title_text = f"Launch History for {os.path.basename(model_filter)} (Last {min(len(entries_to_show), 20)})"
                    entries_to_show = entries_to_show[:20] 
            else: # Global history
                display_title_text = f"Global Launch History (Last {min(len(all_entries), 20)})"
                entries_to_show = all_entries[:20]

            self.history_title_label.configure(text=display_title_text, font=ctk.CTkFont(weight="bold"))

            if not entries_to_show:
                ctk.CTkLabel(self.history_scrollable_frame, text="No relevant history entries to display.").pack(fill="x", pady=10, padx=10)
                return

            for record in entries_to_show:
                # Unpack record, assuming structure from core:
                # (model_path, size_b, quant, is_moe, vram_at_launch_decision_mb, 
                #  attempt_level, outcome, approx_vram_used_kcpp_mb, timestamp)
                model_fn = os.path.basename(record[0])
                size_b = f"{record[1]:.1f}B" if isinstance(record[1], float) else (str(record[1]) + "B" if record[1] is not None else "N/A")
                quant = str(record[2]) if record[2] else "N/A"
                moe = "Y" if record[3] else "N" # Boolean to Y/N
                vram_l = f"{record[4]}MB" if record[4] is not None else "N/A" # VRAM at launch
                lvl = str(record[5]) if record[5] is not None else "N/A" # Attempt level
                outcome = str(record[6]) if record[6] else "N/A"
                vram_u = f"{record[7]}MB" if record[7] is not None else "N/A" # VRAM used
                ts_obj = record[8] # Timestamp object
                ts_str = ts_obj.strftime('%Y-%m-%d %H:%M') if isinstance(ts_obj, tensortune_core.datetime) else str(ts_obj)

                entry_text = (
                    f"Model: {model_fn} ({size_b}, {quant}, MoE:{moe})\n"
                    f"  Lvl: {lvl}, VRAM@Launch: {vram_l}, Actual VRAM Used: {vram_u}\n"
                    f"  Outcome: {outcome}\n"
                    f"  Time: {ts_str}"
                )
                entry_label = ctk.CTkLabel(self.history_scrollable_frame, text=entry_text, justify="left", anchor="w")
                entry_label.pack(fill="x", padx=10, pady=(5, 2))
                
                separator = ctk.CTkFrame(self.history_scrollable_frame, height=1, fg_color="gray50") # Visual separator
                separator.pack(fill="x", padx=10, pady=(0,5))

        except Exception as e:
            self.log_to_console(f"Error loading history: {e}")
            traceback.print_exc()
            if hasattr(self, 'history_title_label') and self.history_title_label.winfo_exists():
                self.history_title_label.configure(text="Error loading history. Check log.")

    def stop_tracked_kcpp_processes(self):
        self.log_to_console("Attempting to stop tracked KoboldCpp processes...")
        killed_any = False
        # Stop directly launched process
        if self.last_process and self.last_process.poll() is None:
            pid_to_kill = self.last_process.pid
            self.log_to_console(f"Stopping directly launched KCPP (PID: {pid_to_kill})...")
            success, msg = tensortune_core.kill_process(pid_to_kill, force=True)
            self.log_to_console(f"Kill PID {pid_to_kill}: {success} - {msg}")
            if success:
                killed_any = True
                self.last_process = None
                self.process_running = False
        
        # Stop monitored process (if any, and if user hasn't already stopped it via its own button)
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
            pid_to_kill_monitor = self.kcpp_process_obj.pid
            self.log_to_console(f"Stopping monitored KCPP (PID: {pid_to_kill_monitor})...")
            self.user_requested_stop_monitoring = True # Signal monitor to stop too
            success_mon, msg_mon = tensortune_core.kill_process(pid_to_kill_monitor, force=True)
            self.log_to_console(f"Kill monitored PID {pid_to_kill_monitor}: {success_mon} - {msg_mon}")
            if success_mon:
                killed_any = True
                self.kcpp_process_obj = None # Clear it

        if not killed_any:
            self.log_to_console("No tracked KCPP processes were found running.")
        else:
            self.log_to_console("Tracked KCPP process stop attempt finished.")
        
        messagebox.showinfo("Process Stop Attempted", "Attempted to stop tracked KoboldCpp processes. Check the launcher log for details.", parent=self)

    def stop_all_kcpp_processes_forcefully(self):
        self.log_to_console("Attempting to stop ALL KoboldCpp processes (sweep)...")
        # First, stop any processes tracked by the launcher itself
        self.stop_tracked_kcpp_processes() # This handles self.last_process and self.kcpp_process_obj

        # If a tuning session is active, end it to prevent further launches
        if self.tuning_in_progress:
            self.end_tuning_session(switch_to_model_selection=False) # Don't switch view yet

        self.check_koboldcpp_executable() # Ensure self.koboldcpp_executable is current
        if self.koboldcpp_executable:
            kcpp_exe_basename = os.path.basename(self.koboldcpp_executable)
            self.log_to_console(f"Performing sweep for processes matching '{kcpp_exe_basename}'...")
            killed_sweep, msg_sweep = tensortune_core.kill_processes_by_name(kcpp_exe_basename)
            self.log_to_console(f"Sweep result: {msg_sweep}")

            # If it's a python script, also try to kill python processes running that script
            if self.koboldcpp_executable.lower().endswith(".py"):
                self.log_to_console(f"Performing sweep for python processes running '{kcpp_exe_basename}'...")
                killed_py_sweep, msg_py_sweep = tensortune_core.kill_processes_by_name("python", cmdline_substr_filter=kcpp_exe_basename)
                self.log_to_console(f"Python sweep result: {msg_py_sweep}")
        else:
            self.log_to_console("KoboldCpp executable path not configured, cannot perform broad sweep by name.")
        
        messagebox.showinfo("Process Stop Attempted", "Attempted to stop all KoboldCpp processes. Check the launcher log for details.", parent=self)

    def _get_param_definitions_for_dialog(self):
        # Filter out --model as it's not typically user-editable in these dialogs
        return [d for d in tensortune_core.KOBOLDCPP_ARG_DEFINITIONS if d["key"] != "--model"]


class FirstTimeSetupDialog(ctk.CTkToplevel):
    def __init__(self, parent, current_config, current_exe_path):
        super().__init__(parent)
        self.title("First-Time Setup")
        self.geometry("650x400")
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        self.parent_app = parent
        self.config_to_update = current_config.copy()
        self.saved_config = None

        # Continue with the original initialization code
        # ...

    def _on_save_setup(self):
        try:
            # Process the inputs and update config as before
            # ...
            
            # When ready to close:
            save_ok, save_msg = tensortune_core.save_launcher_config(self.config_to_update)
            if save_ok:
                self.saved_config = self.config_to_update
                
                # Use safer dialog closing
                try:
                    if self.winfo_exists():
                        self.grab_release()
                        self.destroy()
                        
                    # Use a safer method to focus the parent with delay
                    if hasattr(self.parent_app, 'after') and hasattr(self.parent_app, 'winfo_exists'):
                        self.parent_app.after(200, lambda: self.parent_app.focus_force() if self.parent_app.winfo_exists() else None)
                except Exception as close_err:
                    print(f"Non-critical error during dialog close: {close_err}")
            else:
                messagebox.showerror("Save Error", f"Failed to save initial configuration:\n{save_msg}", parent=self)
                self.saved_config = None
        except Exception as e:
            print(f"Error in setup dialog: {str(e)}")
            import traceback
            print(traceback.format_exc())
            if hasattr(self.parent_app, 'log_to_console'):
                self.parent_app.log_to_console(f"Error in setup dialog: {str(e)}")
            self.saved_config = None

    def _on_cancel(self):
        try:
            self.saved_config = None
            
            # Use safer dialog closing
            try:
                if self.winfo_exists():
                    self.grab_release()
                    self.destroy()
                    
                # Use a safer method to focus the parent with delay
                if hasattr(self.parent_app, 'after') and hasattr(self.parent_app, 'winfo_exists'):
                    self.parent_app.after(200, lambda: self.parent_app.focus_force() if self.parent_app.winfo_exists() else None)
            except Exception as close_err:
                print(f"Non-critical error during dialog cancel: {close_err}")
        except Exception as e:
            print(f"Error closing setup dialog: {str(e)}")
            if hasattr(self.parent_app, 'log_to_console'):
                self.parent_app.log_to_console(f"Error closing setup dialog: {str(e)}")



if __name__ == "__main__":
    # NVML cleanup hook if applicable
    if hasattr(tensortune_core, '_cleanup_nvml') and callable(tensortune_core._cleanup_nvml):
        import atexit
        atexit.register(tensortune_core._cleanup_nvml)
    
    main_app_instance = KoboldLauncherGUI()
    # The window close protocol is now handled internally by KoboldLauncherGUI._on_gui_close_requested
    main_app_instance.mainloop()