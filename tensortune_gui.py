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
import koboldcpp_core
import platform
from pathlib import Path
import webbrowser
import traceback

# --- ToolTip Class (Helper for rich tooltips) ---
class ToolTip(ctk.CTkToplevel):
    def __init__(self, widget, text, delay=500):
        super().__init__(widget)
        self.widget = widget
        self.text = text
        self.delay = delay
        self._scheduled_show = None

        self.withdraw()
        self.overrideredirect(True)

        current_mode = ctk.get_appearance_mode()
        if current_mode == "Light":
            bg_color = "gray90"
            text_color = "black"
        else:
            bg_color = "gray20"
            text_color = "white"

        self.attributes("-alpha", 0.92)
        self.attributes("-topmost", True)

        self.label = ctk.CTkLabel(self, text=self.text, fg_color=bg_color, text_color=text_color,
                                  corner_radius=4, padx=6, pady=4, wraplength=350)
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

        self.title("TensorTune (GUI Edition v1.0.0)") # Incremented for fix
        self.geometry("950x880")
        self.minsize(900, 780)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        core_init_results = koboldcpp_core.initialize_launcher()
        self.config = core_init_results["config"]
        # Ensure these keys exist after loading, even if old config didn't have them
        self.config.setdefault("gpu_selection_mode", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["gpu_selection_mode"])
        self.config.setdefault("selected_gpu_index", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["selected_gpu_index"])
        self.config.setdefault("override_vram_budget", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["override_vram_budget"])
        self.config.setdefault("manual_vram_total_mb", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["manual_vram_total_mb"])


        self.system_info = core_init_results["system_info"]
        self.gpu_info = core_init_results["gpu_info"] # This is now the rich dictionary
        self.koboldcpp_capabilities = core_init_results.get("koboldcpp_capabilities", {})

        if "model_specific_args" not in self.config:
            self.config["model_specific_args"] = {}

        self.settings_dirty = False
        self.current_model_path = None
        self.process_running = False
        self.model_analysis_info = {}
        self.last_process = None
        self.db_path = self.config["db_file"]
        self.default_model_dir = self.config.get("default_gguf_dir", os.getcwd())
        self.koboldcpp_executable = self.config.get("koboldcpp_executable", "")

        self.tuning_in_progress = False
        self.current_tuning_attempt_level = 0
        self.current_tuning_min_level = 0
        self.current_tuning_max_level = 0
        self.current_tuning_session_base_args = {}
        self.current_tuning_model_analysis = {}
        self.current_tuning_model_path = None
        self.level_of_last_monitored_run = 0
        self.current_command_list_for_db = []
        self.vram_at_decision_for_db = None # This stores actual free HW VRAM before KCPP launch

        self.kcpp_monitor_thread = None
        self.kcpp_process_obj = None
        self.kcpp_success_event = threading.Event()
        self.kcpp_oom_event = threading.Event()
        self.kcpp_output_lines_shared = []
        self.monitor_start_time = 0.0
        self.last_approx_vram_used_kcpp_mb = None # Actual VRAM used by KCPP (HW based)
        self.last_free_vram_after_load_mb = None # Budgeted free VRAM after load
        self.last_successful_monitored_run_details_gui = None

        self.gpu_selection_mode_var = ctk.StringVar(value=self.config.get("gpu_selection_mode", "auto"))
        self.selected_gpu_id_display_var = ctk.StringVar(value="N/A (Auto-Detect)")
        self.override_vram_var = ctk.BooleanVar(value=self.config.get("override_vram_budget", False))

        appearance_mode = self.config.get("color_mode", "dark").lower()
        if appearance_mode not in ["dark", "light", "system"]:
            appearance_mode = "dark"
        ctk.set_appearance_mode(appearance_mode)
        ctk.set_default_color_theme("blue")

        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.tab_main = self.tabview.add("Tune & Launch")
        self.tab_settings = self.tabview.add("Settings")
        self.tab_history = self.tabview.add("History")

        self.tabview.set("Tune & Launch")

        for tab in [self.tab_main, self.tab_settings, self.tab_history]:
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)

        self.setup_settings_tab()
        self.load_settings_from_config() # Ensure this handles new config structure

        self.setup_main_tab()
        self.setup_history_tab()

        threading.Thread(target=self.monitor_vram, daemon=True).start()

        self.log_to_console(f"TensorTune GUI Initialized. Core: {self.config.get('launcher_core_version', 'N/A')}")
        self.log_to_console(f"Config loaded: {core_init_results['config_message']}")
        if not core_init_results["db_success"]:
            self.log_to_console(f"DB Warning: {core_init_results['db_message']}")
        self.log_to_console(f"Using DB at: {self.db_path}")
        self.log_to_console(f"Initial GPU Info: {self.gpu_info.get('message', 'N/A')}")
        kcpp_caps_info = self.koboldcpp_capabilities
        if "error" in kcpp_caps_info:
             self.log_to_console(f"KCPP Caps Error: {kcpp_caps_info['error']}")
        else:
            self.log_to_console(f"KCPP Caps: CUDA:{kcpp_caps_info.get('cuda', False)}, ROCm:{kcpp_caps_info.get('rocm',False)}, FlashAttn:{kcpp_caps_info.get('flash_attn',False)}")

        self.check_koboldcpp_executable()
        self._show_model_selection_view()
        self.after(100, self._run_first_time_setup_if_needed)
        self.after(200, self._populate_gpu_id_dropdown_on_startup)

        self.update_save_button_state()

    def browse_executable(self):
        current_path_in_entry = ""
        if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
            current_path_in_entry = self.exe_path_entry.get()
        else:
            print("DEBUG: exe_path_entry not found in browse_executable while trying to get current path.")

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
            else:
                print("DEBUG: exe_path_entry not found in browse_executable when trying to set selected path.")
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

        self.model_selection_frame = ctk.CTkFrame(self.tab_main)
        self.model_selection_frame.grid_columnconfigure(0, weight=1)
        self.model_selection_frame.grid_rowconfigure(5, weight=1)

        title_label = ctk.CTkLabel(self.model_selection_frame, text="TensorTune Model Launcher", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 15), sticky="n")

        model_controls_frame = ctk.CTkFrame(self.model_selection_frame)
        model_controls_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        model_controls_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(model_controls_frame, text="GGUF Model:").grid(row=0, column=0, padx=(10,5), pady=10, sticky="w")
        self.model_path_entry = ctk.CTkEntry(model_controls_frame, width=400)
        self.model_path_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        btn_browse_model = ctk.CTkButton(model_controls_frame, text="Browse", command=self.browse_model, width=80)
        btn_browse_model.grid(row=0, column=2, padx=(5,10), pady=10, sticky="e")
        ToolTip(btn_browse_model, "Select a .gguf model file to load.")

        self.model_info_label = ctk.CTkLabel(model_controls_frame, text="No model selected. Analysis includes: Type, Size, Quant, Layers, Est. VRAM.", justify="left", wraplength=650, font=("Segoe UI", 11))
        self.model_info_label.grid(row=1, column=0, columnspan=3, padx=10, pady=(0,10), sticky="w")

        vram_frame = ctk.CTkFrame(self.model_selection_frame)
        vram_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        vram_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(vram_frame, text="GPU Status:").grid(row=0, column=0, padx=(10,5), pady=5, sticky="w")
        self.vram_progress = ctk.CTkProgressBar(vram_frame, height=18)
        self.vram_progress.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.vram_progress.set(0)
        self.vram_text = ctk.CTkLabel(vram_frame, text="Scanning...", font=("Segoe UI", 10), width=300, anchor="w")
        self.vram_text.grid(row=0, column=2, padx=(5,10), pady=5, sticky="ew")
        btn_refresh_vram = ctk.CTkButton(vram_frame, text="Refresh", width=60, command=self.refresh_vram)
        btn_refresh_vram.grid(row=0, column=3, padx=(5,10), pady=5, sticky="e")
        ToolTip(btn_refresh_vram, "Manually refresh GPU VRAM information based on current GPU selection settings.")


        launch_buttons_frame = ctk.CTkFrame(self.model_selection_frame)
        launch_buttons_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        launch_buttons_frame.grid_columnconfigure((0,1,2), weight=1)

        btn_start_tune = ctk.CTkButton(launch_buttons_frame, text="Start Auto-Tune / Use OT Strategy", command=self.start_tuning_session, height=35, fg_color="seagreen", hover_color="darkgreen")
        btn_start_tune.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ToolTip(btn_start_tune, "Begin an interactive session to find optimal Tensor Offload (OT)\nsettings for the selected model and current VRAM.")

        btn_launch_best = ctk.CTkButton(launch_buttons_frame, text="Launch Best Remembered Config", command=self.launch_best_remembered, height=35, fg_color="cornflowerblue", hover_color="royalblue")
        btn_launch_best.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ToolTip(btn_launch_best, "Load the model using the most successful configuration\npreviously found in the launch history for this model and similar VRAM.")

        btn_launch_direct = ctk.CTkButton(launch_buttons_frame, text="Direct Launch (Settings Defaults)", command=self.launch_direct_defaults, height=35)
        btn_launch_direct.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        ToolTip(btn_launch_direct, "Launch the model directly using your current global default\nsettings combined with any model-specific settings, without auto-tuning.")


        stop_button_frame = ctk.CTkFrame(self.model_selection_frame)
        stop_button_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=(0,5), sticky="ew")
        stop_button_frame.grid_columnconfigure(0, weight=1)
        btn_stop_all = ctk.CTkButton(stop_button_frame, text="Stop Any KCPP Processes", command=self.stop_all_kcpp_processes_forcefully, height=35, fg_color="firebrick", hover_color="darkred")
        btn_stop_all.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ToolTip(btn_stop_all, "Attempt to forcefully terminate any running KoboldCpp processes\nthat might have been launched by this tool or match the configured executable name.\nUSE WITH CAUTION.")


        console_frame_ms = ctk.CTkFrame(self.model_selection_frame)
        console_frame_ms.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        console_frame_ms.grid_columnconfigure(0, weight=1)
        console_frame_ms.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(console_frame_ms, text="Launcher Log:").grid(row=0, column=0, padx=10, pady=(5,0), sticky="w")
        self.console = ctk.CTkTextbox(console_frame_ms, height=120, wrap="word")
        self.console.grid(row=1, column=0, padx=10, pady=(0,5), sticky="nsew")
        self.console.configure(state="disabled")

        self.tuning_mode_frame = ctk.CTkFrame(self.tab_main)
        self.tuning_mode_frame.grid_columnconfigure(0, weight=1)
        self.tuning_mode_frame.grid_rowconfigure(10, weight=1)

        tuning_title_label = ctk.CTkLabel(self.tuning_mode_frame, text="Auto-Tuning Session", font=ctk.CTkFont(size=18, weight="bold"))
        tuning_title_label.grid(row=0, column=0, padx=10, pady=(10,0), sticky="n")
        self.tuning_model_name_label = ctk.CTkLabel(self.tuning_mode_frame, text="Model: N/A", font=ctk.CTkFont(size=14))
        self.tuning_model_name_label.grid(row=1, column=0, padx=10, pady=(0,5), sticky="n")

        self.tuning_view_vram_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.tuning_view_vram_frame.grid(row=2, column=0, padx=10, pady=(0,5), sticky="ew")
        self.tuning_view_vram_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.tuning_view_vram_frame, text="GPU Status:").grid(row=0, column=0, padx=(10,5), pady=5, sticky="w")
        self.tuning_view_vram_progress = ctk.CTkProgressBar(self.tuning_view_vram_frame, height=18)
        self.tuning_view_vram_progress.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.tuning_view_vram_progress.set(0)
        self.tuning_view_vram_text = ctk.CTkLabel(self.tuning_view_vram_frame, text="Scanning...", font=("Segoe UI", 10), width=300, anchor="w")
        self.tuning_view_vram_text.grid(row=0, column=2, padx=(5,10), pady=5, sticky="ew")
        btn_refresh_vram_tuning = ctk.CTkButton(self.tuning_view_vram_frame, text="Refresh", width=60, command=self.refresh_vram)
        btn_refresh_vram_tuning.grid(row=0, column=3, padx=(5,10), pady=5, sticky="e")
        ToolTip(btn_refresh_vram_tuning, "Manually refresh GPU VRAM information.")

        self.last_run_info_frame_tuning = ctk.CTkFrame(self.tuning_mode_frame)
        self.last_run_info_frame_tuning.grid(row=3, column=0, padx=10, pady=2, sticky="ew")
        self.last_run_info_frame_tuning.grid_columnconfigure(0, weight=1)
        self.tuning_last_run_info_label = ctk.CTkLabel(self.last_run_info_frame_tuning, text="Last Monitored Success: None yet in this session.", justify="left", font=ctk.CTkFont(size=11), text_color="gray", anchor="w")
        self.tuning_last_run_info_label.grid(row=0, column=0, padx=5, pady=2, sticky="ew")

        ot_strategy_display_frame = ctk.CTkFrame(self.tuning_mode_frame)
        ot_strategy_display_frame.grid(row=4, column=0, padx=10, pady=2, sticky="ew")
        ot_strategy_display_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(ot_strategy_display_frame, text="Current OT Strategy:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.tuning_ot_level_label = ctk.CTkLabel(ot_strategy_display_frame, text="Level: N/A", justify="left", anchor="w")
        self.tuning_ot_level_label.grid(row=1, column=0, padx=10, pady=1, sticky="ew")
        self.tuning_ot_range_label = ctk.CTkLabel(ot_strategy_display_frame, text="Range: N/A", justify="left", anchor="w")
        self.tuning_ot_range_label.grid(row=2, column=0, padx=10, pady=1, sticky="ew")
        self.tuning_ot_strategy_desc_label = ctk.CTkLabel(ot_strategy_display_frame, text="Strategy: N/A", justify="left", wraplength=650, anchor="w")
        self.tuning_ot_strategy_desc_label.grid(row=3, column=0, padx=10, pady=1, sticky="ew")
        self.tuning_ot_regex_label = ctk.CTkLabel(ot_strategy_display_frame, text="Regex: N/A", justify="left", wraplength=650, font=("Courier New", 10), anchor="w")
        self.tuning_ot_regex_label.grid(row=4, column=0, padx=10, pady=1, sticky="ew")
        self.tuning_gpu_layers_label = ctk.CTkLabel(ot_strategy_display_frame, text="GPU Layers: N/A", justify="left", anchor="w")
        self.tuning_gpu_layers_label.grid(row=5, column=0, padx=10, pady=1, sticky="ew")

        proposed_command_frame = ctk.CTkFrame(self.tuning_mode_frame)
        proposed_command_frame.grid(row=5, column=0, padx=10, pady=2, sticky="ew")
        proposed_command_frame.grid_columnconfigure(0, weight=1)
        command_label_btn_frame = ctk.CTkFrame(proposed_command_frame); command_label_btn_frame.pack(fill="x", pady=(0,2))
        ctk.CTkLabel(command_label_btn_frame, text="Proposed Command:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5, pady=0)
        self.btn_copy_command = ctk.CTkButton(command_label_btn_frame, text="Copy", width=60, height=24, command=self.copy_proposed_command)
        self.btn_copy_command.pack(side="right", padx=5, pady=0)
        ToolTip(self.btn_copy_command, "Copy the full proposed launch command to the clipboard.")
        self.tuning_proposed_command_text = ctk.CTkTextbox(proposed_command_frame, height=100, wrap="word", font=("Courier New", 10))
        self.tuning_proposed_command_text.pack(fill="x", expand=True, padx=5, pady=0)
        self.tuning_proposed_command_text.configure(state="disabled")

        self.tuning_actions_primary_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.tuning_actions_primary_frame.grid(row=6, column=0, padx=10, pady=(5,2), sticky="ew")
        self.tuning_actions_primary_frame.grid_columnconfigure((0,1), weight=1)
        self.btn_tune_launch_monitor = ctk.CTkButton(self.tuning_actions_primary_frame, text="Launch & Monitor Output", command=self.launch_and_monitor_for_tuning, height=35, fg_color="seagreen", hover_color="darkgreen")
        self.btn_tune_launch_monitor.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ToolTip(self.btn_tune_launch_monitor, "Launch KoboldCpp with the current OT strategy and monitor its output for success or errors (e.g., OOM).")
        self.btn_tune_skip_launch_direct = ctk.CTkButton(self.tuning_actions_primary_frame, text="Skip Tune & Launch This Config", command=self.skip_tune_and_launch_direct, height=35)
        self.btn_tune_skip_launch_direct.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ToolTip(self.btn_tune_skip_launch_direct, "Immediately launch KoboldCpp for use with the current OT strategy and base arguments,\nwithout further monitoring or tuning steps.")

        self.tuning_actions_secondary_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.tuning_actions_secondary_frame.grid(row=7, column=0, padx=10, pady=0, sticky="ew")
        self.tuning_actions_secondary_frame.grid_columnconfigure((0,1), weight=1)
        self.btn_tune_more_gpu = ctk.CTkButton(self.tuning_actions_secondary_frame, text="More GPU (↓ Level)", command=lambda: self.adjust_ot_level(-1))
        self.btn_tune_more_gpu.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ToolTip(self.btn_tune_more_gpu, "Adjust the OT strategy to offload more layers/tensors to the GPU (decreases OT level).")
        self.btn_tune_more_cpu = ctk.CTkButton(self.tuning_actions_secondary_frame, text="More CPU (↑ Level)", command=lambda: self.adjust_ot_level(1))
        self.btn_tune_more_cpu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ToolTip(self.btn_tune_more_cpu, "Adjust the OT strategy to offload more layers/tensors to the CPU (increases OT level),\nreducing VRAM usage.")

        self.tuning_edit_args_buttons_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.tuning_edit_args_buttons_frame.grid(row=8, column=0, padx=10, pady=2, sticky="ew")
        self.tuning_edit_args_buttons_frame.grid_columnconfigure((0,1), weight=1)
        self.btn_tune_edit_args = ctk.CTkButton(self.tuning_edit_args_buttons_frame, text="Edit Base Args (This Session)", command=self.edit_base_args_for_tuning_session)
        self.btn_tune_edit_args.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ToolTip(self.btn_tune_edit_args, "Modify the base KoboldCpp arguments (e.g., context size, threads)\nfor this tuning session only. These changes are not saved permanently unless you also use 'Edit Permanent Args'.")
        self.btn_tune_edit_model_perm_args = ctk.CTkButton(self.tuning_edit_args_buttons_frame, text="Edit Base Args (Permanent for This Model)", command=self.edit_permanent_model_args)
        self.btn_tune_edit_model_perm_args.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ToolTip(self.btn_tune_edit_model_perm_args, "Modify and save the base KoboldCpp arguments specifically for the current model.\nThese will become the new defaults when this model is loaded.")

        self.tuning_actions_navigation_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.tuning_actions_navigation_frame.grid(row=9, column=0, padx=10, pady=2, sticky="ew")
        self.tuning_actions_navigation_frame.grid_columnconfigure((0,1,2), weight=1)
        self.btn_tune_new_gguf = ctk.CTkButton(self.tuning_actions_navigation_frame, text="New GGUF Model", command=self.select_new_gguf_during_tuning)
        self.btn_tune_new_gguf.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ToolTip(self.btn_tune_new_gguf, "End the current tuning session and return to the model selection screen.")
        self.btn_tune_history = ctk.CTkButton(self.tuning_actions_navigation_frame, text="View History (This Model)", command=lambda: self.view_history_for_current_model())
        self.btn_tune_history.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ToolTip(self.btn_tune_history, "Open the History tab, filtered to show launch records for the current model.")
        self.btn_tune_quit_tuning = ctk.CTkButton(self.tuning_actions_navigation_frame, text="End Tuning Session", command=self.end_tuning_session, fg_color="firebrick", hover_color="darkred")
        self.btn_tune_quit_tuning.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        ToolTip(self.btn_tune_quit_tuning, "Stop the current tuning session and return to the main model selection view.")

        self.kcpp_output_console_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.kcpp_output_console_frame.grid(row=10, column=0, padx=10, pady=(5,10), sticky="nsew")
        self.kcpp_output_console_frame.grid_columnconfigure(0, weight=1)
        self.kcpp_output_console_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(self.kcpp_output_console_frame, text="KoboldCpp Output (during monitoring):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.kcpp_live_output_text = ctk.CTkTextbox(self.kcpp_output_console_frame, wrap="char", font=("Segoe UI", 10))
        self.kcpp_live_output_text.grid(row=1, column=0, padx=5, pady=2, sticky="nsew")
        self.kcpp_live_output_text.configure(state="disabled")

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
        """Safely populates the GPU ID dropdown after UI is likely initialized."""
        if hasattr(self, 'gpu_id_optionmenu') and self.gpu_id_optionmenu.winfo_exists():
            self._populate_gpu_id_dropdown()
        else: # Retry if UI not ready
            self.after(500, self._populate_gpu_id_dropdown_on_startup)


    def setup_settings_tab(self):
        sf = ctk.CTkScrollableFrame(self.tab_settings)
        sf.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        # UI Theme
        ui_theme_frame = ctk.CTkFrame(sf)
        ui_theme_frame.pack(fill="x", padx=10, pady=(10,5))
        ctk.CTkLabel(ui_theme_frame, text="UI Theme:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=(5,10), pady=10, sticky="w")
        self.theme_var = ctk.StringVar(value=self.config.get("color_mode", "dark").lower()) # Use config val
        theme_option_menu = ctk.CTkOptionMenu(ui_theme_frame, values=["dark", "light", "system"], variable=self.theme_var, command=self.change_theme)
        theme_option_menu.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.theme_var.trace_add("write", self.mark_settings_dirty)

        # KCPP Executable Path
        exe_path_frame = ctk.CTkFrame(sf)
        exe_path_frame.pack(fill="x", padx=10, pady=5)
        exe_path_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(exe_path_frame, text="KoboldCpp Executable:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=(5,10), pady=10, sticky="w")
        self.exe_path_entry = ctk.CTkEntry(exe_path_frame, width=400)
        self.exe_path_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.exe_path_entry.bind("<KeyRelease>", self.mark_settings_dirty)
        btn_browse_exe = ctk.CTkButton(exe_path_frame, text="Browse", command=self.browse_executable, width=80)
        btn_browse_exe.grid(row=0, column=2, padx=10, pady=10, sticky="e")
        ToolTip(btn_browse_exe, "Select your KoboldCpp executable file (e.g., koboldcpp.exe or a .py script).")

        # KCPP Capabilities
        kcpp_caps_frame = ctk.CTkFrame(sf)
        kcpp_caps_frame.pack(fill="x", expand=True, padx=10, pady=(10,5))
        kcpp_caps_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(kcpp_caps_frame, text="Detected KoboldCpp Capabilities:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w",pady=(0,5), padx=5)
        self.kcpp_caps_text_widget = ctk.CTkTextbox(kcpp_caps_frame, height=100, wrap="word", font=("Segoe UI", 11))
        self.kcpp_caps_text_widget.pack(fill="x", expand=True, padx=5, pady=(0,5))
        self.kcpp_caps_text_widget.configure(state="disabled")
        self.btn_redetect_caps = ctk.CTkButton(kcpp_caps_frame, text="Re-detect Capabilities", command=lambda: self.update_kcpp_capabilities_display(re_detect=True))
        self.btn_redetect_caps.pack(pady=(0,5), anchor="e", padx=5)
        ToolTip(self.btn_redetect_caps, "Queries the configured KoboldCpp executable with '--help'\nto determine its supported features and arguments.")

        # Launcher Behavior
        launcher_behavior_frame = ctk.CTkFrame(sf)
        launcher_behavior_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(launcher_behavior_frame, text="Launcher Behavior:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=(5,10), pady=5, sticky="w")
        self.auto_open_webui_var = ctk.BooleanVar(value=self.config.get("auto_open_webui", True)) # Use config val
        auto_open_checkbox = ctk.CTkCheckBox(launcher_behavior_frame, text="Auto-Open Web UI After Successful Launch", variable=self.auto_open_webui_var)
        auto_open_checkbox.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        self.auto_open_webui_var.trace_add("write", self.mark_settings_dirty)

        # --- NEW GPU Management Frame ---
        gpu_management_frame = ctk.CTkFrame(sf)
        gpu_management_frame.pack(fill="x", padx=10, pady=(20,10))
        ctk.CTkLabel(gpu_management_frame, text="GPU Management & VRAM Override", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", pady=(0,10), padx=5)

        gpu_select_subframe = ctk.CTkFrame(gpu_management_frame)
        gpu_select_subframe.pack(fill="x", pady=(0,5))
        ctk.CTkLabel(gpu_select_subframe, text="Target GPU Type:").pack(side="left", padx=(5,5), pady=5)
        self.gpu_type_optionmenu = ctk.CTkOptionMenu(
            gpu_select_subframe, variable=self.gpu_selection_mode_var,
            values=["auto", "nvidia", "amd", "intel", "apple"],
            command=self._gpu_type_selection_changed
        )
        self.gpu_type_optionmenu.pack(side="left", padx=5, pady=5)
        self.gpu_selection_mode_var.trace_add("write", self.mark_settings_dirty)
        ToolTip(self.gpu_type_optionmenu, "Select GPU vendor for specific ID selection, or 'auto' for default detection.\nChanges VRAM display source and targeted GPU for launches if not 'auto'.")

        ctk.CTkLabel(gpu_select_subframe, text="Target GPU ID:").pack(side="left", padx=(15,5), pady=5)
        self.gpu_id_optionmenu = ctk.CTkOptionMenu(
            gpu_select_subframe, variable=self.selected_gpu_id_display_var,
            values=["N/A (Auto-Detect)"], # Initial placeholder
            command=self._gpu_id_selection_changed
        )
        self.gpu_id_optionmenu.pack(side="left", padx=5, pady=5)
        ToolTip(self.gpu_id_optionmenu, "Select specific GPU ID after choosing type and refreshing list.\n'0' is usually the primary. Affects VRAM display and targeted GPU.")

        self.btn_refresh_gpu_list = ctk.CTkButton(gpu_select_subframe, text="Refresh GPU List", command=self._populate_gpu_id_dropdown, width=120)
        self.btn_refresh_gpu_list.pack(side="left", padx=(10,5), pady=5)
        ToolTip(self.btn_refresh_gpu_list, "Scan for GPUs of the selected type and update the ID list.\nIf type is 'auto', it tries to detect the active GPU type first and lists its IDs.")

        vram_override_subframe = ctk.CTkFrame(gpu_management_frame)
        vram_override_subframe.pack(fill="x", pady=(5,5))
        self.override_vram_checkbox = ctk.CTkCheckBox(
            vram_override_subframe, text="Override Detected Total VRAM for Launcher Calculations",
            variable=self.override_vram_var,
            command=self._toggle_manual_vram_entry_state
        )
        self.override_vram_checkbox.pack(side="left", padx=5, pady=5)
        self.override_vram_var.trace_add("write", self.mark_settings_dirty)
        ToolTip(self.override_vram_checkbox, "Manually set the total VRAM budget the launcher should assume for calculations.\nUseful if auto-detection is problematic or for testing.")

        ctk.CTkLabel(vram_override_subframe, text="Manual Total (MB):").pack(side="left", padx=(10,2), pady=5)
        self.manual_vram_entry = ctk.CTkEntry(vram_override_subframe, width=100)
        self.manual_vram_entry.pack(side="left", padx=2, pady=5)
        self.manual_vram_entry.bind("<KeyRelease>", self.mark_settings_dirty)
        # --- END NEW GPU Management Frame ---

        ctk.CTkLabel(sf, text="Global KoboldCpp Default Arguments", font=ctk.CTkFont(size=16, weight="bold")).pack(fill="x", padx=10, pady=(20,10))
        self.settings_widgets = {}
        for setting_def in koboldcpp_core.KOBOLDCPP_ARG_DEFINITIONS:
            param_key = setting_def["key"]
            if param_key == "--model": continue
            core_default_value = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].get(param_key)
            type_hint = setting_def.get("type_hint", "str")
            arg_frame = ctk.CTkFrame(sf); arg_frame.pack(fill="x", padx=10, pady=3); arg_frame.grid_columnconfigure(2, weight=1)
            ctk.CTkLabel(arg_frame, text=f"{setting_def['name']}:", width=180, anchor="w").grid(row=0, column=0, padx=(5,0), pady=2, sticky="w")
            widget_instance = None
            if type_hint in ["bool", "bool_flag"]:
                initial_bool_value = False
                if isinstance(core_default_value, bool): initial_bool_value = core_default_value
                elif isinstance(core_default_value, str): initial_bool_value = core_default_value.lower() == 'true'
                var = ctk.BooleanVar(value=initial_bool_value)
                widget_instance = ctk.CTkCheckBox(arg_frame, text="", variable=var)
                widget_instance.grid(row=0, column=1, padx=(0,5), pady=2, sticky="w")
                widget_instance.variable = var; var.trace_add("write", self.mark_settings_dirty)
            else:
                widget_instance = ctk.CTkEntry(arg_frame, width=120)
                if core_default_value is not None: widget_instance.insert(0, str(core_default_value))
                widget_instance.grid(row=0, column=1, padx=(0,5), pady=2, sticky="w")
                widget_instance.bind("<KeyRelease>", self.mark_settings_dirty)
            ctk.CTkLabel(arg_frame, text=setting_def.get("help", ""), font=ctk.CTkFont(size=11), text_color="gray", anchor="w", justify="left", wraplength=380).grid(row=0, column=2, padx=5, pady=2, sticky="ew")
            self.settings_widgets[param_key] = {"widget": widget_instance, "type_hint": type_hint}

        ctk.CTkLabel(sf, text="Manage Model-Specific Configurations", font=ctk.CTkFont(size=16, weight="bold")).pack(fill="x", padx=10, pady=(25,10))
        self.model_specifics_outer_frame = ctk.CTkFrame(sf)
        self.model_specifics_outer_frame.pack(fill="x", expand=False, padx=10, pady=5)
        self.model_specifics_list_frame = ctk.CTkScrollableFrame(self.model_specifics_outer_frame, height=180)
        self.model_specifics_list_frame.pack(fill="x", expand=True, padx=0, pady=0)

        action_buttons_frame = ctk.CTkFrame(sf)
        action_buttons_frame.pack(fill="x", padx=10, pady=(20,10))
        self.btn_save_settings_tab = ctk.CTkButton(action_buttons_frame, text="Save Settings", command=self.save_settings_action)
        self.btn_save_settings_tab.pack(side="left", padx=5, pady=5)
        ToolTip(self.btn_save_settings_tab, "Save all changes made in this Settings tab to the launcher's configuration file.")
        btn_reset_settings = ctk.CTkButton(action_buttons_frame, text="Reset All to Defaults", command=self.reset_config_action, fg_color="#dc3545", hover_color="#c82333")
        btn_reset_settings.pack(side="left", padx=5, pady=5)
        ToolTip(btn_reset_settings, "WARNING: Resets all launcher settings to their original defaults.\nA backup of your current config will be attempted.")
        config_io_frame = ctk.CTkFrame(action_buttons_frame); config_io_frame.pack(side="left", padx=(10,0))
        self.btn_export_config = ctk.CTkButton(config_io_frame, text="Export Config", command=self.export_gui_config)
        self.btn_export_config.pack(side="left", padx=(0,5), pady=5)
        ToolTip(self.btn_export_config, "Save all current launcher settings (excluding history DB content)\nto a JSON file for backup or sharing.")
        self.btn_import_config = ctk.CTkButton(config_io_frame, text="Import Config", command=self.import_gui_config)
        self.btn_import_config.pack(side="left", padx=0, pady=5)
        ToolTip(self.btn_import_config, "Load all launcher settings from a previously exported JSON file.\nWARNING: This will overwrite current settings.")


    def _toggle_manual_vram_entry_state(self, *args): # Bound to checkbox command
        if hasattr(self, 'manual_vram_entry') and self.manual_vram_entry.winfo_exists():
            if self.override_vram_var.get():
                self.manual_vram_entry.configure(state="normal")
            else:
                self.manual_vram_entry.configure(state="disabled")
        self.mark_settings_dirty()
        self.refresh_vram() # Update VRAM display whenever this override state changes

    def _gpu_type_selection_changed(self, selected_type: str):
        # self.gpu_selection_mode_var is already updated by CTkOptionMenu's variable binding
        self._populate_gpu_id_dropdown() # Refresh ID list for the new type
        self.mark_settings_dirty()
        self.refresh_vram() # VRAM display depends on the selected type/ID

    def _gpu_id_selection_changed(self, selected_id_display_str: str):
        # self.selected_gpu_id_display_var is updated by CTkOptionMenu.
        # We need to parse the actual ID from this display string to store in self.config
        actual_id_to_store = 0 # Default to 0 if parsing fails or N/A
        if selected_id_display_str and "N/A" not in selected_id_display_str:
            try:
                # Assuming format "ID X: Name" or "(Auto: VENDOR) ID X: Name"
                match = re.search(r"ID\s*(\d+)", selected_id_display_str)
                if match:
                    actual_id_to_store = int(match.group(1))
            except ValueError:
                self.log_to_console(f"Warning: Could not parse GPU ID from '{selected_id_display_str}'. Defaulting to 0.")
        
        if self.config.get("selected_gpu_index") != actual_id_to_store:
            self.config["selected_gpu_index"] = actual_id_to_store # Update config directly
            self.mark_settings_dirty()
        self.refresh_vram() # VRAM display depends on the selected ID

    def _populate_gpu_id_dropdown(self):
        if not (hasattr(self, 'gpu_id_optionmenu') and self.gpu_id_optionmenu.winfo_exists()):
            self.log_to_console("GPU ID dropdown not ready for population.")
            return

        user_selected_type = self.gpu_selection_mode_var.get()
        gpu_list_from_core = []
        effective_type_for_listing = user_selected_type
        display_prefix = ""
        
        self.selected_gpu_id_display_var.set("Scanning...")
        self.gpu_id_optionmenu.configure(values=["Scanning..."])
        self.update_idletasks() 

        if user_selected_type == "auto":
            self.log_to_console("GPU type is 'auto', determining actual detected GPU type for ID listing...")
            # Call get_available_vram_mb with no specific type to let it auto-detect.
            # We pass the *currently configured* selected_gpu_index here, as get_available_vram_mb
            # might use it if it detects a multi-GPU setup of the auto-detected type.
            _, _, _, gpu_info_dict = koboldcpp_core.get_available_vram_mb(
                self.config, 
                target_gpu_type=None, 
                target_gpu_index=self.config.get("selected_gpu_index", 0)
            )
            
            if gpu_info_dict and gpu_info_dict.get("success") and gpu_info_dict.get("type") not in ["Unknown/None_Auto", "N/A", "Unknown/None"]:
                detected_gpu_vendor = gpu_info_dict.get("type", "").lower()
                if detected_gpu_vendor == "apple_metal": detected_gpu_vendor = "apple" # Normalize
                
                if detected_gpu_vendor in ["nvidia", "amd", "intel", "apple"]:
                    effective_type_for_listing = detected_gpu_vendor
                    display_prefix = f"(Auto: {detected_gpu_vendor.upper()}) "
                    self.log_to_console(f"Auto-detected GPU type as '{detected_gpu_vendor}' for ID listing.")
                else:
                    self.log_to_console(f"Auto-detection yielded an unusable type: '{detected_gpu_vendor}'. No IDs to list for 'auto'.")
                    self.gpu_id_optionmenu.configure(values=["N/A (Auto: No specific GPU type found)"])
                    self.selected_gpu_id_display_var.set("N/A (Auto: No specific GPU type found)")
                    return
            else:
                msg = gpu_info_dict.get('message', 'N/A') if gpu_info_dict else "N/A (core func error)"
                self.log_to_console(f"Auto GPU detection failed or no specific GPU found. Message: {msg}")
                self.gpu_id_optionmenu.configure(values=["N/A (Auto: Detection Failed/No GPU)"])
                self.selected_gpu_id_display_var.set("N/A (Auto: Detection Failed/No GPU)")
                return
        
        # Fetch GPU list based on effective_type_for_listing
        if effective_type_for_listing == "nvidia":
            gpu_list_from_core = koboldcpp_core.list_nvidia_gpus()
        elif effective_type_for_listing == "amd":
            if platform.system() == "Windows": gpu_list_from_core = koboldcpp_core.list_amd_gpus_windows()
            elif platform.system() == "Linux": gpu_list_from_core = koboldcpp_core.list_amd_gpus_linux()
        elif effective_type_for_listing == "intel":
            gpu_list_from_core = koboldcpp_core.list_intel_gpus()
        elif effective_type_for_listing == "apple" and platform.system() == "darwin": # platform.system() is 'darwin' for macOS
            gpu_list_from_core = koboldcpp_core.list_apple_gpus()

        display_values = []
        if gpu_list_from_core:
            display_values = [f"{display_prefix}ID {gpu['id']}: {gpu['name']}" for gpu in gpu_list_from_core]
        else: # No GPUs found for the effective_type_for_listing
            if user_selected_type == "auto": # Should have been caught above, but for safety
                display_values = ["N/A (Auto: No GPUs found for detected type)"]
            else:
                display_values = [f"No {effective_type_for_listing.upper()} GPUs Found"]
        
        self.gpu_id_optionmenu.configure(values=display_values)
        
        # Try to set the dropdown to the currently configured selected_gpu_index
        current_config_idx = self.config.get("selected_gpu_index", 0)
        newly_selected_id_str_for_var = None
        
        if display_values and "N/A" not in display_values[0] and "No " not in display_values[0] and "Error" not in display_values[0]:
            for option_str in display_values: # Try to find the exact configured ID
                match = re.search(r"ID\s*(\d+)", option_str)
                if match:
                    try:
                        if int(match.group(1)) == current_config_idx:
                            newly_selected_id_str_for_var = option_str
                            break
                    except ValueError: continue
            
            if not newly_selected_id_str_for_var: # Configured ID not found, default to first
                newly_selected_id_str_for_var = display_values[0]
                match_first = re.search(r"ID\s*(\d+)", display_values[0])
                new_default_parsed_id = 0
                if match_first:
                    try: new_default_parsed_id = int(match_first.group(1))
                    except ValueError: pass
                if self.config.get("selected_gpu_index") != new_default_parsed_id:
                    self.config["selected_gpu_index"] = new_default_parsed_id
                    self.log_to_console(f"Configured GPU index {current_config_idx} not in list for type '{effective_type_for_listing}'. Defaulted to index {new_default_parsed_id}.")
        else: # List is empty or shows an error/N/A message
            newly_selected_id_str_for_var = display_values[0] if display_values else "Error (No GPUs)"
            if self.config.get("selected_gpu_index") != 0: # Reset index if no valid IDs
                 self.config["selected_gpu_index"] = 0

        self.selected_gpu_id_display_var.set(newly_selected_id_str_for_var)
        self.log_to_console(f"GPU ID list for user type '{user_selected_type}' (effective: '{effective_type_for_listing}') refreshed. Displaying: {newly_selected_id_str_for_var}")


    def load_settings_from_config(self):
        if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
            self.exe_path_entry.delete(0, "end"); self.exe_path_entry.insert(0, self.config.get("koboldcpp_executable", ""))
        if hasattr(self, 'theme_var') and self.theme_var:
            self.theme_var.set(self.config.get("color_mode", "dark").lower())
        if hasattr(self, 'auto_open_webui_var') and self.auto_open_webui_var:
            self.auto_open_webui_var.set(self.config.get("auto_open_webui", True))

        self.gpu_selection_mode_var.set(self.config.get("gpu_selection_mode", "auto"))
        # self.selected_gpu_id_display_var will be set by _populate_gpu_id_dropdown called later
        self.override_vram_var.set(self.config.get("override_vram_budget", False))
        if hasattr(self, 'manual_vram_entry') and self.manual_vram_entry.winfo_exists():
            self.manual_vram_entry.delete(0, "end")
            self.manual_vram_entry.insert(0, str(self.config.get("manual_vram_total_mb", 8192)))
        
        self._toggle_manual_vram_entry_state() 

        global_default_args_from_config = self.config.get("default_args", {})
        for arg_definition in koboldcpp_core.KOBOLDCPP_ARG_DEFINITIONS:
            param_key = arg_definition["key"]
            if param_key not in self.settings_widgets: continue
            widget_info = self.settings_widgets[param_key]; widget = widget_info["widget"]; type_hint = arg_definition.get("type_hint", "str")
            value_from_config = global_default_args_from_config.get(param_key)
            core_template_default_value = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].get(param_key)
            final_value_to_set = value_from_config if value_from_config is not None else core_template_default_value
            if type_hint in ["bool", "bool_flag"]:
                bool_value = False
                if isinstance(final_value_to_set, bool): bool_value = final_value_to_set
                elif isinstance(final_value_to_set, str): bool_value = final_value_to_set.lower() == 'true'
                if hasattr(widget, 'variable'): widget.variable.set(bool_value)
                elif isinstance(widget, ctk.CTkCheckBox):
                    if bool_value: 
                        widget.select()
                    else: 
                        widget.deselect() 
            elif isinstance(widget, ctk.CTkEntry):
                widget.delete(0, "end")
                if final_value_to_set is not None: widget.insert(0, str(final_value_to_set))

        if hasattr(self, 'populate_model_specifics_list_display'): self.populate_model_specifics_list_display()
        if hasattr(self, 'update_kcpp_capabilities_display'): self.update_kcpp_capabilities_display(re_detect=False)
        # _populate_gpu_id_dropdown_on_startup (called from __init__) will handle the GPU ID dropdown

        self.settings_dirty = False
        self.update_save_button_state()
        self.log_to_console("Settings tab populated from current configuration.")

    def save_config(self):
        if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
            self.config["koboldcpp_executable"] = self.exe_path_entry.get().strip()
        self.config["default_gguf_dir"] = self.default_model_dir
        self.config["db_file"] = self.db_path
        if hasattr(self, 'theme_var') and self.theme_var:
            self.config["color_mode"] = self.theme_var.get().lower()
        else:
            self.config["color_mode"] = ctk.get_appearance_mode().lower()
        if hasattr(self, 'auto_open_webui_var') and self.auto_open_webui_var:
            self.config["auto_open_webui"] = self.auto_open_webui_var.get()

        self.config["gpu_selection_mode"] = self.gpu_selection_mode_var.get()
        # self.config["selected_gpu_index"] is updated by _gpu_id_selection_changed
        self.config["override_vram_budget"] = self.override_vram_var.get()
        if hasattr(self, 'manual_vram_entry') and self.manual_vram_entry.winfo_exists():
            try:
                manual_vram_val = self.manual_vram_entry.get()
                self.config["manual_vram_total_mb"] = int(manual_vram_val) if manual_vram_val and manual_vram_val.isdigit() else 8192
            except ValueError:
                self.config["manual_vram_total_mb"] = 8192
                self.log_to_console("Warning: Invalid manual VRAM total, defaulted to 8192MB.")
        else:
            self.config["manual_vram_total_mb"] = 8192 # Should not happen if UI is consistent

        current_global_defaults_in_config = self.config.get("default_args", {}).copy()
        for arg_definition in koboldcpp_core.KOBOLDCPP_ARG_DEFINITIONS:
            param_key = arg_definition["key"]
            if param_key not in self.settings_widgets: continue
            widget_info = self.settings_widgets[param_key]; widget = widget_info["widget"]; type_hint = arg_definition.get("type_hint", "str")
            new_value = None
            if type_hint in ["bool", "bool_flag"]:
                if hasattr(widget, 'variable'): new_value = widget.variable.get()
                elif isinstance(widget, ctk.CTkCheckBox): new_value = (widget.get() == 1)
            elif isinstance(widget, ctk.CTkEntry):
                new_value_str = widget.get().strip()
                core_template_val = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].get(param_key)
                if isinstance(core_template_val, int) and new_value_str.lower() != "auto":
                    try: new_value = int(new_value_str)
                    except ValueError: new_value = new_value_str
                elif isinstance(core_template_val, float) and new_value_str.lower() != "auto":
                    try: new_value = float(new_value_str)
                    except ValueError: new_value = new_value_str
                else: new_value = new_value_str
            if new_value is not None: current_global_defaults_in_config[param_key] = new_value
        self.config["default_args"] = current_global_defaults_in_config
        if "model_specific_args" not in self.config: self.config["model_specific_args"] = {}

        success, message = koboldcpp_core.save_launcher_config(self.config)
        if success:
            self.log_to_console(message)
            self.koboldcpp_executable = self.config["koboldcpp_executable"]
            self.default_model_dir = self.config.get("default_gguf_dir", "")
            self.db_path = self.config.get("db_file", self.db_path)
            self.settings_dirty = False
            self.update_save_button_state()
        else:
            self.log_to_console(f"Error saving configuration: {message}")
            messagebox.showerror("Configuration Save Error", f"Could not save configuration:\n{message}", parent=self)
        return success

    def monitor_vram(self):
        while True:
            try:
                selected_mode_from_cfg = self.config.get("gpu_selection_mode", "auto")
                selected_idx_from_cfg = self.config.get("selected_gpu_index", 0)
                
                target_type_for_query = selected_mode_from_cfg if selected_mode_from_cfg != "auto" else None
                target_idx_for_query = selected_idx_from_cfg
                                
                # get_available_vram_mb returns (budgeted_free, budgeted_total, message, rich_gpu_info_dict)
                _, _, _, gpu_info_dict_core = koboldcpp_core.get_available_vram_mb(
                    current_config=self.config, 
                    target_gpu_type=target_type_for_query, 
                    target_gpu_index=target_idx_for_query 
                )
                self.gpu_info = gpu_info_dict_core # Store the rich dict

                # Values for display should be based on the budget if override is active
                total_mb_for_display = float(gpu_info_dict_core.get("total_mb_budgeted", 0.0))
                free_mb_for_display = float(gpu_info_dict_core.get("free_mb_budgeted", 0.0))
                
                used_mb_for_display = 0.0
                if total_mb_for_display > 0:
                     # If override is active, used_percent_budgeted is (actual_hw_used / manual_total_budget)
                     # If not, used_percent_budgeted is same as used_percent (actual_hw_used / actual_hw_total)
                    used_mb_for_display = total_mb_for_display - free_mb_for_display
                
                used_mb_for_display = max(0.0, min(used_mb_for_display, total_mb_for_display if total_mb_for_display > 0 else 0.0))
                final_message_for_display = gpu_info_dict_core.get("message", "N/A")
                
                if hasattr(self, 'after') and self.winfo_exists():
                    self.after(0, lambda u=used_mb_for_display, t=total_mb_for_display, msg=final_message_for_display: self.update_vram_display(u, t, msg))
                else: 
                    break
            except Exception as e_vram_mon:
                print(f"Error in VRAM monitor thread: {e_vram_mon}")
                traceback.print_exc()
                if hasattr(self, 'after') and self.winfo_exists():
                    self.after(0, lambda: self.update_vram_display(0,0, "VRAM monitor error"))
                else: 
                    break
            time.sleep(5) 

    def update_vram_display(self, used_mb: float, total_mb: float, message_from_core: str = ""):
        # This 'used_mb' and 'total_mb' are already budget-aware from monitor_vram/refresh_vram callers
        if hasattr(self, 'vram_progress') and self.vram_progress.winfo_exists():
            if total_mb > 0:
                percentage = min(1.0, max(0.0, used_mb / total_mb)) 
                self.vram_progress.set(percentage)
                progress_color = "#28a745"; 
                if percentage > 0.9: progress_color = "#dc3545" 
                elif percentage > 0.7: progress_color = "#ffc107" 
                self.vram_progress.configure(progress_color=progress_color)
            else:
                self.vram_progress.set(0) 
                self.vram_progress.configure(progress_color=ctk.ThemeManager.theme["CTkProgressBar"]["progress_color"]) 
            if hasattr(self, 'vram_text') and self.vram_text.winfo_exists():
                self.vram_text.configure(text=f"{message_from_core if message_from_core else 'VRAM N/A'}")

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
                self.tuning_view_vram_text.configure(text=f"{message_from_core if message_from_core else 'VRAM N/A'}")

    def refresh_vram(self):
        self.log_to_console("User requested VRAM refresh...")
        try:
            selected_mode_from_cfg = self.config.get("gpu_selection_mode", "auto")
            selected_idx_from_cfg = self.config.get("selected_gpu_index", 0)
            
            target_type_for_query = selected_mode_from_cfg if selected_mode_from_cfg != "auto" else None
            target_idx_for_query = selected_idx_from_cfg

            _, _, _, gpu_info_dict_core = koboldcpp_core.get_available_vram_mb(
                current_config=self.config,
                target_gpu_type=target_type_for_query,
                target_gpu_index=target_idx_for_query
            )
            self.gpu_info = gpu_info_dict_core # Store rich dict

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
            self.update_vram_display(0,0,"Error during VRAM refresh")

    def populate_model_specifics_list_display(self):
        if not hasattr(self, 'model_specifics_list_frame') or not self.model_specifics_list_frame.winfo_exists():
            return
        for widget in self.model_specifics_list_frame.winfo_children(): widget.destroy()
        model_configs = self.config.get("model_specific_args", {})
        if not model_configs:
            ctk.CTkLabel(self.model_specifics_list_frame, text="No model-specific configurations saved.").pack(pady=10, padx=10)
            return
        sorted_model_paths = sorted(model_configs.keys())
        for model_path in sorted_model_paths:
            item_frame = ctk.CTkFrame(self.model_specifics_list_frame); item_frame.pack(fill="x", pady=(3,0), padx=2)
            model_display_name = os.path.basename(model_path)
            if len(model_display_name) > 55: model_display_name = model_display_name[:26] + "..." + model_display_name[-26:]
            ctk.CTkLabel(item_frame, text=model_display_name, anchor="w", wraplength=400).pack(side="left", padx=(5,10), pady=5, fill="x", expand=True)
            edit_button = ctk.CTkButton(item_frame, text="Edit", width=50, height=24, command=lambda mp=model_path: self.open_model_specific_edit_dialog(mp))
            edit_button.pack(side="left", padx=(0,5), pady=5)
            ToolTip(edit_button, f"Edit specific settings for\n{os.path.basename(model_path)}")
            delete_button = ctk.CTkButton(item_frame, text="Delete", width=50, height=24, fg_color="firebrick", hover_color="darkred", command=lambda mp=model_path: self.delete_single_model_specific_config(mp))
            delete_button.pack(side="left", padx=(0,5), pady=5)
            ToolTip(delete_button, f"Delete specific settings for\n{os.path.basename(model_path)}")

    def open_model_specific_edit_dialog(self, model_path_to_edit):
        if not model_path_to_edit: return
        dialog = ctk.CTkToplevel(self); dialog.title(f"Edit Specific Args: {os.path.basename(model_path_to_edit)}"); dialog.geometry("800x700"); dialog.transient(self); dialog.grab_set(); dialog.attributes("-topmost", True)
        param_defs = self._get_param_definitions_for_dialog()
        args_for_display = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
        args_for_display.update(self.config.get("default_args", {}))
        args_for_display.update(self.config.get("model_specific_args", {}).get(model_path_to_edit, {}))
        main_dialog_frame = ctk.CTkFrame(dialog); main_dialog_frame.pack(fill="both", expand=True); main_dialog_frame.grid_columnconfigure(0, weight=1); main_dialog_frame.grid_rowconfigure(0, weight=1)
        content_frame, widgets_info_dialog = self._create_args_dialog_content_revised(main_dialog_frame, args_for_display, param_defs)
        content_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        def save_model_specifics_action():
            model_specifics_to_save = self.config["model_specific_args"].get(model_path_to_edit, {}).copy()
            global_baseline_args = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
            global_baseline_args.update(self.config.get("default_args", {}))
            config_changed_for_this_model = False
            for param_key, info in widgets_info_dialog.items():
                widget = info["widget"]; type_hint = info["type_hint"]; baseline_val_for_compare = global_baseline_args.get(param_key)
                current_widget_val = None
                if type_hint in ["bool", "bool_flag"]:
                    current_widget_val = widget.variable.get()
                    norm_baseline_bool = False
                    if isinstance(baseline_val_for_compare, bool): norm_baseline_bool = baseline_val_for_compare
                    elif isinstance(baseline_val_for_compare, str): norm_baseline_bool = baseline_val_for_compare.lower() == 'true'
                    baseline_val_for_compare = norm_baseline_bool
                else:
                    current_widget_val_str = widget.get().strip()
                    if not current_widget_val_str: # If field is cleared
                        if param_key in model_specifics_to_save: # And it was previously set for this model
                            del model_specifics_to_save[param_key]
                            config_changed_for_this_model = True
                        continue # Move to next param
                    else: # Field has a value
                        current_widget_val = current_widget_val_str
                        # Try to cast to baseline type for comparison, if baseline is int/float
                        if isinstance(baseline_val_for_compare, (int, float)) and current_widget_val_str.lower() != "auto":
                            try: current_widget_val = type(baseline_val_for_compare)(current_widget_val_str)
                            except ValueError: pass # Keep as string if cast fails
                        baseline_val_for_compare = str(baseline_val_for_compare) if baseline_val_for_compare is not None else ""

                # Compare current widget value to the global baseline
                if str(current_widget_val) != str(baseline_val_for_compare): # If different from global baseline
                    if model_specifics_to_save.get(param_key) != current_widget_val: # And different from current model-specific value (or not set)
                        model_specifics_to_save[param_key] = current_widget_val
                        config_changed_for_this_model = True
                elif param_key in model_specifics_to_save: # If same as global baseline, but was set for model
                    del model_specifics_to_save[param_key] # Remove it from model-specific
                    config_changed_for_this_model = True

            if config_changed_for_this_model:
                if model_specifics_to_save: self.config["model_specific_args"][model_path_to_edit] = model_specifics_to_save
                elif model_path_to_edit in self.config["model_specific_args"]: del self.config["model_specific_args"][model_path_to_edit]
                if self.save_config():
                    self.log_to_console(f"Model-specific args updated for {os.path.basename(model_path_to_edit)}")
                    self.populate_model_specifics_list_display()
                    if self.tuning_in_progress and self.current_tuning_model_path == model_path_to_edit: self._reinitialize_session_base_args(); self.update_tuning_display()
                else: self.log_to_console("Failed to save config after editing model-specific args.")
            dialog.destroy()
        button_frame_dialog = ctk.CTkFrame(main_dialog_frame); button_frame_dialog.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        ctk.CTkButton(button_frame_dialog, text="Save Specifics for This Model", command=save_model_specifics_action).pack(side="left", padx=10)
        ctk.CTkButton(button_frame_dialog, text="Cancel", command=dialog.destroy).pack(side="right", padx=10)
        dialog.wait_window()

    def _create_args_dialog_content_revised(self, parent_frame_for_scrollable, current_args_to_display, param_definitions_list):
        scrollable_content_frame = ctk.CTkScrollableFrame(parent_frame_for_scrollable, label_text="Model Arguments")
        widgets_information = {}
        for setting_def in param_definitions_list:
            param_key = setting_def["key"]
            if param_key == "--model": continue
            current_value = current_args_to_display.get(param_key); type_hint = setting_def.get("type_hint", "str")
            row_frame = ctk.CTkFrame(scrollable_content_frame); row_frame.pack(fill="x", pady=2, padx=2)
            ctk.CTkLabel(row_frame, text=f"{setting_def['name']}:", width=180, anchor="w").pack(side="left", padx=(5,0))
            widget_for_param = None
            if type_hint in ["bool", "bool_flag"]:
                bool_value_for_widget = False
                if isinstance(current_value, bool): bool_value_for_widget = current_value
                elif isinstance(current_value, str): bool_value_for_widget = current_value.lower() == 'true'
                var = ctk.BooleanVar(value=bool_value_for_widget)
                widget_for_param = ctk.CTkCheckBox(row_frame, text="", variable=var, width=20); widget_for_param.pack(side="left", padx=(0,5)); widget_for_param.variable = var
            else:
                widget_for_param = ctk.CTkEntry(row_frame, width=150)
                if current_value is not None: widget_for_param.insert(0, str(current_value))
                widget_for_param.pack(side="left", padx=(0,5))
            help_text = setting_def.get("help", "")
            if help_text: ctk.CTkLabel(row_frame, text=help_text, font=ctk.CTkFont(size=10), text_color="gray", anchor="w", justify="left", wraplength=350).pack(side="left", padx=5, fill="x", expand=True)
            widgets_information[param_key] = {"widget": widget_for_param, "type_hint": type_hint}
        return scrollable_content_frame, widgets_information

    def delete_single_model_specific_config(self, model_path_to_delete):
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the specific configuration for:\n{os.path.basename(model_path_to_delete)}?", parent=self, icon=messagebox.WARNING):
            if model_path_to_delete in self.config.get("model_specific_args", {}):
                del self.config["model_specific_args"][model_path_to_delete]
                if self.save_config():
                    self.log_to_console(f"Deleted specific config for {os.path.basename(model_path_to_delete)}")
                    self.populate_model_specifics_list_display()
                    if self.tuning_in_progress and self.current_tuning_model_path == model_path_to_delete: self._reinitialize_session_base_args(); self.update_tuning_display()
                else: self.log_to_console("Failed to save config after deleting model specific.")
            else: self.log_to_console(f"No specific config found to delete for {os.path.basename(model_path_to_delete)}")

    def export_gui_config(self):
        if self.settings_dirty:
            if not messagebox.askyesno("Unsaved Settings", "You have unsaved changes in the Settings tab. Export the currently saved configuration (ignoring unsaved changes)?", detail="Choose 'Yes' to export last saved config, 'No' to cancel export.", parent=self, icon=messagebox.QUESTION):
                self.log_to_console("Export cancelled due to unsaved settings."); return
        export_filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Config files", "*.json"), ("All files", "*.*")], title="Export Launcher Configuration to...", initialdir=Path.home(), parent=self)
        if export_filepath:
            success, message = koboldcpp_core.export_config_to_file(self.config.copy(), export_filepath)
            if success: self.log_to_console(message); messagebox.showinfo("Export Successful", message, parent=self)
            else: self.log_to_console(f"Export failed: {message}"); messagebox.showerror("Export Error", f"Could not export configuration:\n{message}", parent=self)

    def import_gui_config(self):
        import_filepath = filedialog.askopenfilename(filetypes=[("JSON Config files", "*.json"), ("All files", "*.*")], title="Import Launcher Configuration from...", initialdir=Path.home(), parent=self)
        if import_filepath:
            imported_config_data, message = koboldcpp_core.import_config_from_file(import_filepath)
            if imported_config_data:
                if messagebox.askyesno("Confirm Import", "This will OVERWRITE your current launcher settings with the content of the selected file.\nA backup of your current settings will be attempted.\n\nDo you want to proceed with the import?", icon=messagebox.WARNING, parent=self):
                    try:
                        current_config_file_path = koboldcpp_core.CONFIG_FILE
                        if os.path.exists(current_config_file_path):
                            backup_path = current_config_file_path + f".backup_before_import_{time.strftime('%Y%m%d-%H%M%S')}.json"
                            shutil.copy2(current_config_file_path, backup_path)
                            self.log_to_console(f"Current configuration backed up to: {backup_path}")
                        save_success, save_message = koboldcpp_core.save_launcher_config(imported_config_data)
                        if not save_success:
                            self.log_to_console(f"Failed to save imported config to primary location: {save_message}"); messagebox.showerror("Import Error", f"Failed to save imported configuration: {save_message}", parent=self); return
                        self.log_to_console("Configuration data imported. Re-initializing launcher core and GUI...")
                        core_init_results = koboldcpp_core.initialize_launcher(); self.config = core_init_results["config"]; self.system_info = core_init_results["system_info"]; self.gpu_info = core_init_results["gpu_info"]; self.koboldcpp_capabilities = core_init_results.get("koboldcpp_capabilities", {}); self.db_path = self.config["db_file"]; self.default_model_dir = self.config.get("default_gguf_dir", os.getcwd()); self.koboldcpp_executable = self.config.get("koboldcpp_executable")
                        self.load_settings_from_config(); self.check_koboldcpp_executable()
                        if hasattr(self, 'populate_model_specifics_list_display'): self.populate_model_specifics_list_display()
                        if hasattr(self, 'update_kcpp_capabilities_display'): self.update_kcpp_capabilities_display(re_detect=True)
                        self.after(100, self._populate_gpu_id_dropdown_on_startup) 
                        self.refresh_vram(); self.load_history()
                        if self.tuning_in_progress: self.end_tuning_session(switch_to_model_selection=True)
                        else: self._show_model_selection_view()
                        self.settings_dirty = False; self.update_save_button_state()
                        self.log_to_console("Launcher re-initialized with imported settings.")
                        messagebox.showinfo("Import Successful", f"Configuration successfully imported from:\n{os.path.basename(import_filepath)}\n\nLauncher has been re-initialized.", parent=self)
                    except Exception as e_apply:
                        error_msg_apply = f"A critical error occurred while applying the imported configuration: {e_apply}\n{traceback.format_exc()}"; self.log_to_console(error_msg_apply); messagebox.showerror("Import Application Error", error_msg_apply, parent=self)
            else: self.log_to_console(f"Import failed: {message}"); messagebox.showerror("Import Error", f"Could not import configuration:\n{message}", parent=self)

    def update_kcpp_capabilities_display(self, re_detect=True):
        if not hasattr(self, 'kcpp_caps_text_widget') or not self.kcpp_caps_text_widget.winfo_exists(): return
        if re_detect:
            self.log_to_console("Attempting to re-detect KoboldCpp capabilities...")
            if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
                self.koboldcpp_executable = self.exe_path_entry.get().strip(); self.config["koboldcpp_executable"] = self.koboldcpp_executable
            self.check_koboldcpp_executable()
            if not self.koboldcpp_executable or not (os.path.exists(self.koboldcpp_executable) or shutil.which(self.koboldcpp_executable)):
                self.log_to_console("Cannot re-detect capabilities: KoboldCpp executable path is not valid or found.")
                self.koboldcpp_capabilities = {"error": "KoboldCpp executable path is not valid or not found."}
            else: self.koboldcpp_capabilities = koboldcpp_core.detect_koboldcpp_capabilities(self.koboldcpp_executable)
            if "error" not in self.koboldcpp_capabilities: self.log_to_console("KoboldCpp capabilities re-detected successfully."); self._update_gpu_backend_flags_in_config()
            else: self.log_to_console(f"Error re-detecting KCPP capabilities: {self.koboldcpp_capabilities['error']}")
        caps_str_parts = []
        if "error" in self.koboldcpp_capabilities: caps_str_parts.append(f"Error detecting capabilities:\n {self.koboldcpp_capabilities['error']}")
        else:
            display_order = ["cuda", "rocm", "opencl", "vulkan", "flash_attn", "auto_quantkv", "overridetensors"]
            friendly_names = {"cuda": "CUDA (cuBLAS)", "rocm": "ROCm (hipBLAS/rocBLAS)", "opencl": "OpenCL (CLBlast)", "vulkan": "Vulkan Backend", "flash_attn": "FlashAttention Support", "auto_quantkv": "Auto QuantKV Option", "overridetensors": "Tensor Override Support"}
            for key in display_order:
                if key in self.koboldcpp_capabilities: status = "Yes" if self.koboldcpp_capabilities.get(key) else "No"; caps_str_parts.append(f"{friendly_names.get(key, key.capitalize())}: {status}")
            for key, value in self.koboldcpp_capabilities.items():
                if key not in display_order and key not in ["error", "available_args"]: status = "Yes" if value else "No"; caps_str_parts.append(f"{key.replace('_',' ').capitalize()}: {status}")
        caps_display_str = "\n".join(caps_str_parts) if caps_str_parts else "No capabilities data available or N/A."
        self.kcpp_caps_text_widget.configure(state="normal"); self.kcpp_caps_text_widget.delete("1.0", "end"); self.kcpp_caps_text_widget.insert("1.0", caps_display_str); self.kcpp_caps_text_widget.configure(state="disabled")

    def _update_gpu_backend_flags_in_config(self):
        if "error" in self.koboldcpp_capabilities: return
        if "default_args" not in self.config: self.config["default_args"] = {}
        original_cublas = self.config["default_args"].get("--usecublas", False)
        original_hipblas = self.config["default_args"].get("--usehipblas", False)
        self.config["default_args"]["--usecublas"] = False; self.config["default_args"]["--usehipblas"] = False
        current_gpu_details = self.gpu_info # self.gpu_info is the rich dict from core
        new_cublas, new_hipblas = False, False
        if current_gpu_details and current_gpu_details.get("type") == "NVIDIA" and self.config.get("gpu_detection",{}).get("nvidia",True):
            if self.koboldcpp_capabilities.get("cuda"): new_cublas = True
        elif current_gpu_details and current_gpu_details.get("type") == "AMD" and self.config.get("gpu_detection",{}).get("amd",True):
            if self.koboldcpp_capabilities.get("rocm"): new_hipblas = True
        if new_cublas != original_cublas or new_hipblas != original_hipblas:
            self.config["default_args"]["--usecublas"] = new_cublas; self.config["default_args"]["--usehipblas"] = new_hipblas
            self.log_to_console(f"Auto-updated GPU backend flags: CUBLAS={new_cublas}, HIPBLAS={new_hipblas}. Refreshing settings UI.")
            self.load_settings_from_config(); self.mark_settings_dirty()

    def start_tuning_session(self):
        if not self.current_model_path: messagebox.showwarning("No Model Selected", "Please select a GGUF model first.", parent=self); return
        if not self.model_analysis_info or 'filepath' not in self.model_analysis_info:
            self.log_to_console("Model analysis info is missing. Re-analyzing..."); self.analyze_model_action(self.current_model_path)
            if not self.model_analysis_info or 'filepath' not in self.model_analysis_info: messagebox.showerror("Model Error", "Failed to analyze model. Cannot start tuning.", parent=self); return
        self.log_to_console(f"Starting tuning session for: {os.path.basename(self.current_model_path)}"); self.tuning_in_progress = True; self.current_tuning_model_path = self.current_model_path; self.current_tuning_model_analysis = self.model_analysis_info.copy(); self._reinitialize_session_base_args(); self.last_successful_monitored_run_details_gui = None
        if hasattr(self, 'tuning_model_name_label') and self.tuning_model_name_label.winfo_exists(): self.tuning_model_name_label.configure(text=f"Model: {os.path.basename(self.current_tuning_model_path)}")
        
        # Get current VRAM info (budgeted values if override is on)
        current_vram_budgeted, _, _, current_gpu_full_info = koboldcpp_core.get_available_vram_mb(
            self.config, 
            target_gpu_type=self.config.get("gpu_selection_mode", "auto") if self.config.get("gpu_selection_mode", "auto") != "auto" else None, 
            target_gpu_index=self.config.get("selected_gpu_index", 0)
        )
        # For history lookup and heuristics, we might prefer actual hardware VRAM if available and not overridden
        # For find_best_historical_config, we pass current_available_dedicated_vram_mb
        # This should be the *actual* hardware VRAM available, not the budgeted amount,
        # because historical records store vram_at_launch_decision_mb as actual.
        current_actual_hw_vram_mb = current_gpu_full_info.get("free_mb", 0.0)


        is_moe = self.current_tuning_model_analysis.get('is_moe', False); estimated_vram_needed_gb = self.current_tuning_model_analysis.get('estimated_vram_gb_full_gpu', 0); estimated_vram_needed_mb = float(estimated_vram_needed_gb) * 1024 if estimated_vram_needed_gb else 0.0
        if is_moe: self.current_tuning_min_level, self.current_tuning_max_level, initial_heuristic_level = -25, 10, -10
        else:
            self.current_tuning_min_level, self.current_tuning_max_level = -17, 9
            size_b_val = self.current_tuning_model_analysis.get('size_b', 0); size_b = 0.0
            if isinstance(size_b_val, (int, float)): size_b = float(size_b_val)
            elif isinstance(size_b_val, str):
                try: size_b = float(size_b_val)
                except ValueError: size_b = 0.0
            if size_b >= 30: initial_heuristic_level = -3
            elif size_b >= 20: initial_heuristic_level = -5
            else: initial_heuristic_level = -7
        
        safety_buffer_mb = float(self.config.get("vram_safety_buffer_mb", 768)); 
        min_free_after_load_mb = float(self.config.get("min_vram_free_after_load_success_mb", 512)); 
        
        # Effective VRAM budget for heuristic is based on the budgeted total, not raw hardware total
        effective_vram_budget_for_heuristic_mb = current_gpu_full_info.get("total_mb_budgeted", 0.0) - safety_buffer_mb - min_free_after_load_mb
        
        if estimated_vram_needed_mb > 0 and current_vram_budgeted > 0 : # current_vram_budgeted is free budgeted
            if estimated_vram_needed_mb > effective_vram_budget_for_heuristic_mb * 1.1: 
                initial_heuristic_level = max(initial_heuristic_level, -3 if not is_moe else -6)
                self.log_to_console(f"Heuristic: Est. VRAM ({estimated_vram_needed_mb:.0f}MB) > effective budget ({effective_vram_budget_for_heuristic_mb:.0f}MB). Adjusting OT towards CPU.")
            elif estimated_vram_needed_mb < effective_vram_budget_for_heuristic_mb * 0.7: 
                initial_heuristic_level = min(initial_heuristic_level, -12 if not is_moe else -18)
                self.log_to_console(f"Heuristic: Ample VRAM budget. Adjusting OT towards GPU.")
        
        # Use actual hardware free VRAM for historical lookup comparison
        best_hist_config = koboldcpp_core.find_best_historical_config(self.db_path, self.current_tuning_model_analysis, current_actual_hw_vram_mb, self.config)

        if best_hist_config and "attempt_level" in best_hist_config:
            self.log_to_console(f"Found historical config. Level: {best_hist_config['attempt_level']}, Outcome: {best_hist_config['outcome']}"); hist_level, hist_outcome = best_hist_config['attempt_level'], best_hist_config.get('outcome', ""); approx_hist_vram_used = best_hist_config.get('approx_vram_used_kcpp_mb')
            # Compare historical usage against current *actual* hardware VRAM
            if approx_hist_vram_used is not None and (float(approx_hist_vram_used) + safety_buffer_mb < current_actual_hw_vram_mb): 
                initial_heuristic_level = max(self.current_tuning_min_level, hist_level -1 if hist_level > self.current_tuning_min_level else hist_level)
                self.log_to_console(f"Historical success fits actual VRAM. Starting near: {initial_heuristic_level}")
            elif hist_outcome.startswith("SUCCESS_LOAD_VRAM_OK") or hist_outcome.startswith("SUCCESS_USER_CONFIRMED") or hist_outcome.endswith("_USER_SAVED_GOOD_GPU_GUI"): 
                initial_heuristic_level = max(self.current_tuning_min_level, hist_level -1 if hist_level > self.current_tuning_min_level else hist_level)
            elif hist_outcome.endswith("_USER_AUTO_ADJUST_CPU_GUI") or hist_outcome.endswith("_USER_TRIED_CPU_FAIL_GUI") or "OOM" in hist_outcome.upper() or "TIGHT" in hist_outcome.upper(): 
                initial_heuristic_level = min(self.current_tuning_max_level, hist_level + 1 if hist_level < self.current_tuning_max_level else hist_level)
            else: initial_heuristic_level = hist_level
            remembered_args_list = best_hist_config.get("args_list", []);
            if remembered_args_list: remembered_args_dict = koboldcpp_core.args_list_to_dict(remembered_args_list); remembered_args_dict.pop("--model", None); remembered_args_dict.pop("--overridetensors", None); self.current_tuning_session_base_args.update(remembered_args_dict); self.log_to_console(f"Applied remembered arguments to current session base. OT Level target adjusted to: {initial_heuristic_level}")
        else: self.log_to_console(f"No suitable historical config found. Starting with heuristic OT Level: {initial_heuristic_level}")
        self.current_tuning_attempt_level = max(self.current_tuning_min_level, min(initial_heuristic_level, self.current_tuning_max_level)); self.level_of_last_monitored_run = self.current_tuning_attempt_level
        self._show_tuning_mode_view(); self.update_tuning_display()
        if hasattr(self, 'kcpp_live_output_text') and self.kcpp_live_output_text.winfo_exists(): self.kcpp_live_output_text.configure(state="normal"); self.kcpp_live_output_text.delete("1.0", "end"); self.kcpp_live_output_text.configure(state="disabled")

    def update_tuning_display(self):
        if not self.tuning_in_progress: return
        self.current_tuning_attempt_level = max(self.current_tuning_min_level, min(self.current_tuning_attempt_level, self.current_tuning_max_level))
        if hasattr(self, 'tuning_model_name_label') and self.tuning_model_name_label.winfo_exists() and self.current_tuning_model_path: self.tuning_model_name_label.configure(text=f"Model: {os.path.basename(self.current_tuning_model_path)}")
        if hasattr(self, 'tuning_last_run_info_label') and self.tuning_last_run_info_label.winfo_exists():
            if self.last_successful_monitored_run_details_gui: level = self.last_successful_monitored_run_details_gui.get("level", "N/A"); outcome = self.last_successful_monitored_run_details_gui.get("outcome", "N/A"); vram_used_last = self.last_successful_monitored_run_details_gui.get("vram_used_mb", "N/A"); self.tuning_last_run_info_label.configure(text=f"Last Monitored Success: Level {level}, Outcome: {outcome}, Approx Actual VRAM Used: {vram_used_last}MB", anchor="w")
            else: self.tuning_last_run_info_label.configure(text="Last Monitored Success: None yet in this session.", anchor="w")
        ot_string = koboldcpp_core.generate_overridetensors(self.current_tuning_model_analysis, self.current_tuning_attempt_level); description = koboldcpp_core.get_offload_description(self.current_tuning_model_analysis, self.current_tuning_attempt_level, ot_string); gpu_layers = koboldcpp_core.get_gpu_layers_for_level(self.current_tuning_model_analysis, self.current_tuning_attempt_level); total_layers = self.current_tuning_model_analysis.get('num_layers', 32)
        if hasattr(self, 'tuning_ot_level_label'): self.tuning_ot_level_label.configure(text=f"Level: {self.current_tuning_attempt_level}")
        range_text_suffix = 'SuperMaxCPU' if self.current_tuning_model_analysis.get('is_moe') else 'MaxCPU'; range_text = f"{self.current_tuning_min_level}=MaxGPU ... {self.current_tuning_max_level}={range_text_suffix}"
        if hasattr(self, 'tuning_ot_range_label'): self.tuning_ot_range_label.configure(text=f"Range: {range_text}")
        if hasattr(self, 'tuning_ot_strategy_desc_label'): self.tuning_ot_strategy_desc_label.configure(text=f"Strategy: {description}")
        regex_display = ot_string if ot_string else "None (Max GPU layers if --gpulayers is high)"
        if hasattr(self, 'tuning_ot_regex_label'): self.tuning_ot_regex_label.configure(text=f"Regex: {regex_display}")
        if hasattr(self, 'tuning_gpu_layers_label'): self.tuning_gpu_layers_label.configure(text=f"Effective GPU Layers: {gpu_layers}/{total_layers}")
        args_for_kcpp_list = koboldcpp_core.build_command(self.current_tuning_model_path, ot_string, self.current_tuning_model_analysis, self.current_tuning_session_base_args); full_command_list = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, args_for_kcpp_list); display_command_str = koboldcpp_core.format_command_for_display(full_command_list)
        if hasattr(self, 'tuning_proposed_command_text'): self.tuning_proposed_command_text.configure(state="normal"); self.tuning_proposed_command_text.delete("1.0", "end"); self.tuning_proposed_command_text.insert("1.0", display_command_str); self.tuning_proposed_command_text.configure(state="disabled")
        if hasattr(self, 'btn_tune_more_gpu'): self.btn_tune_more_gpu.configure(state="normal" if self.current_tuning_attempt_level > self.current_tuning_min_level else "disabled")
        if hasattr(self, 'btn_tune_more_cpu'): self.btn_tune_more_cpu.configure(state="normal" if self.current_tuning_attempt_level < self.current_tuning_max_level else "disabled")
        if hasattr(self, 'btn_tune_edit_model_perm_args'): self.btn_tune_edit_model_perm_args.configure(state="normal" if self.current_tuning_model_path else "disabled")

    def _handle_monitoring_completion(self, initial_outcome_key: str):
        self.log_to_console(f"Monitoring completed. Initial Outcome: {initial_outcome_key}"); self._log_to_kcpp_live_output(f"\n--- Monitoring Finished: {initial_outcome_key} ---\n")
        if initial_outcome_key in ["TIMEOUT_NO_SIGNAL_GUI", "OOM_CRASH_DETECTED_GUI", "PREMATURE_EXIT_GUI"] or "OOM" in initial_outcome_key.upper() or "CRASH" in initial_outcome_key.upper():
            if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None: self.log_to_console("Terminating KCPP process due to unfavorable outcome..."); koboldcpp_core.kill_process(self.kcpp_process_obj.pid, force=True)
            self.kcpp_process_obj = None
        
        self.last_approx_vram_used_kcpp_mb = None
        self.last_free_vram_after_load_mb = None # This will store budgeted free VRAM
        final_db_outcome = initial_outcome_key
        
        if initial_outcome_key == "SUCCESS_LOAD_DETECTED_GUI":
            self._log_to_kcpp_live_output("API detected. Waiting a few seconds for VRAM to stabilize...\n"); stabilization_wait_s = float(self.config.get("vram_stabilization_wait_s", 3.0)); time.sleep(max(2.0, stabilization_wait_s))
            
            selected_gpu_type_for_vram = self.config.get("gpu_selection_mode", "auto")
            target_type = selected_gpu_type_for_vram if selected_gpu_type_for_vram != "auto" else None
            target_idx = self.config.get("selected_gpu_index", 0)
            # Get full GPU info after load
            _, _, _, gpu_info_after_load = koboldcpp_core.get_available_vram_mb(self.config, target_type, target_idx)

            self.last_free_vram_after_load_mb = gpu_info_after_load.get("free_mb_budgeted") # Budgeted free VRAM
            
            # Calculate actual KCPP VRAM usage using raw hardware values
            actual_hw_free_vram_after_load = gpu_info_after_load.get("free_mb") # Raw HW free
            actual_hw_total_vram = gpu_info_after_load.get("total_mb")     # Raw HW total

            if actual_hw_total_vram > 0 and self.vram_at_decision_for_db is not None and actual_hw_free_vram_after_load is not None:
                # self.vram_at_decision_for_db is actual HW free VRAM *before* KCPP launch
                vram_used_by_kcpp_hw = self.vram_at_decision_for_db - actual_hw_free_vram_after_load
                self.last_approx_vram_used_kcpp_mb = max(0, min(vram_used_by_kcpp_hw, actual_hw_total_vram))
                
                self._log_to_kcpp_live_output(
                    f"VRAM After Load (Budgeted Free): {self.last_free_vram_after_load_mb:.0f}MB. "
                    f"Approx Actual KCPP VRAM Used: {self.last_approx_vram_used_kcpp_mb:.0f}MB\n"
                )
                
                min_vram_free_target = float(self.config.get("min_vram_free_after_load_success_mb", 512))
                
                # Decision based on budgeted free VRAM
                if self.last_free_vram_after_load_mb < min_vram_free_target:
                    self._log_to_kcpp_live_output(f"WARNING: Budgeted VRAM tight! {self.last_free_vram_after_load_mb:.0f}MB free < {min_vram_free_target}MB target.\n")
                    final_db_outcome = "SUCCESS_LOAD_VRAM_TIGHT_GUI"
                else:
                    self._log_to_kcpp_live_output("Budgeted VRAM usage OK.\n")
                    final_db_outcome = "SUCCESS_LOAD_VRAM_OK_GUI"

                # Check if actual usage exceeded manual budget and add a note
                if gpu_info_after_load.get("override_active", False) and \
                   self.last_approx_vram_used_kcpp_mb > gpu_info_after_load.get("total_mb_budgeted", 0):
                    self._log_to_kcpp_live_output(
                        f"NOTE: Actual KCPP VRAM usage ({self.last_approx_vram_used_kcpp_mb:.0f}MB) "
                        f"exceeded the manual VRAM budget ({gpu_info_after_load.get('total_mb_budgeted', 0):.0f}MB).\n"
                    )
                
                self.last_successful_monitored_run_details_gui = {
                    "level": self.level_of_last_monitored_run, 
                    "outcome": final_db_outcome, 
                    "vram_used_mb": f"{self.last_approx_vram_used_kcpp_mb:.0f}" if self.last_approx_vram_used_kcpp_mb is not None else "N/A"
                }
                self.update_tuning_display()
            else: # Fallback if full VRAM info for precise usage calc wasn't available
                final_db_outcome = "SUCCESS_LOAD_NO_VRAM_CHECK_GUI"
                self.last_successful_monitored_run_details_gui = {"level": self.level_of_last_monitored_run, "outcome": final_db_outcome, "vram_used_mb": "N/A"}
                self.update_tuning_display()
        
        koboldcpp_core.save_config_to_db(self.db_path, self.current_tuning_model_path, self.current_tuning_model_analysis, 
                                        self.vram_at_decision_for_db, # This is actual HW free VRAM at launch decision
                                        self.current_command_list_for_db, self.level_of_last_monitored_run, 
                                        final_db_outcome, self.last_approx_vram_used_kcpp_mb) # This is actual KCPP usage
        self.load_history(); self._present_post_monitoring_choices(final_db_outcome)

    def _present_post_monitoring_choices(self, outcome: str):
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None: self.log_to_console("Stopping monitored KCPP instance before showing post-monitoring choices (unless user chooses to keep it).")
        frames_to_hide_names = ['tuning_actions_primary_frame', 'tuning_actions_secondary_frame', 'tuning_edit_args_buttons_frame', 'tuning_actions_navigation_frame']
        for frame_name in frames_to_hide_names:
            frame_widget = getattr(self, frame_name, None)
            if frame_widget and hasattr(frame_widget, 'winfo_exists') and frame_widget.winfo_exists(): frame_widget.grid_remove()
        if not hasattr(self, 'post_monitor_choices_frame') or not self.post_monitor_choices_frame.winfo_exists():
            self.post_monitor_choices_frame = ctk.CTkFrame(self.tuning_mode_frame); self.post_monitor_choices_frame.grid(row=6, column=0, rowspan=4, padx=10, pady=5, sticky="news"); self.post_monitor_choices_frame.grid_columnconfigure(0, weight=1)
        else: self.post_monitor_choices_frame.grid()
        for widget in self.post_monitor_choices_frame.winfo_children(): widget.destroy()
        ctk.CTkLabel(self.post_monitor_choices_frame, text=f"Outcome: {outcome}", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 2), anchor="w", padx=5)
        if self.last_free_vram_after_load_mb is not None: ctk.CTkLabel(self.post_monitor_choices_frame, text=f"Budgeted VRAM After Load: {self.last_free_vram_after_load_mb:.0f} MB free").pack(pady=1, anchor="w", padx=5)
        if self.last_approx_vram_used_kcpp_mb is not None: ctk.CTkLabel(self.post_monitor_choices_frame, text=f"Approx. Actual KCPP VRAM Used: {self.last_approx_vram_used_kcpp_mb:.0f} MB").pack(pady=1, anchor="w", padx=5)
        vram_status_message = "VRAM Status: Check KCPP Log.";
        if "SUCCESS_LOAD_VRAM_OK" in outcome: vram_status_message = "Budgeted VRAM Usage: OK"
        elif "SUCCESS_LOAD_VRAM_TIGHT" in outcome: vram_status_message = "Budgeted VRAM Usage: TIGHT"
        elif "SUCCESS_LOAD_NO_VRAM_CHECK" in outcome: vram_status_message = "Budgeted VRAM Usage: Not Checked by Launcher"
        ctk.CTkLabel(self.post_monitor_choices_frame, text=vram_status_message).pack(pady=(1, 10), anchor="w", padx=5)
        if "SUCCESS_LOAD_VRAM_OK" in outcome:
            btn_accept = ctk.CTkButton(self.post_monitor_choices_frame, text="✅ Accept & Use this KCPP instance", command=lambda: self._handle_post_monitor_action("launch_for_use", outcome)); btn_accept.pack(fill="x", pady=3, padx=5); ToolTip(btn_accept, "Keep the currently running (monitored) KoboldCpp instance and use it.\nTuning session will end.")
            btn_save_good_gpu = ctk.CTkButton(self.post_monitor_choices_frame, text="💾 Save as Good, Auto-Adjust for More GPU & Continue Tuning", command=lambda: self._handle_post_monitor_action("save_good_more_gpu", outcome)); btn_save_good_gpu.pack(fill="x", pady=3, padx=5); ToolTip(btn_save_good_gpu, "Mark this configuration as successful, try to use even more GPU,\nand continue the tuning session.")
            btn_manual_gpu = ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Manually Try More GPU (This Session) & Continue Tuning", command=lambda: self._handle_post_monitor_action("more_gpu_now", outcome)); btn_manual_gpu.pack(fill="x", pady=3, padx=5); ToolTip(btn_manual_gpu, "Manually decrease the OT level to use more GPU and continue tuning.")
            btn_manual_cpu = ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Manually Try More CPU (This Session) & Continue Tuning", command=lambda: self._handle_post_monitor_action("more_cpu_now", outcome)); btn_manual_cpu.pack(fill="x", pady=3, padx=5); ToolTip(btn_manual_cpu, "Manually increase the OT level to use more CPU and continue tuning.")
        elif "SUCCESS_LOAD_VRAM_TIGHT" in outcome:
            btn_auto_cpu_tight = ctk.CTkButton(self.post_monitor_choices_frame, text="⚠️ Auto-Adjust for More CPU & Continue Tuning", command=lambda: self._handle_post_monitor_action("auto_adjust_cpu", outcome)); btn_auto_cpu_tight.pack(fill="x", pady=3, padx=5); ToolTip(btn_auto_cpu_tight, "VRAM is tight. Automatically increase OT level (more CPU) and continue tuning.")
            btn_launch_risky = ctk.CTkButton(self.post_monitor_choices_frame, text="🚀 Launch This Config Anyway (Risky)", command=lambda: self._handle_post_monitor_action("launch_for_use_risky", outcome)); btn_launch_risky.pack(fill="x", pady=3, padx=5); ToolTip(btn_launch_risky, "Launch the current (monitored) KoboldCpp instance for use, despite tight VRAM.\nThis might lead to instability.")
        elif "OOM" in outcome or "CRASH" in outcome or "PREMATURE_EXIT" in outcome :
            btn_auto_cpu_fail = ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Auto-Adjust for More CPU & Continue Tuning", command=lambda: self._handle_post_monitor_action("more_cpu_after_fail", outcome)); btn_auto_cpu_fail.pack(fill="x", pady=3, padx=5); ToolTip(btn_auto_cpu_fail, "Launch failed (OOM/Crash). Automatically increase OT level (more CPU) and continue tuning.")
        elif "TIMEOUT" in outcome:
            ctk.CTkLabel(self.post_monitor_choices_frame, text="KCPP launch timed out without a clear success/OOM signal.").pack(pady=3, anchor="w", padx=5)
            btn_auto_cpu_timeout = ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Auto-Adjust for More CPU (Assume OOM) & Continue Tuning", command=lambda: self._handle_post_monitor_action("more_cpu_after_fail", outcome)); btn_auto_cpu_timeout.pack(fill="x", pady=3, padx=5); ToolTip(btn_auto_cpu_timeout, "Launch timed out. Assume it was an OOM or similar issue, increase OT level (more CPU), and continue tuning.")
        else: # Generic success or other unhandled outcomes
            if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None: btn_accept_generic = ctk.CTkButton(self.post_monitor_choices_frame, text="✅ Keep KCPP Running & Use", command=lambda: self._handle_post_monitor_action("launch_for_use", outcome)); btn_accept_generic.pack(fill="x", pady=3, padx=5); ToolTip(btn_accept_generic, "The monitored KCPP instance is still running. Accept and use it.")
        btn_return_menu = ctk.CTkButton(self.post_monitor_choices_frame, text="↩️ Save Outcome & Return to Tuning Menu (Manual Adjust)", command=lambda: self._handle_post_monitor_action("return_to_tuning_menu", outcome)); btn_return_menu.pack(fill="x", pady=3, padx=5); ToolTip(btn_return_menu, "Save the outcome of this monitored run and return to the main tuning strategy screen\nfor manual adjustments or to try other actions.")
        self.log_to_console("Presented post-monitoring choices to user.")

    def _return_to_full_tuning_menu(self):
        if hasattr(self, 'post_monitor_choices_frame') and self.post_monitor_choices_frame.winfo_exists(): self.post_monitor_choices_frame.grid_remove()
        frames_to_show_names = ['tuning_actions_primary_frame', 'tuning_actions_secondary_frame', 'tuning_edit_args_buttons_frame', 'tuning_actions_navigation_frame']
        for frame_name in frames_to_show_names:
            frame_widget = getattr(self, frame_name, None)
            if frame_widget and hasattr(frame_widget, 'winfo_exists') and frame_widget.winfo_exists(): frame_widget.grid()
        self._set_tuning_buttons_state("normal"); self.update_tuning_display()

    def _run_first_time_setup_if_needed(self):
        if not self.config.get("first_run_completed", False):
            self.log_to_console("Performing first-time setup...")
            dialog = FirstTimeSetupDialog(self, self.config, self.koboldcpp_executable); self.wait_window(dialog)
            if dialog.saved_config:
                self.config = dialog.saved_config; self.koboldcpp_executable = self.config["koboldcpp_executable"]; self.default_model_dir = self.config.get("default_gguf_dir", ""); self.db_path = self.config["db_file"]
                self.load_settings_from_config(); self.log_to_console("First-time setup complete. Configuration updated."); self.update_kcpp_capabilities_display(re_detect=True); self.after(100, self._populate_gpu_id_dropdown_on_startup)
            else: self.log_to_console("First-time setup cancelled or not completed."); messagebox.showwarning("Setup Incomplete", "First-time setup was not completed. Please configure the KoboldCpp executable in Settings.", parent=self)

    def _get_merged_args_for_model(self, model_path):
        merged_args = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy(); merged_args.update(self.config.get("default_args", {}))
        if model_path: merged_args.update(self.config.get("model_specific_args", {}).get(model_path, {}))
        return merged_args

    def _reinitialize_session_base_args(self):
        effective_args_base = self._get_merged_args_for_model(self.current_tuning_model_path) if self.current_tuning_model_path else {**koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"], **self.config.get("default_args", {})}
        self.current_tuning_session_base_args = {}
        for k, v_raw in effective_args_base.items():
            v = v_raw; arg_def = next((d for d in koboldcpp_core.KOBOLDCPP_ARG_DEFINITIONS if d["key"] == k), None); is_bool_type = arg_def and arg_def.get("type_hint") in ["bool", "bool_flag"]
            if is_bool_type:
                if isinstance(v_raw, bool): v = v_raw
                elif isinstance(v_raw, str): v = v_raw.lower() == 'true'
                else: v = False # Default for bool if malformed
                self.current_tuning_session_base_args[k] = v
            elif v is not None : self.current_tuning_session_base_args[k] = str(v).strip() if isinstance(v, str) else v
        model_name_display = os.path.basename(self.current_tuning_model_path) if self.current_tuning_model_path else "No Model"
        self.log_to_console(f"Session base arguments reinitialized for '{model_name_display}'.")

    def check_koboldcpp_executable(self):
        current_exe_path_from_gui = ""; config_needs_update_due_to_autocorrect = False
        if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists(): current_exe_path_from_gui = self.exe_path_entry.get().strip()
        if current_exe_path_from_gui and current_exe_path_from_gui != self.koboldcpp_executable: self.koboldcpp_executable = current_exe_path_from_gui
        current_exe_path_to_check = self.koboldcpp_executable; original_exe_path = self.koboldcpp_executable
        if not current_exe_path_to_check or not (os.path.exists(current_exe_path_to_check) and (os.access(current_exe_path_to_check, os.X_OK) or current_exe_path_to_check.lower().endswith(".py"))):
            self.log_to_console(f"Warning: KoboldCPP executable '{current_exe_path_to_check}' not found or not executable.")
            executable_basename = os.path.basename(current_exe_path_to_check) if current_exe_path_to_check else ("koboldcpp.exe" if platform.system() == "Windows" else "koboldcpp")
            found_exe_in_path = shutil.which(executable_basename)
            if found_exe_in_path and (os.access(found_exe_in_path, os.X_OK) or found_exe_in_path.lower().endswith(".py")): self.log_to_console(f"Found '{executable_basename}' in PATH: {found_exe_in_path}"); self.koboldcpp_executable = os.path.abspath(found_exe_in_path)
            elif executable_basename.lower().endswith(".py"):
                try:
                    launcher_dir = os.path.dirname(os.path.abspath(__file__)) if not getattr(sys, 'frozen', False) else os.path.dirname(sys.executable)
                    potential_relative_path = os.path.join(launcher_dir, executable_basename)
                    if os.path.exists(potential_relative_path) and (os.access(potential_relative_path, os.X_OK) or potential_relative_path.lower().endswith(".py")): self.log_to_console(f"Found '{executable_basename}' relative to launcher: {potential_relative_path}"); self.koboldcpp_executable = os.path.abspath(potential_relative_path)
                    else: self.log_to_console(f"Could not find '{executable_basename}' as .py script relative to launcher.")
                except NameError: self.log_to_console(f"Could not determine launcher script directory for '{executable_basename}'.")
            else: self.log_to_console(f"Could not find '{executable_basename}' in PATH or as relative script.")
        else: self.log_to_console(f"KoboldCPP executable '{current_exe_path_to_check}' verified.")
        if self.koboldcpp_executable != original_exe_path:
            config_needs_update_due_to_autocorrect = True
            if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists(): self.exe_path_entry.delete(0, "end"); self.exe_path_entry.insert(0, self.koboldcpp_executable)
        if self.config.get("koboldcpp_executable") != self.koboldcpp_executable:
            self.config["koboldcpp_executable"] = self.koboldcpp_executable
            if config_needs_update_due_to_autocorrect: self.log_to_console("KoboldCpp exe path auto-corrected. Saving config..."); self.save_config()
            else: self.mark_settings_dirty()

    def _show_model_selection_view(self):
        if hasattr(self, 'tuning_mode_frame') and self.tuning_mode_frame.winfo_exists(): self.tuning_mode_frame.grid_remove()
        if hasattr(self, 'model_selection_frame') and self.model_selection_frame.winfo_exists(): self.model_selection_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        else: self.setup_main_tab(); self.model_selection_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.log_to_console("Switched to Model Selection view."); self.tabview.set("Tune & Launch")

    def _show_tuning_mode_view(self):
        if hasattr(self, 'model_selection_frame') and self.model_selection_frame.winfo_exists(): self.model_selection_frame.grid_remove()
        if hasattr(self, 'tuning_mode_frame') and self.tuning_mode_frame.winfo_exists(): self.tuning_mode_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        else: self.setup_main_tab(); self.tuning_mode_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.log_to_console("Switched to Tuning Mode view."); self.tabview.set("Tune & Launch"); self._return_to_full_tuning_menu()

    def adjust_ot_level(self, delta):
        if not self.tuning_in_progress: return
        self.current_tuning_attempt_level += delta; self.update_tuning_display()

    def edit_base_args_for_tuning_session(self):
        if not self.tuning_in_progress: messagebox.showwarning("Not Tuning", "Tuning session is not active.", parent=self); return
        if not self.current_tuning_model_path: messagebox.showerror("Error", "No model associated with current tuning session.", parent=self); return
        dialog = ctk.CTkToplevel(self); dialog.title("Edit Base Args (Current Tuning Session)"); dialog.geometry("800x700"); dialog.transient(self); dialog.grab_set(); dialog.attributes("-topmost", True)
        param_defs = self._get_param_definitions_for_dialog(); effective_base_for_session = self._get_merged_args_for_model(self.current_tuning_model_path); current_display_args = effective_base_for_session.copy(); current_display_args.update(self.current_tuning_session_base_args)
        dialog_main_frame = ctk.CTkFrame(dialog); dialog_main_frame.pack(fill="both", expand=True); dialog_main_frame.grid_columnconfigure(0, weight=1); dialog_main_frame.grid_rowconfigure(0, weight=1)
        content_frame, widgets_info = self._create_args_dialog_content_revised(dialog_main_frame, current_display_args, param_defs); content_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        def save_session_args_action():
            changes_made_to_session = False
            for param_key, info in widgets_info.items():
                widget = info["widget"]; type_hint = info["type_hint"]; new_widget_value = None
                if type_hint in ["bool", "bool_flag"]: new_widget_value = widget.variable.get()
                else:
                    new_widget_value_str = widget.get().strip()
                    if not new_widget_value_str:
                        if param_key in self.current_tuning_session_base_args: del self.current_tuning_session_base_args[param_key]; changes_made_to_session = True
                        continue
                    new_widget_value = new_widget_value_str
                value_without_session_override = effective_base_for_session.get(param_key)
                if type_hint in ["bool", "bool_flag"] and isinstance(value_without_session_override, str): value_without_session_override = value_without_session_override.lower() == 'true'
                if str(new_widget_value) != str(value_without_session_override):
                    if self.current_tuning_session_base_args.get(param_key) != new_widget_value: self.current_tuning_session_base_args[param_key] = new_widget_value; changes_made_to_session = True
                elif param_key in self.current_tuning_session_base_args: del self.current_tuning_session_base_args[param_key]; changes_made_to_session = True
            if changes_made_to_session: self.log_to_console("Session base arguments updated for current tuning."); self.update_tuning_display()
            dialog.destroy()
        button_frame = ctk.CTkFrame(dialog_main_frame); button_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        ctk.CTkButton(button_frame, text="Apply Session Args", command=save_session_args_action).pack(side="left", padx=10); ctk.CTkButton(button_frame, text="Cancel", command=dialog.destroy).pack(side="right", padx=10)
        dialog.wait_window()

    def edit_permanent_model_args(self):
        if not self.tuning_in_progress or not self.current_tuning_model_path: messagebox.showwarning("Not Available", "This option is available only during an active tuning session for a selected model.", parent=self); return
        self.open_model_specific_edit_dialog(self.current_tuning_model_path)

    def select_new_gguf_during_tuning(self):
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
            if not messagebox.askyesno("Process Running", "A KoboldCpp monitoring process might be running. Stop it and select a new model?", parent=self): return
            self.log_to_console("Stopping monitored KoboldCpp process..."); koboldcpp_core.kill_process(self.kcpp_process_obj.pid, force=True); self.kcpp_process_obj = None
        self.end_tuning_session(switch_to_model_selection=True)

    def end_tuning_session(self, switch_to_model_selection=True):
        if not self.tuning_in_progress: return
        self.log_to_console("Ending tuning session.")
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None: self.log_to_console(f"Stopping any active monitored KCPP process (PID: {self.kcpp_process_obj.pid}) from tuning session..."); koboldcpp_core.kill_process(self.kcpp_process_obj.pid, force=True); self.kcpp_process_obj = None
        self.tuning_in_progress = False; self.current_tuning_session_base_args = {}; self.last_free_vram_after_load_mb = None; self.last_approx_vram_used_kcpp_mb = None; self.last_successful_monitored_run_details_gui = None
        self._return_to_full_tuning_menu()
        if switch_to_model_selection: self._show_model_selection_view()

    def launch_and_monitor_for_tuning(self):
        if not self.tuning_in_progress: messagebox.showerror("Error", "Tuning session not active.", parent=self); return
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None: messagebox.showwarning("Process Running", "A KoboldCpp process is already being monitored. Please wait or stop it.", parent=self); return
        self.log_to_console(f"Tuning: Launch & Monitor OT Level {self.current_tuning_attempt_level}")
        if hasattr(self, 'kcpp_live_output_text') and self.kcpp_live_output_text.winfo_exists(): self.kcpp_live_output_text.configure(state="normal"); self.kcpp_live_output_text.delete("1.0", "end"); self.kcpp_live_output_text.insert("1.0", f"Preparing to launch KoboldCpp with OT Level {self.current_tuning_attempt_level}...\n"); self.kcpp_live_output_text.configure(state="disabled")
        self._set_tuning_buttons_state("disabled")
        if hasattr(self, 'post_monitor_choices_frame') and self.post_monitor_choices_frame.winfo_exists(): self.post_monitor_choices_frame.grid_remove()
        self.kcpp_success_event.clear(); self.kcpp_oom_event.clear(); self.kcpp_output_lines_shared.clear(); self.last_free_vram_after_load_mb = None; self.last_approx_vram_used_kcpp_mb = None; self.level_of_last_monitored_run = self.current_tuning_attempt_level
        ot_string = koboldcpp_core.generate_overridetensors(self.current_tuning_model_analysis, self.current_tuning_attempt_level)
        args_list = koboldcpp_core.build_command(self.current_tuning_model_path, ot_string, self.current_tuning_model_analysis, self.current_tuning_session_base_args)
        self.current_command_list_for_db = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, args_list)
        
        # Get *actual hardware* free VRAM before launch for accurate KCPP usage calculation later
        _, _, _, gpu_info_before_launch = koboldcpp_core.get_available_vram_mb(
            self.config,
            target_gpu_type=self.config.get("gpu_selection_mode", "auto") if self.config.get("gpu_selection_mode", "auto") != "auto" else None,
            target_gpu_index=self.config.get("selected_gpu_index", 0)
        )
        self.vram_at_decision_for_db = gpu_info_before_launch.get("free_mb") # Store raw HW free VRAM

        self.kcpp_process_obj, launch_error_msg = koboldcpp_core.launch_process(self.current_command_list_for_db, capture_output=True, new_console=False, use_text_mode=False)
        
        if launch_error_msg or not self.kcpp_process_obj:
            self.log_to_console(f"Failed to launch KoboldCpp for monitoring: {launch_error_msg or 'Unknown error'}"); self._log_to_kcpp_live_output(f"LAUNCH ERROR: {launch_error_msg or 'Unknown error'}\n")
            koboldcpp_core.save_config_to_db(self.db_path, self.current_tuning_model_path, self.current_tuning_model_analysis, self.vram_at_decision_for_db, self.current_command_list_for_db, self.level_of_last_monitored_run, "LAUNCH_FAILED_SETUP_GUI", None)
            self._set_tuning_buttons_state("normal"); return
        self._log_to_kcpp_live_output(f"KoboldCpp process started (PID: {self.kcpp_process_obj.pid}). Monitoring output...\n")
        effective_args_for_port_check = {**self.config.get("default_args", {}), **self.current_tuning_session_base_args}; target_port_str_for_success = effective_args_for_port_check.get("--port", "5000")
        success_pattern_regex = self.config.get("kobold_success_pattern", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["kobold_success_pattern"]); oom_keywords_list = self.config.get("oom_error_keywords", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["oom_error_keywords"])
        self.kcpp_monitor_thread = threading.Thread(target=self._monitor_kcpp_output_thread_target, args=(self.kcpp_process_obj, success_pattern_regex, oom_keywords_list, target_port_str_for_success), daemon=True); self.kcpp_monitor_thread.start(); self.monitor_start_time = time.monotonic(); self._poll_monitor_status()

    def _log_to_kcpp_live_output(self, text_line: str):
        def _update():
            if hasattr(self, 'kcpp_live_output_text') and self.kcpp_live_output_text.winfo_exists(): self.kcpp_live_output_text.configure(state="normal"); self.kcpp_live_output_text.insert("end", text_line); self.kcpp_live_output_text.see("end"); self.kcpp_live_output_text.configure(state="disabled")
        if hasattr(self, 'after'): self.after(0, _update)

    def _monitor_kcpp_output_thread_target(self, process, success_regex, oom_keywords, target_port):
        try:
            for line_bytes in iter(process.stdout.readline, b''):
                if not line_bytes: break
                try: line_decoded = line_bytes.decode('utf-8', errors='replace')
                except UnicodeDecodeError: line_decoded = line_bytes.decode('latin-1', errors='replace') # Fallback
                
                self._log_to_kcpp_live_output(line_decoded) 
                
                line_strip_lower = line_decoded.strip().lower()
                if line_strip_lower: 
                    self.kcpp_output_lines_shared.append(line_decoded.strip()) 
                    
                    if not self.kcpp_success_event.is_set() and not self.kcpp_oom_event.is_set():
                        success_match = re.search(success_regex, line_decoded.strip(), re.IGNORECASE)
                        if success_match:
                            launched_port_from_log = target_port 
                            try: launched_port_from_log = success_match.group(1) 
                            except IndexError: pass 
                            
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
                except: pass 

    def _poll_monitor_status(self):
        loading_timeout_seconds = float(self.config.get("loading_timeout_seconds", 60)); elapsed_time = time.monotonic() - self.monitor_start_time
        process_has_exited = False
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is not None:
            if not self.kcpp_success_event.is_set() and not self.kcpp_oom_event.is_set(): process_has_exited = True
        if self.kcpp_success_event.is_set(): self._handle_monitoring_completion("SUCCESS_LOAD_DETECTED_GUI")
        elif self.kcpp_oom_event.is_set(): self._handle_monitoring_completion("OOM_CRASH_DETECTED_GUI")
        elif process_has_exited: self._handle_monitoring_completion("PREMATURE_EXIT_GUI")
        elif elapsed_time > loading_timeout_seconds: self._handle_monitoring_completion("TIMEOUT_NO_SIGNAL_GUI")
        else: self.after(250, self._poll_monitor_status)

    def _set_tuning_buttons_state(self, state="normal"):
        button_attr_names = ['btn_tune_launch_monitor', 'btn_tune_skip_launch_direct', 'btn_tune_more_gpu', 'btn_tune_more_cpu', 'btn_tune_edit_args', 'btn_tune_edit_model_perm_args', 'btn_tune_new_gguf', 'btn_tune_history', 'btn_tune_quit_tuning']
        buttons_to_configure = [getattr(self, name, None) for name in button_attr_names]
        for btn in buttons_to_configure:
            if btn and hasattr(btn, 'winfo_exists') and btn.winfo_exists(): btn.configure(state=state)
        if state == "normal" and self.tuning_in_progress:
            if hasattr(self, 'btn_tune_more_gpu') and self.btn_tune_more_gpu.winfo_exists(): self.btn_tune_more_gpu.configure(state="normal" if self.current_tuning_attempt_level > self.current_tuning_min_level else "disabled")
            if hasattr(self, 'btn_tune_more_cpu') and self.btn_tune_more_cpu.winfo_exists(): self.btn_tune_more_cpu.configure(state="normal" if self.current_tuning_attempt_level < self.current_tuning_max_level else "disabled")
            if hasattr(self, 'btn_tune_edit_model_perm_args') and self.btn_tune_edit_model_perm_args.winfo_exists(): self.btn_tune_edit_model_perm_args.configure(state="normal" if self.current_tuning_model_path else "disabled")

    def _handle_post_monitor_action(self, action_key: str, original_outcome: str):
        self.log_to_console(f"User selected post-monitoring action: '{action_key}' for outcome '{original_outcome}'"); should_stop_monitored_kcpp = True
        if action_key == "launch_for_use" or action_key == "launch_for_use_risky":
            if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None: should_stop_monitored_kcpp = False
            else: self.log_to_console("KCPP instance for 'launch_for_use' is not running. Will attempt a new launch."); should_stop_monitored_kcpp = True
        if should_stop_monitored_kcpp and self.kcpp_process_obj and self.kcpp_process_obj.poll() is None: self.log_to_console(f"Stopping monitored KCPP instance (PID: {self.kcpp_process_obj.pid}) based on user action '{action_key}'."); koboldcpp_core.kill_process(self.kcpp_process_obj.pid, force=True)
        if should_stop_monitored_kcpp : self.kcpp_process_obj = None
        command_that_led_to_outcome = self.current_command_list_for_db; db_outcome_suffix = "_GUI"
        if action_key == "launch_for_use": db_outcome_suffix = "_USER_ACCEPTED_TUNED_GUI"
        elif action_key == "launch_for_use_risky": db_outcome_suffix = "_USER_LAUNCHED_RISKY_GUI"
        elif action_key == "save_good_more_gpu": db_outcome_suffix = "_USER_SAVED_GOOD_GPU_GUI"
        elif action_key == "more_gpu_now": db_outcome_suffix = "_USER_WANTS_MORE_GPU_GUI"
        elif action_key == "auto_adjust_cpu": db_outcome_suffix = "_USER_AUTO_ADJUST_CPU_GUI"
        elif action_key == "more_cpu_after_fail": db_outcome_suffix = "_USER_TRIED_CPU_FAIL_GUI"
        elif action_key == "more_cpu_now": db_outcome_suffix = "_USER_WANTS_MORE_CPU_GUI"
        elif action_key == "return_to_tuning_menu": db_outcome_suffix = "_USER_RETURNED_MENU_GUI"
        final_db_outcome_for_this_run = original_outcome + db_outcome_suffix
        koboldcpp_core.save_config_to_db(self.db_path, self.current_tuning_model_path, self.current_tuning_model_analysis, self.vram_at_decision_for_db, command_that_led_to_outcome, self.level_of_last_monitored_run, final_db_outcome_for_this_run, self.last_approx_vram_used_kcpp_mb); self.load_history()
        if action_key == "launch_for_use":
            if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
                self.log_to_console("Transferring control of monitored KCPP instance for use."); self.last_process = self.kcpp_process_obj; self.process_running = True; self.kcpp_process_obj = None
                if self.config.get("auto_open_webui", True): effective_args_for_webui = {**self.config.get("default_args", {}), **self.current_tuning_session_base_args}; port_to_open = effective_args_for_webui.get("--port", "5000"); webbrowser.open(f"http://localhost:{port_to_open}")
                self.end_tuning_session(switch_to_model_selection=True); return
            else: self.log_to_console("Monitored KCPP not running, re-launching for use with accepted config."); self._launch_final_koboldcpp(command_that_led_to_outcome, final_db_outcome_for_this_run, self.level_of_last_monitored_run); self.end_tuning_session(switch_to_model_selection=True); return
        elif action_key == "launch_for_use_risky": self.log_to_console("Attempting to launch risky configuration for use..."); self._launch_final_koboldcpp(command_that_led_to_outcome, final_db_outcome_for_this_run, self.level_of_last_monitored_run); self.end_tuning_session(switch_to_model_selection=True); return
        if action_key in ["save_good_more_gpu", "more_gpu_now"]:
            if self.current_tuning_attempt_level > self.current_tuning_min_level: self.current_tuning_attempt_level -= 1
            else: self.log_to_console("Already at Max GPU (Min Level). Cannot decrease further.")
        elif action_key in ["auto_adjust_cpu", "more_cpu_after_fail", "more_cpu_now"]:
            if self.current_tuning_attempt_level < self.current_tuning_max_level: self.current_tuning_attempt_level += 1
            else: self.log_to_console("Already at Max CPU (Max Level). Cannot increase further.")
        self._return_to_full_tuning_menu()

    def _launch_final_koboldcpp(self, command_list_to_run: list, db_outcome_reason: str, attempt_level_for_db: int):
        self.check_koboldcpp_executable()
        if not self.koboldcpp_executable or not (os.path.exists(self.koboldcpp_executable) or shutil.which(self.koboldcpp_executable)): messagebox.showerror("KoboldCpp Not Found", f"The KoboldCpp executable path is not valid or not found:\n{self.koboldcpp_executable}", parent=self); self.log_to_console(f"Launch Aborted: KCPP executable '{self.koboldcpp_executable}' not found."); return
        self.log_to_console(f"Preparing to launch KoboldCpp for use. DB Outcome Reason: {db_outcome_reason}")
        if self.last_process and self.last_process.poll() is None: self.log_to_console(f"Stopping previously launched KCPP (PID: {self.last_process.pid})..."); koboldcpp_core.kill_process(self.last_process.pid, force=True); self.last_process = None
        self.process_running = False
        
        # Get actual HW VRAM at this launch decision for DB logging
        _, _, _, gpu_info_at_final_launch = koboldcpp_core.get_available_vram_mb(
            self.config,
            target_gpu_type=self.config.get("gpu_selection_mode", "auto") if self.config.get("gpu_selection_mode", "auto") != "auto" else None,
            target_gpu_index=self.config.get("selected_gpu_index", 0)
        )
        vram_at_this_launch_decision = gpu_info_at_final_launch.get("free_mb") # Actual HW free

        model_path_for_log = self.current_tuning_model_path if self.tuning_in_progress else self.current_model_path; model_analysis_for_log = self.current_tuning_model_analysis if self.tuning_in_progress else self.model_analysis_info
        if not model_path_for_log or not model_analysis_for_log: self.log_to_console("Warning: Model path or analysis missing for DB logging during final launch.");
        if self.current_model_path: model_path_for_log = self.current_model_path
        if self.model_analysis_info: model_analysis_for_log = self.model_analysis_info
        cmd_final = command_list_to_run[:]
        if cmd_final:
            if cmd_final[0].lower() == sys.executable.lower() and len(cmd_final) > 1 and cmd_final[1].lower().endswith(".py"):
                if self.koboldcpp_executable.lower().endswith(".py"): cmd_final[1] = self.koboldcpp_executable
            else: cmd_final[0] = self.koboldcpp_executable
        else: self.log_to_console("Error: Empty command list for final launch."); messagebox.showerror("Launch Error", "Cannot launch: Command list is empty.", parent=self); return
        
        # Use actual HW VRAM for DB
        koboldcpp_core.save_config_to_db(self.db_path, model_path_for_log, model_analysis_for_log, vram_at_this_launch_decision, cmd_final, attempt_level_for_db, db_outcome_reason, self.last_approx_vram_used_kcpp_mb); 
        self.load_history()
        
        launched_kcpp_process, launch_err_msg = koboldcpp_core.launch_process(cmd_final, capture_output=False, new_console=True)
        if launch_err_msg or not launched_kcpp_process:
            self.log_to_console(f"Failed to launch KoboldCpp: {launch_err_msg or 'Unknown error'}"); messagebox.showerror("Launch Error", f"Failed to launch KoboldCpp:\n{launch_err_msg or 'Unknown error'}", parent=self)
            koboldcpp_core.save_config_to_db(self.db_path, model_path_for_log, model_analysis_for_log, vram_at_this_launch_decision, cmd_final, attempt_level_for_db, "LAUNCH_FOR_USE_FAILED_GUI", self.last_approx_vram_used_kcpp_mb); self.load_history()
        else:
            self.log_to_console(f"KoboldCpp launched successfully for use (PID: {launched_kcpp_process.pid}). New console window opened."); self.last_process = launched_kcpp_process; self.process_running = True
            if self.config.get("auto_open_webui", True):
                args_dict_from_cmd = koboldcpp_core.args_list_to_dict(cmd_final[1:]) # Get args part only
                port_to_open_webui = "5000" 
                
                base_args_for_launch = self._get_merged_args_for_model(model_path_for_log)
                launch_specific_args_dict = koboldcpp_core.args_list_to_dict(koboldcpp_core.build_command(model_path_for_log, None, model_analysis_for_log, base_args_for_launch)) 
                launch_specific_args_dict.update(args_dict_from_cmd)

                if "--port" in launch_specific_args_dict:
                    port_to_open_webui = launch_specific_args_dict["--port"]
                
                try:
                    port_num = int(port_to_open_webui)
                    if 1 <= port_num <= 65535: url_to_open = f"http://localhost:{port_num}"; self.log_to_console(f"Auto-opening Web UI at {url_to_open} in a few seconds..."); threading.Timer(3.0, lambda: webbrowser.open(url_to_open)).start()
                    else: self.log_to_console(f"Invalid port number '{port_to_open_webui}' for auto-opening Web UI.")
                except ValueError: self.log_to_console(f"Invalid port value '{port_to_open_webui}' configured; cannot auto-open Web UI.")
            else: self.log_to_console("Auto-open Web UI is disabled in settings.")

    def skip_tune_and_launch_direct(self):
        if not self.tuning_in_progress: messagebox.showwarning("Not Tuning", "Tuning session not active. Cannot skip.", parent=self); return
        self.log_to_console("User chose to skip further tuning and launch current configuration directly.")

        # Get actual HW VRAM at this launch decision for DB logging
        _, _, _, gpu_info_at_direct_launch = koboldcpp_core.get_available_vram_mb(
            self.config,
            target_gpu_type=self.config.get("gpu_selection_mode", "auto") if self.config.get("gpu_selection_mode", "auto") != "auto" else None,
            target_gpu_index=self.config.get("selected_gpu_index", 0)
        )
        # self.vram_at_decision_for_db is already set with HW free VRAM if a monitor run happened.
        # If not (e.g., direct skip after starting tune), use current HW free VRAM.
        if self.vram_at_decision_for_db is None:
            self.vram_at_decision_for_db = gpu_info_at_direct_launch.get("free_mb")


        ot_string_for_launch = koboldcpp_core.generate_overridetensors(self.current_tuning_model_analysis, self.current_tuning_attempt_level)
        args_list_for_launch = koboldcpp_core.build_command(self.current_tuning_model_path, ot_string_for_launch, self.current_tuning_model_analysis, self.current_tuning_session_base_args)
        command_to_run_final = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, args_list_for_launch)
        self._launch_final_koboldcpp(command_to_run_final, "SUCCESS_USER_DIRECT_LAUNCH_GUI", self.current_tuning_attempt_level)
        self.end_tuning_session(switch_to_model_selection=True)

    def save_settings_action(self):
        if self.save_config(): messagebox.showinfo("Settings Saved", "Global launcher settings have been saved!", parent=self)

    def reset_config_action(self):
        if messagebox.askyesno("Reset Configuration", "Are you sure you want to reset ALL launcher settings to their original defaults?\nThis will overwrite your current configuration file.\n\nA backup of your current settings will be attempted.", icon=messagebox.WARNING, parent=self):
            try:
                core_config_filepath = koboldcpp_core.CONFIG_FILE
                if os.path.exists(core_config_filepath): backup_filepath = core_config_filepath + f".backup_reset_{time.strftime('%Y%m%d-%H%M%S')}.json"; shutil.copy2(core_config_filepath, backup_filepath); self.log_to_console(f"Current configuration backed up to: {backup_filepath}")
                if os.path.exists(core_config_filepath): os.remove(core_config_filepath)
                core_init_results_after_reset = koboldcpp_core.initialize_launcher(); self.config = core_init_results_after_reset["config"]; self.system_info = core_init_results_after_reset["system_info"]; self.gpu_info = core_init_results_after_reset["gpu_info"]; self.koboldcpp_capabilities = core_init_results_after_reset.get("koboldcpp_capabilities", {})
                self.koboldcpp_executable = self.config.get("koboldcpp_executable"); self.default_model_dir = self.config.get("default_gguf_dir", ""); self.db_path = self.config["db_file"]
                self.load_settings_from_config(); self.check_koboldcpp_executable()
                if hasattr(self, 'populate_model_specifics_list_display'): self.populate_model_specifics_list_display()
                if hasattr(self, 'update_kcpp_capabilities_display'): self.update_kcpp_capabilities_display(re_detect=True)
                self.after(100, self._populate_gpu_id_dropdown_on_startup)
                self.log_to_console(f"Configuration reset to defaults. DB path: {self.db_path}"); messagebox.showinfo("Configuration Reset", "Launcher settings have been reset to defaults.\nYou might need to go through the first-time setup again if paths are incorrect.", parent=self); self.after(100, self._run_first_time_setup_if_needed)
            except Exception as e_reset: error_msg_reset = f"An error occurred while resetting configuration: {e_reset}\n{traceback.format_exc()}"; self.log_to_console(error_msg_reset); messagebox.showerror("Reset Error", error_msg_reset, parent=self)

    def launch_direct_defaults(self):
        if not self.current_model_path: messagebox.showwarning("No Model Selected", "Please select a GGUF model file first.", parent=self); return
        self.check_koboldcpp_executable()
        if not self.koboldcpp_executable or not (os.path.exists(self.koboldcpp_executable) or shutil.which(self.koboldcpp_executable)): messagebox.showerror("KoboldCpp Not Found", f"The KoboldCpp executable path is not valid or not found:\n{self.koboldcpp_executable}", parent=self); return
        self.log_to_console(f"Attempting direct launch for: {os.path.basename(self.current_model_path)} using default settings.")
        effective_args_for_launch = self._get_merged_args_for_model(self.current_model_path); current_model_analysis = self.model_analysis_info
        if not current_model_analysis or current_model_analysis.get('filepath') != self.current_model_path: self.log_to_console("Re-analyzing model for direct launch..."); self.analyze_model_action(self.current_model_path); current_model_analysis = self.model_analysis_info
        if not current_model_analysis or 'filepath' not in current_model_analysis : messagebox.showerror("Analysis Error", "Failed to analyze model. Cannot proceed with direct launch.", parent=self); self.log_to_console("Direct launch aborted: Model analysis failed."); return
        args_list_for_kcpp = koboldcpp_core.build_command(self.current_model_path, None, current_model_analysis, effective_args_for_launch); command_list_final = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, args_list_for_kcpp)
        
        # Get actual HW VRAM at this launch decision for DB logging
        _, _, _, gpu_info_at_direct_launch = koboldcpp_core.get_available_vram_mb(
            self.config,
            target_gpu_type=self.config.get("gpu_selection_mode", "auto") if self.config.get("gpu_selection_mode", "auto") != "auto" else None,
            target_gpu_index=self.config.get("selected_gpu_index", 0)
        )
        self.vram_at_decision_for_db = gpu_info_at_direct_launch.get("free_mb") # Actual HW free
        
        self.last_approx_vram_used_kcpp_mb = None # No monitoring for direct launch
        self._launch_final_koboldcpp(command_list_final, "SUCCESS_USER_DIRECT_SETTINGS_GUI", 0)

    def launch_best_remembered(self):
        if not self.current_model_path: messagebox.showwarning("No Model Selected", "Please select a GGUF model file first.", parent=self); return
        current_model_analysis = self.model_analysis_info
        if not current_model_analysis or current_model_analysis.get('filepath') != self.current_model_path: self.log_to_console("Re-analyzing model for launching best remembered..."); self.analyze_model_action(self.current_model_path); current_model_analysis = self.model_analysis_info
        if not current_model_analysis or 'filepath' not in current_model_analysis: messagebox.showerror("Analysis Error", "Failed to analyze model. Cannot find best remembered config.", parent=self); self.log_to_console("Launch best remembered aborted: Model analysis failed."); return
        self.check_koboldcpp_executable()
        if not self.koboldcpp_executable or not (os.path.exists(self.koboldcpp_executable) or shutil.which(self.koboldcpp_executable)): messagebox.showerror("KoboldCpp Not Found", f"The KoboldCpp executable path is not valid or not found:\n{self.koboldcpp_executable}", parent=self); return
        self.log_to_console("Attempting to launch using the best remembered configuration...")
        
        # Get actual hardware VRAM for history lookup
        _, _, _, current_gpu_full_info = koboldcpp_core.get_available_vram_mb(
            self.config,
            target_gpu_type=self.config.get("gpu_selection_mode", "auto") if self.config.get("gpu_selection_mode", "auto") != "auto" else None,
            target_gpu_index=self.config.get("selected_gpu_index", 0)
        )
        current_actual_hw_vram_mb = current_gpu_full_info.get("free_mb", 0.0)

        best_historical_config_found = koboldcpp_core.find_best_historical_config(self.db_path, current_model_analysis, current_actual_hw_vram_mb, self.config)
        if best_historical_config_found and best_historical_config_found.get("args_list"):
            self.log_to_console(f"Found best remembered config - Level: {best_historical_config_found['attempt_level']}, Outcome: {best_historical_config_found['outcome']}"); remembered_args_list = best_historical_config_found["args_list"]; remembered_args_dict = koboldcpp_core.args_list_to_dict(remembered_args_list); historical_ot_string = remembered_args_dict.pop("--overridetensors", None); base_args_for_this_launch = self._get_merged_args_for_model(self.current_model_path); final_effective_args_dict = base_args_for_this_launch.copy(); remembered_args_dict.pop("--model", None); final_effective_args_dict.update(remembered_args_dict)
            final_command_args_list = koboldcpp_core.build_command(self.current_model_path, historical_ot_string, current_model_analysis, final_effective_args_dict); command_list_to_execute = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, final_command_args_list)
            
            self.vram_at_decision_for_db = current_actual_hw_vram_mb # Actual HW free VRAM for DB
            self.last_approx_vram_used_kcpp_mb = best_historical_config_found.get("approx_vram_used_kcpp_mb"); # Actual HW used from history
            historical_attempt_level = best_historical_config_found.get("attempt_level", 0)
            self._launch_final_koboldcpp(command_list_to_execute, "SUCCESS_USER_LAUNCHED_BEST_REMEMBERED_GUI", historical_attempt_level)
        else: self.log_to_console("No suitable remembered configuration found in history. Falling back to direct launch with default settings."); messagebox.showinfo("Not Found", "No suitable remembered configuration was found in the launch history for this model and current VRAM.\n\nLaunching with default settings instead.", parent=self); self.launch_direct_defaults()

    def change_theme(self, theme_name: str):
        ctk.set_appearance_mode(theme_name.lower()); self.mark_settings_dirty(); self.log_to_console(f"UI Theme changed to: {theme_name}.")

    def log_to_console(self, text_message: str):
        def _perform_log():
            if hasattr(self,'console') and self.console.winfo_exists(): self.console.configure(state="normal"); timestamp = time.strftime("%H:%M:%S"); self.console.insert("end",f"[{timestamp}] {text_message}\n"); self.console.see("end"); self.console.configure(state="disabled")
        if hasattr(self,'after'): self.after(0, _perform_log)
        else: print(f"LOG: {text_message}")

    def browse_model(self):
        initial_dir_for_dialog = self.config.get("last_used_gguf_dir", self.default_model_dir)
        if not initial_dir_for_dialog or not os.path.isdir(initial_dir_for_dialog): initial_dir_for_dialog = os.getcwd()
        filepath_selected = filedialog.askopenfilename(title="Select GGUF Model File", filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")], initialdir=initial_dir_for_dialog, parent=self)
        if filepath_selected:
            if hasattr(self, 'model_path_entry') and self.model_path_entry.winfo_exists(): self.model_path_entry.delete(0, "end"); self.model_path_entry.insert(0, filepath_selected)
            self.current_model_path = os.path.abspath(filepath_selected); new_last_dir = os.path.dirname(self.current_model_path)
            if self.config.get("last_used_gguf_dir") != new_last_dir: self.config["last_used_gguf_dir"] = new_last_dir; self.save_config()
            self.analyze_model_action(self.current_model_path)

    def analyze_model_action(self, model_filepath_to_analyze: str):
        self.log_to_console(f"Analyzing model: {os.path.basename(model_filepath_to_analyze)}"); self.model_analysis_info = koboldcpp_core.analyze_filename(model_filepath_to_analyze)
        moe_str = 'MoE' if self.model_analysis_info.get('is_moe') else 'Dense'; size_b_str = self.model_analysis_info.get('size_b', "N/A"); quant_str = self.model_analysis_info.get('quant', "N/A"); num_layers_str = self.model_analysis_info.get('num_layers', "N/A"); est_vram_str = self.model_analysis_info.get('estimated_vram_gb_full_gpu', "N/A")
        info_text = (f"Type: {moe_str}, Size: ~{size_b_str}B, Quant: {quant_str}, Layers: {num_layers_str}, Est. Full VRAM: {est_vram_str}GB")
        if hasattr(self,'model_info_label') and self.model_info_label.winfo_exists(): self.model_info_label.configure(text=info_text)
        self.log_to_console(f"Model Analysis Complete - {info_text}")
        if self.tuning_in_progress and self.current_tuning_model_path == model_filepath_to_analyze: self.current_tuning_model_analysis = self.model_analysis_info.copy(); self.update_tuning_display()

    def view_history_for_current_model(self):
        model_path_for_history = None
        if self.tuning_in_progress and self.current_tuning_model_path: model_path_for_history = self.current_tuning_model_path; self.log_to_console(f"Viewing history for current tuning model: {os.path.basename(model_path_for_history)}")
        elif self.current_model_path: model_path_for_history = self.current_model_path; self.log_to_console(f"Viewing history for selected model: {os.path.basename(model_path_for_history)}")
        if model_path_for_history: self.load_history(model_filter=model_path_for_history); self.tabview.set("History")
        else: messagebox.showwarning("No Model Context", "No model is currently selected or active in a tuning session to show specific history.", parent=self)

    def setup_history_tab(self):
            self.history_tab_content_frame = ctk.CTkFrame(self.tab_history, fg_color="transparent")
            self.history_tab_content_frame.pack(fill="both", expand=True, padx=5, pady=5)
            self.history_tab_content_frame.grid_columnconfigure(0, weight=1)
            self.history_tab_content_frame.grid_rowconfigure(1, weight=1) # Make row 1 (scrollable part) expand

            # Permanent frame for controls
            history_controls_frame = ctk.CTkFrame(self.history_tab_content_frame)
            history_controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
            
            self.btn_refresh_history = ctk.CTkButton(history_controls_frame, text="Refresh History", command=lambda: self.load_history(model_filter=None))
            self.btn_refresh_history.pack(side="left", padx=10, pady=5)
            ToolTip(self.btn_refresh_history, "Reload and display all launch history records from the database.")

            # This label will show "Loading..." or the title like "History for Model X"
            self.history_title_label = ctk.CTkLabel(history_controls_frame, text="Loading history...", justify="left", anchor="w")
            self.history_title_label.pack(side="left", padx=10, pady=5, fill="x", expand=True)

            # Scrollable frame for the actual history entries
            self.history_scrollable_frame = ctk.CTkScrollableFrame(self.history_tab_content_frame)
            self.history_scrollable_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
            self.history_scrollable_frame.grid_columnconfigure(0, weight=1)
            
            self.load_history()

    def load_history(self, model_filter=None):
        if not hasattr(self, 'history_scrollable_frame') or not self.history_scrollable_frame.winfo_exists():
            self.log_to_console("History scrollable frame not ready for loading history.")
            return
        
        if not hasattr(self, 'history_title_label') or not self.history_title_label.winfo_exists():
            self.log_to_console("History title label not ready.")
            return

        # Clear only the scrollable frame's children (the history entries)
        for widget in self.history_scrollable_frame.winfo_children():
            if widget and widget.winfo_exists():
                widget.destroy()

        self.history_title_label.configure(text="Fetching history...")
        try:
            limit = 100
            all_entries = koboldcpp_core.get_history_entries(self.db_path, limit=limit)
            
            display_title_text = ""
            entries_to_show = []

            if not all_entries:
                display_title_text = "No launch history found."
                entries_to_show = []
            elif model_filter:
                entries_to_show = [entry for entry in all_entries if entry[0] == model_filter]
                if not entries_to_show:
                    display_title_text = f"No history found for model: {os.path.basename(model_filter)}"
                else:
                    display_title_text = f"Launch History for {os.path.basename(model_filter)} (Last {min(len(entries_to_show), 20)})"
                    entries_to_show = entries_to_show[:20] # Show top 20 for filtered
            else:
                display_title_text = f"Global Launch History (Last {min(len(all_entries), 20)})"
                entries_to_show = all_entries[:20] # Show top 20 for global

            self.history_title_label.configure(text=display_title_text, font=ctk.CTkFont(weight="bold"))

            if not entries_to_show:
                no_hist_label = ctk.CTkLabel(self.history_scrollable_frame, text="No relevant history entries to display.")
                no_hist_label.pack(fill="x", pady=10, padx=10)
                return
            
            for record in entries_to_show:
                model_fn = os.path.basename(record[0])
                size_b = f"{record[1]:.1f}B" if isinstance(record[1], float) else (str(record[1]) + "B" if record[1] is not None else "N/A")
                quant = str(record[2]) if record[2] else "N/A"
                moe = "Y" if record[3] else "N"
                vram_l = f"{record[4]}MB" if record[4] is not None else "N/A" # VRAM at launch decision (actual HW)
                lvl = str(record[5]) if record[5] is not None else "N/A"
                outcome = str(record[6]) if record[6] else "N/A"
                vram_u = f"{record[7]}MB" if record[7] is not None else "N/A" # Approx actual KCPP VRAM used
                ts_obj = record[8]
                ts_str = ts_obj.strftime('%Y-%m-%d %H:%M') if isinstance(ts_obj, koboldcpp_core.datetime) else str(ts_obj)
                
                entry_text = (f"Model: {model_fn} ({size_b}, {quant}, MoE:{moe})\n"
                              f"  Lvl: {lvl}, VRAM@Launch: {vram_l}, Actual VRAM Used: {vram_u}\n"
                              f"  Outcome: {outcome}\n"
                              f"  Time: {ts_str}")
                
                entry_label = ctk.CTkLabel(self.history_scrollable_frame, text=entry_text, justify="left", anchor="w")
                entry_label.pack(fill="x", padx=10, pady=(5,2)) # Pack into the scrollable frame
                
                separator = ctk.CTkFrame(self.history_scrollable_frame, height=1, fg_color="gray50")
                separator.pack(fill="x", padx=10, pady=(0,5)) # Pack into the scrollable frame

        except Exception as e: 
            self.log_to_console(f"Error loading history: {e}")
            traceback.print_exc() 
            if hasattr(self, 'history_title_label') and self.history_title_label.winfo_exists():
                 self.history_title_label.configure(text=f"Error loading history. Check log.")


    def clear_history_display(self): pass 

    def stop_all_kcpp_processes_forcefully(self):
        self.log_to_console("Attempting to stop all KoboldCpp processes...")
        if self.last_process and self.last_process.poll() is None: pid_to_kill = self.last_process.pid; self.log_to_console(f"Stopping directly launched KCPP (PID: {pid_to_kill})..."); success, msg = koboldcpp_core.kill_process(pid_to_kill, force=True); self.log_to_console(f"Kill PID {pid_to_kill}: {success} - {msg}"); self.last_process = None; self.process_running = False
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None: pid_to_kill_monitor = self.kcpp_process_obj.pid; self.log_to_console(f"Stopping monitored KCPP (PID: {pid_to_kill_monitor})..."); success_mon, msg_mon = koboldcpp_core.kill_process(pid_to_kill_monitor, force=True); self.log_to_console(f"Kill monitored PID {pid_to_kill_monitor}: {success_mon} - {msg_mon}"); self.kcpp_process_obj = None
        if self.tuning_in_progress: self.end_tuning_session(switch_to_model_selection=False)
        self.check_koboldcpp_executable()
        if self.koboldcpp_executable:
            kcpp_exe_basename = os.path.basename(self.koboldcpp_executable); self.log_to_console(f"Performing sweep for processes matching '{kcpp_exe_basename}'..."); killed_sweep, msg_sweep = koboldcpp_core.kill_processes_by_name(kcpp_exe_basename); self.log_to_console(f"Sweep result: {msg_sweep}")
            if self.koboldcpp_executable.lower().endswith(".py"): self.log_to_console(f"Performing sweep for python processes running '{kcpp_exe_basename}'..."); killed_py_sweep, msg_py_sweep = koboldcpp_core.kill_processes_by_name("python", cmdline_substr_filter=kcpp_exe_basename); self.log_to_console(f"Python sweep result: {msg_py_sweep}")
        else: self.log_to_console("KoboldCpp executable path not configured, cannot perform broad sweep by name.")
        messagebox.showinfo("Process Stop Attempted", "Attempted to stop KoboldCpp processes. Check the launcher log for details.", parent=self)

    def _get_param_definitions_for_dialog(self):
        return [d for d in koboldcpp_core.KOBOLDCPP_ARG_DEFINITIONS if d["key"] != "--model"]


class FirstTimeSetupDialog(ctk.CTkToplevel):
    def __init__(self, parent, current_config, current_exe_path):
        super().__init__(parent)
        self.title("First-Time Setup"); self.geometry("650x400"); self.transient(parent); self.grab_set(); self.protocol("WM_DELETE_WINDOW", self._on_cancel); self.parent_app = parent; self.config_to_update = current_config.copy(); self.saved_config = None
        ctk.CTkLabel(self, text="Welcome to TensorTune Launcher!", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(15,10))
        ctk.CTkLabel(self, text="Please verify the path to your KoboldCpp executable/script and other initial settings.", wraplength=580).pack(pady=(0,15))
        
        exe_frame = ctk.CTkFrame(self); exe_frame.pack(fill="x", padx=20, pady=10); exe_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(exe_frame, text="KoboldCpp Executable:").grid(row=0, column=0, padx=(5,10), pady=5, sticky="w")
        self.exe_entry = ctk.CTkEntry(exe_frame, width=350)
        # current_exe_path is what the main app thinks is the exe path (could be from config or auto-detected)
        self.exe_entry.insert(0, current_exe_path if current_exe_path else self.config_to_update.get("koboldcpp_executable", ""))
        self.exe_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        btn_browse_exe_dialog = ctk.CTkButton(exe_frame, text="Browse", command=self._browse_exe_for_dialog, width=80); btn_browse_exe_dialog.grid(row=0, column=2, padx=5, pady=5); ToolTip(btn_browse_exe_dialog, "Select your KoboldCpp executable (e.g., .exe, .sh, or .py script).")
        
        gguf_frame = ctk.CTkFrame(self); gguf_frame.pack(fill="x", padx=20, pady=10); gguf_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(gguf_frame, text="Default GGUF Directory\n(Optional):").grid(row=0, column=0, padx=(5,10), pady=5, sticky="w")
        self.gguf_dir_entry = ctk.CTkEntry(gguf_frame, width=350)
        self.gguf_dir_entry.insert(0, self.config_to_update.get("default_gguf_dir", os.getcwd()))
        self.gguf_dir_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        btn_browse_gguf_dialog = ctk.CTkButton(gguf_frame, text="Browse", command=self._browse_gguf_dir_for_dialog, width=80); btn_browse_gguf_dialog.grid(row=0, column=2, padx=5, pady=5); ToolTip(btn_browse_gguf_dialog, "Choose a default folder where your .gguf model files are stored.")
        
        self.auto_open_webui_var_dialog = ctk.BooleanVar(value=self.config_to_update.get("auto_open_webui", True))
        ctk.CTkCheckBox(self, text="Auto-Open Web UI After Successful Launch", variable=self.auto_open_webui_var_dialog).pack(pady=15, padx=20, anchor="w")
        
        button_frame = ctk.CTkFrame(self); button_frame.pack(pady=20, fill="x", padx=20); button_frame.grid_columnconfigure((0,1), weight=1)
        btn_save = ctk.CTkButton(button_frame, text="Save and Continue", command=self._on_save_setup, width=180); btn_save.grid(row=0, column=0, padx=(0,5), pady=5, sticky="e"); ToolTip(btn_save, "Save these initial settings and proceed.")
        btn_cancel = ctk.CTkButton(button_frame, text="Cancel Setup", command=self._on_cancel, fg_color="gray", width=180); btn_cancel.grid(row=0, column=1, padx=(5,0), pady=5, sticky="w"); ToolTip(btn_cancel, "Skip this initial setup for now. You can configure settings later.")

    def _browse_exe_for_dialog(self):
        current_path_in_entry = self.exe_entry.get().strip()
        initial_dir_to_use = os.getcwd() # Ultimate fallback

        if current_path_in_entry:
            if os.path.isdir(current_path_in_entry): # If user typed a directory path
                initial_dir_to_use = current_path_in_entry
            else: # User likely typed a file path or partial file path
                parent_dir = os.path.dirname(current_path_in_entry)
                if parent_dir and os.path.isdir(parent_dir):
                    initial_dir_to_use = parent_dir
                # If parent_dir is empty (e.g., just "file.exe") or not a dir, fallback is CWD
        else: # Entry is empty, try to use parent of configured executable
            configured_exe_path = self.config_to_update.get("koboldcpp_executable", "")
            if configured_exe_path:
                parent_of_configured_exe = os.path.dirname(configured_exe_path)
                if parent_of_configured_exe and os.path.isdir(parent_of_configured_exe):
                    initial_dir_to_use = parent_of_configured_exe
            # else, fallback is CWD

        filepath_selected = filedialog.askopenfilename(
            title="Select KoboldCpp Executable or Script",
            filetypes=[("Executables/Scripts", "*.exe *.py *.sh *"), ("All files", "*.*")],
            initialdir=initial_dir_to_use,
            parent=self
        )
        if filepath_selected:
            self.exe_entry.delete(0, "end")
            self.exe_entry.insert(0, filepath_selected)

    def _browse_gguf_dir_for_dialog(self):
        current_text_in_entry = self.gguf_dir_entry.get().strip()
        
        # Start by assuming the config's default_gguf_dir (or CWD if not set/invalid)
        initial_dir_for_dialog = self.config_to_update.get("default_gguf_dir", "")
        if not initial_dir_for_dialog or not os.path.isdir(initial_dir_for_dialog):
            initial_dir_for_dialog = os.getcwd()

        # If user has typed something valid into the entry, prioritize that
        if current_text_in_entry:
            if os.path.isdir(current_text_in_entry):
                initial_dir_for_dialog = current_text_in_entry
            else:
                parent_of_text = os.path.dirname(current_text_in_entry)
                if parent_of_text and os.path.isdir(parent_of_text):
                    initial_dir_for_dialog = parent_of_text
                # If what they typed is not a dir, and its parent is not a dir,
                # initial_dir_for_dialog remains what it was (from config or CWD as determined above).

        dir_path_selected = filedialog.askdirectory(
            title="Select Default GGUF Model Directory",
            initialdir=initial_dir_for_dialog,
            parent=self
        )
        if dir_path_selected:
            self.gguf_dir_entry.delete(0, "end")
            self.gguf_dir_entry.insert(0, dir_path_selected)

    def _on_save_setup(self):
        exe_path_str = self.exe_entry.get().strip()
        gguf_dir_str = self.gguf_dir_entry.get().strip()
        valid_exe_found = False
        resolved_exe_path = ""

        if exe_path_str:
            if os.path.exists(exe_path_str) and (os.access(exe_path_str, os.X_OK) or exe_path_str.lower().endswith(".py")):
                valid_exe_found = True
                resolved_exe_path = os.path.abspath(exe_path_str)
            else:
                found_in_path = shutil.which(exe_path_str)
                if found_in_path and (os.access(found_in_path, os.X_OK) or found_in_path.lower().endswith(".py")):
                    valid_exe_found = True
                    resolved_exe_path = os.path.abspath(found_in_path)
                elif exe_path_str.lower().endswith(".py"):
                    try:
                        # Try relative to launcher script itself (if not frozen) or executable (if frozen)
                        launcher_script_dir = os.path.dirname(os.path.abspath(__file__)) if not getattr(sys,'frozen',False) else os.path.dirname(sys.executable)
                        potential_relative = os.path.join(launcher_script_dir, exe_path_str)
                        if os.path.exists(potential_relative) and (os.access(potential_relative, os.X_OK) or potential_relative.lower().endswith(".py")):
                            valid_exe_found = True
                            resolved_exe_path = os.path.abspath(potential_relative)
                    except NameError: # __file__ not defined (e.g. interactive)
                        pass # valid_exe_found remains False
        
        if not valid_exe_found:
            messagebox.showerror("Invalid Path", "The KoboldCpp executable path is not valid, not found, or not executable. Please provide a correct path.", parent=self)
            return

        if gguf_dir_str and not os.path.isdir(gguf_dir_str):
            messagebox.showwarning("Invalid Directory", "The default GGUF model directory is not a valid directory. It will be cleared.", parent=self)
            gguf_dir_str = ""
            
        self.config_to_update["koboldcpp_executable"] = resolved_exe_path
        self.config_to_update["default_gguf_dir"] = gguf_dir_str
        self.config_to_update["auto_open_webui"] = self.auto_open_webui_var_dialog.get()
        self.config_to_update["first_run_completed"] = True
        self.config_to_update["first_run_intro_shown"] = True 
        
        save_ok, save_msg = koboldcpp_core.save_launcher_config(self.config_to_update)
        if save_ok:
            self.saved_config = self.config_to_update
            self.destroy()
        else:
            messagebox.showerror("Save Error", f"Failed to save initial configuration:\n{save_msg}", parent=self)
            self.saved_config = None # Indicate save failed

    def _on_cancel(self):
        self.saved_config = None # Indicate no save occurred
        self.destroy()


if __name__ == "__main__":
    if hasattr(koboldcpp_core, '_cleanup_nvml') and callable(koboldcpp_core._cleanup_nvml):
        import atexit
        atexit.register(koboldcpp_core._cleanup_nvml)

    app = KoboldLauncherGUI()

    def on_closing_gui():
        kcpp_monitored_is_running = hasattr(app,'kcpp_process_obj') and app.kcpp_process_obj and app.kcpp_process_obj.poll() is None
        kcpp_direct_launch_is_running = app.last_process and app.last_process.poll() is None
        any_kcpp_potentially_running = app.process_running or kcpp_monitored_is_running or kcpp_direct_launch_is_running
        if any_kcpp_potentially_running:
            if messagebox.askyesno("Exit Confirmation", "A KoboldCpp process launched or monitored by this tool might still be running.\nDo you want to stop all related KoboldCpp processes and exit the launcher?", icon=messagebox.QUESTION, parent=app):
                app.log_to_console("User chose to stop processes and exit."); app.stop_all_kcpp_processes_forcefully(); time.sleep(0.2); app.destroy()
            else: app.log_to_console("User chose not to stop processes; exit cancelled."); return
        else: app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_closing_gui)
    app.mainloop()
