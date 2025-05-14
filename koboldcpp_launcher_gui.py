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
import koboldcpp_core # Assuming this is your core logic file
import platform
from pathlib import Path

class KoboldLauncherGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("KoboldCpp Smart Launcher (GUI Edition)")
        self.geometry("950x780") 
        self.minsize(850, 680)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        core_init_results = koboldcpp_core.initialize_launcher()
        self.config = core_init_results["config"]
        self.system_info = core_init_results["system_info"]
        self.gpu_info = core_init_results["gpu_info"]

        if "model_specific_args" not in self.config:
            self.config["model_specific_args"] = {}
        if "model_specific_args" not in koboldcpp_core.DEFAULT_CONFIG_TEMPLATE:
            koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["model_specific_args"] = {}
        
        # Ensure all default_args keys from template are in config
        for k_default, v_default in koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].items():
            if k_default not in self.config["default_args"]:
                self.config["default_args"][k_default] = v_default


        self.current_model_path = None
        self.process_running = False
        self.model_analysis_info = {}
        self.last_process = None

        self.db_path = self.config.get("db_file", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["db_file"])
        self.default_model_dir = self.config.get("default_gguf_dir", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE.get("default_gguf_dir", ""))
        if not self.default_model_dir or not os.path.isdir(self.default_model_dir):
            self.default_model_dir = os.getcwd()

        self.koboldcpp_executable = self.config.get("koboldcpp_executable", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["koboldcpp_executable"])

        self.tuning_in_progress = False
        self.current_tuning_attempt_level = 0
        self.current_tuning_min_level = 0
        self.current_tuning_max_level = 0
        self.current_tuning_session_base_args = {}
        self.current_tuning_model_analysis = {}
        self.current_tuning_model_path = None
        self.level_of_last_monitored_run = 0
        self.current_command_list_for_db = []
        self.vram_at_decision_for_db = None

        self.kcpp_monitor_thread = None
        self.kcpp_process_obj = None
        self.kcpp_success_event = threading.Event()
        self.kcpp_oom_event = threading.Event()
        self.kcpp_output_lines_shared = []
        self.last_approx_vram_used_kcpp_mb = None
        self.last_free_vram_after_load_mb = None

        appearance_mode = self.config.get("color_mode", "dark").lower()
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
        self.load_settings_from_config()
        self.setup_main_tab()
        self.setup_history_tab()

        threading.Thread(target=self.monitor_vram, daemon=True).start()

        self.log_to_console(f"KoboldCpp Smart Launcher GUI Initialized. Core config loaded: {core_init_results['config_message']}")
        self.check_koboldcpp_executable()
        self._show_model_selection_view()

    def _get_merged_args_for_model(self, model_path):
        merged_args = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
        merged_args.update(self.config.get("default_args", {}))
        if model_path:
            model_specifics = self.config.get("model_specific_args", {}).get(model_path, {})
            merged_args.update(model_specifics)
        return merged_args

    def _reinitialize_session_base_args(self):
        if not self.current_tuning_model_path:
            effective_args = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
            effective_args.update(self.config.get("default_args", {}))
        else:
            effective_args = self._get_merged_args_for_model(self.current_tuning_model_path)

        self.current_tuning_session_base_args = {}
        for k, v_raw in effective_args.items():
            v = v_raw
            if isinstance(v_raw, str):
                if v_raw.lower() == 'true': v = True
                elif v_raw.lower() == 'false': v = False

            if isinstance(v, bool):
                if v: # Only add if True for boolean flags that are just flags
                    self.current_tuning_session_base_args[k] = True
            elif isinstance(v, str) and v.strip():
                self.current_tuning_session_base_args[k] = v.strip()
            elif not isinstance(v, str) and v is not None :
                self.current_tuning_session_base_args[k] = v
        self.log_to_console(f"Session base args reinitialized for {os.path.basename(self.current_tuning_model_path or 'No Model')}")

    def check_koboldcpp_executable(self):
        if not os.path.exists(self.koboldcpp_executable):
            self.log_to_console(f"Warning: KoboldCPP executable '{self.koboldcpp_executable}' not found at specified path.")
            executable_basename = os.path.basename(self.koboldcpp_executable)
            found_exe = shutil.which(executable_basename)
            if found_exe:
                self.log_to_console(f"Found '{executable_basename}' in PATH: {found_exe}")
                self.koboldcpp_executable = found_exe
                if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
                    self.exe_path_entry.delete(0, "end")
                    self.exe_path_entry.insert(0, self.koboldcpp_executable)
                self.config["koboldcpp_executable"] = self.koboldcpp_executable
            else:
                self.log_to_console(f"Could not find '{executable_basename}' in PATH either.")
        else:
            self.log_to_console(f"KoboldCPP executable '{self.koboldcpp_executable}' found at specified path.")
      
    def load_settings_from_config(self):
        # Ensure all default_args keys from template exist in config
        for k_default, v_default in koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].items():
            if k_default not in self.config["default_args"]:
                self.config["default_args"][k_default] = v_default

        global_default_args = self.config.get("default_args", {})
        
        for param, widget_info in self.settings_widgets.items():
            widget = widget_info["widget"]
            value_from_config = global_default_args.get(param)
            core_default_for_param = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].get(param)
            final_value_to_set = value_from_config if value_from_config is not None else core_default_for_param

            if isinstance(widget, ctk.CTkEntry):
                widget.delete(0, "end")
                if final_value_to_set is not None:
                    widget.insert(0, str(final_value_to_set))
            elif hasattr(widget, "variable"): # CheckBox
                val_to_set_bool = False
                if isinstance(final_value_to_set, bool):
                    val_to_set_bool = final_value_to_set
                elif isinstance(final_value_to_set, str):
                    val_to_set_bool = final_value_to_set.lower() == 'true'
                widget.variable.set(val_to_set_bool)

        if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
            self.exe_path_entry.delete(0, "end")
            self.exe_path_entry.insert(0, self.koboldcpp_executable)

        if hasattr(self, 'auto_open_webui_var'):
            self.auto_open_webui_var.set(self.config.get("auto_open_webui", True))

    def save_config(self):
        if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
            self.config["koboldcpp_executable"] = self.exe_path_entry.get().strip()

        self.config["default_gguf_dir"] = self.default_model_dir
        self.config["db_file"] = self.db_path
        self.config["color_mode"] = ctk.get_appearance_mode().lower()
        
        if hasattr(self, 'auto_open_webui_var'):
            self.config["auto_open_webui"] = self.auto_open_webui_var.get()
        else:
            self.config["auto_open_webui"] = self.config.get("auto_open_webui", True)

        current_global_default_args = self.config.get("default_args", {})
        for param, widget_info in self.settings_widgets.items():
            widget = widget_info["widget"]
            if isinstance(widget, ctk.CTkEntry):
                current_global_default_args[param] = widget.get().strip()
            elif hasattr(widget, "variable"): # CheckBox
                current_global_default_args[param] = widget.variable.get()
        self.config["default_args"] = current_global_default_args

        if "model_specific_args" not in self.config:
             self.config["model_specific_args"] = {}

        success, message = koboldcpp_core.save_launcher_config(self.config)
        if success:
            self.log_to_console(message)
            self.koboldcpp_executable = self.config["koboldcpp_executable"]
            self.default_model_dir = self.config.get("default_gguf_dir", "")
        else:
            self.log_to_console(f"Error saving config: {message}")
            messagebox.showerror("Save Error", f"Could not save configuration: {message}")
        return success

    def setup_main_tab(self):
        self.tab_main.grid_rowconfigure(0, weight=1) # Ensure main_tab itself can resize
        # Model Selection View (initially visible)
        self.model_selection_frame = ctk.CTkFrame(self.tab_main)
        self.model_selection_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.model_selection_frame.grid_columnconfigure(0, weight=1) # Allow content to expand width-wise
        self.model_selection_frame.grid_rowconfigure(5, weight=1) # Allow console to expand height-wise

        title_label = ctk.CTkLabel(self.model_selection_frame, text="KoboldCpp Model Launcher", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 15), sticky="n")

        model_controls_frame = ctk.CTkFrame(self.model_selection_frame)
        model_controls_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        model_controls_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(model_controls_frame, text="GGUF Model:").grid(row=0, column=0, padx=(10,5), pady=10, sticky="w")
        self.model_path_entry = ctk.CTkEntry(model_controls_frame, width=400)
        self.model_path_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        ctk.CTkButton(model_controls_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=(5,10), pady=10, sticky="e")
        self.model_info_label = ctk.CTkLabel(model_controls_frame, text="No model selected", justify="left")
        self.model_info_label.grid(row=1, column=0, columnspan=3, padx=10, pady=(0,10), sticky="w")

        vram_frame = ctk.CTkFrame(self.model_selection_frame)
        vram_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        vram_frame.grid_columnconfigure(1, weight=1) # ProgressBar takes available width
        ctk.CTkLabel(vram_frame, text="GPU VRAM Status:").grid(row=0, column=0, padx=(10,5), pady=5, sticky="w")
        self.vram_progress = ctk.CTkProgressBar(vram_frame, height=18) # Slightly taller
        self.vram_progress.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.vram_progress.set(0)
        self.vram_text = ctk.CTkLabel(vram_frame, text="Scanning...")
        self.vram_text.grid(row=0, column=2, padx=(5,10), pady=5, sticky="e")
        ctk.CTkButton(vram_frame, text="Refresh", width=60, command=self.refresh_vram).grid(row=0, column=3, padx=(5,10), pady=5, sticky="e")

        launch_buttons_frame = ctk.CTkFrame(self.model_selection_frame)
        launch_buttons_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        launch_buttons_frame.grid_columnconfigure((0,1,2), weight=1) # Distribute space among buttons
        ctk.CTkButton(launch_buttons_frame, text="Start Auto-Tune / Use OT Strategy", command=self.start_tuning_session, height=35, fg_color="seagreen", hover_color="darkgreen").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(launch_buttons_frame, text="Launch Best Remembered Config", command=self.launch_best_remembered, height=35, fg_color="cornflowerblue", hover_color="royalblue").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(launch_buttons_frame, text="Direct Launch (Settings Defaults)", command=self.launch_direct_defaults, height=35).grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        stop_button_frame = ctk.CTkFrame(self.model_selection_frame)
        stop_button_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=(0,5), sticky="ew")
        stop_button_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(stop_button_frame, text="Stop Any KCPP Processes", command=self.stop_all_kcpp_processes_forcefully, height=35, fg_color="firebrick", hover_color="darkred").grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        console_frame_ms = ctk.CTkFrame(self.model_selection_frame) # Console for model selection view
        console_frame_ms.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        console_frame_ms.grid_columnconfigure(0, weight=1); console_frame_ms.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(console_frame_ms, text="Launcher Log:").grid(row=0, column=0, padx=10, pady=(5,0), sticky="w")
        self.console = ctk.CTkTextbox(console_frame_ms, height=100) # Min height for console
        self.console.grid(row=1, column=0, padx=10, pady=(0,5), sticky="nsew")
        self.console.configure(state="disabled")

        # Tuning Mode View (initially hidden)
        self.tuning_mode_frame = ctk.CTkFrame(self.tab_main) # Parent is tab_main
        # NO grid() call here initially for tuning_mode_frame
        self.tuning_mode_frame.grid_columnconfigure(0, weight=1) # Allow content to expand width-wise
        self.tuning_mode_frame.grid_rowconfigure(7, weight=1) # Allow KCPP output console to expand

        tuning_title_label = ctk.CTkLabel(self.tuning_mode_frame, text="Auto-Tuning Session", font=ctk.CTkFont(size=18, weight="bold"))
        tuning_title_label.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        ot_strategy_display_frame = ctk.CTkFrame(self.tuning_mode_frame)
        ot_strategy_display_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ot_strategy_display_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(ot_strategy_display_frame, text="Current OT Strategy:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=5, pady=2, sticky="w")
        self.tuning_ot_level_label = ctk.CTkLabel(ot_strategy_display_frame, text="Level: N/A", justify="left")
        self.tuning_ot_level_label.grid(row=1, column=0, padx=10, pady=1, sticky="w")
        self.tuning_ot_range_label = ctk.CTkLabel(ot_strategy_display_frame, text="Range: N/A", justify="left")
        self.tuning_ot_range_label.grid(row=2, column=0, padx=10, pady=1, sticky="w")
        self.tuning_ot_strategy_desc_label = ctk.CTkLabel(ot_strategy_display_frame, text="Strategy: N/A", justify="left", wraplength=600)
        self.tuning_ot_strategy_desc_label.grid(row=3, column=0, columnspan=2, padx=10, pady=1, sticky="w")
        self.tuning_ot_regex_label = ctk.CTkLabel(ot_strategy_display_frame, text="Regex: N/A", justify="left", wraplength=600)
        self.tuning_ot_regex_label.grid(row=4, column=0, columnspan=2, padx=10, pady=1, sticky="w")
        self.tuning_gpu_layers_label = ctk.CTkLabel(ot_strategy_display_frame, text="GPU Layers: N/A", justify="left", wraplength=600)
        self.tuning_gpu_layers_label.grid(row=5, column=0, columnspan=2, padx=10, pady=1, sticky="w")

        proposed_command_frame = ctk.CTkFrame(self.tuning_mode_frame)
        proposed_command_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        proposed_command_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(proposed_command_frame, text="Proposed Command:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.tuning_proposed_command_text = ctk.CTkTextbox(proposed_command_frame, height=120, wrap="word")
        self.tuning_proposed_command_text.grid(row=1, column=0, padx=5, pady=2, sticky="nsew")
        self.tuning_proposed_command_text.configure(state="disabled")

        self.tuning_actions_primary_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.tuning_actions_primary_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        self.tuning_actions_primary_frame.grid_columnconfigure((0,1), weight=1)
        self.btn_tune_launch_monitor = ctk.CTkButton(self.tuning_actions_primary_frame, text="Launch & Monitor Output", command=self.launch_and_monitor_for_tuning, height=35, fg_color="seagreen", hover_color="darkgreen")
        self.btn_tune_launch_monitor.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.btn_tune_skip_launch_direct = ctk.CTkButton(self.tuning_actions_primary_frame, text="Skip Tune & Launch This Config", command=self.skip_tune_and_launch_direct, height=35)
        self.btn_tune_skip_launch_direct.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.tuning_actions_secondary_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.tuning_actions_secondary_frame.grid(row=4, column=0, padx=10, pady=0, sticky="ew") # pady reduced
        self.tuning_actions_secondary_frame.grid_columnconfigure((0,1,2), weight=1)
        self.btn_tune_more_gpu = ctk.CTkButton(self.tuning_actions_secondary_frame, text="More GPU (↓ Level)", command=lambda: self.adjust_ot_level(-1))
        self.btn_tune_more_gpu.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.btn_tune_more_cpu = ctk.CTkButton(self.tuning_actions_secondary_frame, text="More CPU (↑ Level)", command=lambda: self.adjust_ot_level(1))
        self.btn_tune_more_cpu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.btn_tune_edit_args = ctk.CTkButton(self.tuning_actions_secondary_frame, text="Edit Base Args (Session)", command=self.edit_base_args_for_tuning_session)
        self.btn_tune_edit_args.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.tuning_model_config_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.tuning_model_config_frame.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
        self.tuning_model_config_frame.grid_columnconfigure(0, weight=1) # Single button, full width
        self.btn_tune_edit_model_perm_args = ctk.CTkButton(self.tuning_model_config_frame, text="Edit Base Args (Permanent for This Model)", command=self.edit_permanent_model_args)
        self.btn_tune_edit_model_perm_args.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.tuning_actions_navigation_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.tuning_actions_navigation_frame.grid(row=6, column=0, padx=10, pady=5, sticky="ew")
        self.tuning_actions_navigation_frame.grid_columnconfigure((0,1,2), weight=1)
        self.btn_tune_new_gguf = ctk.CTkButton(self.tuning_actions_navigation_frame, text="New GGUF Model", command=self.select_new_gguf_during_tuning)
        self.btn_tune_new_gguf.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.btn_tune_history = ctk.CTkButton(self.tuning_actions_navigation_frame, text="View History", command=lambda: self.tabview.set("History"))
        self.btn_tune_history.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.btn_tune_quit_tuning = ctk.CTkButton(self.tuning_actions_navigation_frame, text="End Tuning Session", command=self.end_tuning_session, fg_color="firebrick", hover_color="darkred")
        self.btn_tune_quit_tuning.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        self.kcpp_output_console_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.kcpp_output_console_frame.grid(row=7, column=0, padx=10, pady=10, sticky="nsew")
        self.kcpp_output_console_frame.grid_columnconfigure(0, weight=1); self.kcpp_output_console_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(self.kcpp_output_console_frame, text="KoboldCpp Output (during monitoring):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.kcpp_live_output_text = ctk.CTkTextbox(self.kcpp_output_console_frame, wrap="char") # wrap="none" or "char"
        self.kcpp_live_output_text.grid(row=1, column=0, padx=5, pady=2, sticky="nsew")
        self.kcpp_live_output_text.configure(state="disabled")

    def _show_model_selection_view(self):
        if hasattr(self, 'tuning_mode_frame') and self.tuning_mode_frame.winfo_exists():
            self.tuning_mode_frame.grid_remove()
        if hasattr(self, 'model_selection_frame') and self.model_selection_frame.winfo_exists():
            self.model_selection_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.log_to_console("Switched to Model Selection view.")

    def _show_tuning_mode_view(self):
        if hasattr(self, 'model_selection_frame') and self.model_selection_frame.winfo_exists():
            self.model_selection_frame.grid_remove()
        if hasattr(self, 'tuning_mode_frame') and self.tuning_mode_frame.winfo_exists():
            self.tuning_mode_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.log_to_console("Switched to Tuning Mode view.")

    def start_tuning_session(self):
        if not self.current_model_path:
            messagebox.showwarning("No Model Selected", "Please select a GGUF model first.")
            return
        if not self.model_analysis_info or 'filepath' not in self.model_analysis_info:
            self.log_to_console("Model analysis info is missing. Re-analyzing...")
            self.analyze_model_action(self.current_model_path)
            if not self.model_analysis_info or 'filepath' not in self.model_analysis_info:
                messagebox.showerror("Model Error", "Failed to analyze model. Cannot start tuning.")
                return

        self.log_to_console(f"Starting tuning session for: {os.path.basename(self.current_model_path)}")
        self.tuning_in_progress = True
        self.current_tuning_model_path = self.current_model_path
        self.current_tuning_model_analysis = self.model_analysis_info.copy()
        self._reinitialize_session_base_args()

        if self.current_tuning_model_analysis.get('is_moe'):
            self.current_tuning_min_level, self.current_tuning_max_level, initial_heuristic_level = -25, 10, -10 # Expanded MoE range
        else:
            self.current_tuning_min_level, self.current_tuning_max_level = -17, 9 # Expanded Dense range
            size_b = self.current_tuning_model_analysis.get('size_b', 0)
            if isinstance(size_b, (int, float)):
                if size_b >= 30: initial_heuristic_level = -3
                elif size_b >= 20: initial_heuristic_level = -5
                else: initial_heuristic_level = -7
            else: initial_heuristic_level = -5

        current_vram, _, _, _ = koboldcpp_core.get_available_vram_mb()
        best_hist_config = koboldcpp_core.find_best_historical_config(self.db_path, self.current_tuning_model_analysis, current_vram)

        if best_hist_config and "attempt_level" in best_hist_config:
            self.log_to_console(f"Found historical config. Level: {best_hist_config['attempt_level']}, Outcome: {best_hist_config['outcome']}")
            hist_level, hist_outcome = best_hist_config['attempt_level'], best_hist_config.get('outcome', "")
            if hist_outcome.startswith("SUCCESS_LOAD_VRAM_OK") or hist_outcome.startswith("SUCCESS_USER_CONFIRMED_AUTO_OK") or hist_outcome.endswith("_USER_SAVED_GOOD_MORE_GPU"):
                initial_heuristic_level = max(self.current_tuning_min_level, hist_level - 1) # Start one level lower (more GPU)
            elif hist_outcome.endswith("_USER_ACCEPTED_AUTO_ADJUST_CPU") or hist_outcome.endswith("_USER_TRIED_MORE_CPU_AFTER_FAIL"):
                 initial_heuristic_level = min(self.current_tuning_max_level, hist_level + 1) # Start one level higher (more CPU)
            else: initial_heuristic_level = hist_level
            self.log_to_console(f"Adapted historical level to start at: {initial_heuristic_level}")
            # Apply args from historical config to session base
            remembered_args_list = best_hist_config.get("args_list", [])
            if remembered_args_list:
                remembered_args_dict = koboldcpp_core.args_list_to_dict(remembered_args_list)
                remembered_args_dict.pop("--model", None) # Don't override current model
                remembered_args_dict.pop("--overridetensors", None) # OT string is generated by level
                self.current_tuning_session_base_args.update(remembered_args_dict)
                self.log_to_console("Applied remembered arguments to current session base.")

        else:
            self.log_to_console(f"No suitable historical config. Starting with heuristic level: {initial_heuristic_level}")

        self.current_tuning_attempt_level = max(self.current_tuning_min_level, min(initial_heuristic_level, self.current_tuning_max_level))
        self._show_tuning_mode_view()
        self.update_tuning_display()
        if hasattr(self, 'kcpp_live_output_text') and self.kcpp_live_output_text.winfo_exists():
            self.kcpp_live_output_text.configure(state="normal")
            self.kcpp_live_output_text.delete("1.0", "end")
            self.kcpp_live_output_text.configure(state="disabled")

    def update_tuning_display(self):
        if not self.tuning_in_progress: return
        self.current_tuning_attempt_level = max(self.current_tuning_min_level, min(self.current_tuning_attempt_level, self.current_tuning_max_level))
        ot_string = koboldcpp_core.generate_overridetensors(self.current_tuning_model_analysis, self.current_tuning_attempt_level)
        description = koboldcpp_core.get_offload_description(self.current_tuning_model_analysis, self.current_tuning_attempt_level, ot_string)
    
        gpu_layers = koboldcpp_core.get_gpu_layers_for_level(self.current_tuning_model_analysis, self.current_tuning_attempt_level)
        total_layers = self.current_tuning_model_analysis.get('num_layers', 32)
    
        if hasattr(self, 'tuning_ot_level_label') and self.tuning_ot_level_label.winfo_exists():
            self.tuning_ot_level_label.configure(text=f"Level: {self.current_tuning_attempt_level}")
    
        range_text_suffix = 'SuperMaxCPU' if self.current_tuning_model_analysis.get('is_moe') else 'MaxCPU'
        range_text = f"{self.current_tuning_min_level}=MaxGPU ... {self.current_tuning_max_level}={range_text_suffix}"
        if hasattr(self, 'tuning_ot_range_label') and self.tuning_ot_range_label.winfo_exists():
            self.tuning_ot_range_label.configure(text=f"Range: {range_text}")
    
        if hasattr(self, 'tuning_ot_strategy_desc_label') and self.tuning_ot_strategy_desc_label.winfo_exists():
            self.tuning_ot_strategy_desc_label.configure(text=f"Strategy: {description}")
    
        regex_display = ot_string or 'None (Max GPU layers if --gpulayers is high)'
        if hasattr(self, 'tuning_ot_regex_label') and self.tuning_ot_regex_label.winfo_exists():
            self.tuning_ot_regex_label.configure(text=f"Regex: {regex_display}")
    
        if hasattr(self, 'tuning_gpu_layers_label') and self.tuning_gpu_layers_label.winfo_exists():
            self.tuning_gpu_layers_label.configure(text=f"GPU Layers: {gpu_layers}/{total_layers}")
    
        args_for_kcpp_list = koboldcpp_core.build_command(
            self.current_tuning_model_path, 
            ot_string, 
            self.current_tuning_model_analysis, 
            self.current_tuning_session_base_args
        )
        full_command_list = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, args_for_kcpp_list)
        display_command_str = koboldcpp_core.format_command_for_display(full_command_list)
    
        if hasattr(self, 'tuning_proposed_command_text') and self.tuning_proposed_command_text.winfo_exists():
            self.tuning_proposed_command_text.configure(state="normal")
            self.tuning_proposed_command_text.delete("1.0", "end")
            self.tuning_proposed_command_text.insert("1.0", display_command_str)
            self.tuning_proposed_command_text.configure(state="disabled")
    
        if hasattr(self, 'btn_tune_more_gpu') and self.btn_tune_more_gpu.winfo_exists():
            self.btn_tune_more_gpu.configure(state="normal" if self.current_tuning_attempt_level > self.current_tuning_min_level else "disabled")
    
        if hasattr(self, 'btn_tune_more_cpu') and self.btn_tune_more_cpu.winfo_exists():
            self.btn_tune_more_cpu.configure(state="normal" if self.current_tuning_attempt_level < self.current_tuning_max_level else "disabled")
    
        self.log_to_console(f"Tuning display updated for OT Level: {self.current_tuning_attempt_level}")

    def adjust_ot_level(self, delta):
        if not self.tuning_in_progress: return
        self.current_tuning_attempt_level += delta
        self.update_tuning_display()

    def _create_args_dialog_content(self, parent_dialog, current_args_dict_for_dialog, param_definitions_list):
        scrollable_frame = ctk.CTkScrollableFrame(parent_dialog)
        scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)
        temp_arg_widgets = {}
        for setting_def in param_definitions_list:
            param = setting_def["param"]
            current_val_for_field = current_args_dict_for_dialog.get(param) # This is the effective value
            
            frame = ctk.CTkFrame(scrollable_frame); frame.pack(fill="x", pady=2)
            ctk.CTkLabel(frame, text=f"{setting_def['name']}:", width=160, anchor="w").pack(side="left", padx=5) # Increased width
            
            # Determine if the parameter is a boolean type from core defaults
            # This check needs to be robust, as some bools might be 'True'/'False' strings initially
            core_default_val = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].get(param)
            is_bool_default_type = isinstance(core_default_val, bool) or \
                                   (isinstance(core_default_val, str) and core_default_val.lower() in ['true', 'false']) or \
                                   setting_def.get("type") == "bool" # Explicit type from definition
            
            if is_bool_default_type:
                # Determine initial checkbox state from current_val_for_field
                bool_val_to_set = False
                if isinstance(current_val_for_field, bool):
                    bool_val_to_set = current_val_for_field
                elif isinstance(current_val_for_field, str):
                    bool_val_to_set = current_val_for_field.lower() == 'true'
                
                var = ctk.BooleanVar(value=bool_val_to_set)
                widget = ctk.CTkCheckBox(frame, text="", variable=var); widget.pack(side="left", padx=5)
                widget.variable = var; temp_arg_widgets[param] = {"widget": widget, "type": "bool"}
            else: # Entry for string/number/auto
                entry = ctk.CTkEntry(frame, width=150) # Set a fixed width for entries
                if current_val_for_field is not None:
                    entry.insert(0, str(current_val_for_field))
                entry.pack(side="left", padx=5, fill="x", expand=False); # expand=False for fixed width
                temp_arg_widgets[param] = {"widget": entry, "type": "str_num"}
            
            # Add help text label
            ctk.CTkLabel(frame, text=setting_def.get("help", ""), font=ctk.CTkFont(size=10), text_color="gray").pack(side="left", padx=5, fill="x", expand=True)

        return temp_arg_widgets


    def _get_param_definitions_for_dialog(self):
        # Centralized list for dialogs, matching settings tab order as much as possible
        return [
            {"name": "Threads", "param": "--threads", "help": "CPU threads ('auto' or number)."},
            {"name": "BLAS Threads (nblas)", "param": "--nblas", "help": "BLAS threads ('auto' or number)."},
            {"name": "Context Size", "param": "--contextsize", "help": "Max context tokens (e.g., 4096, 16384)."},
            {"name": "Prompt Limit", "param": "--promptlimit", "help": "Max prompt tokens (<= contextsize)."},
            {"name": "GPU Layers", "param": "--gpulayers", "help": "Layers on GPU. 'off', 0 for CPU. 999 for max."},
            {"name": "Use CUBLAS", "param": "--usecublas", "help": "Enable CUDA BLAS (NVIDIA).", "type": "bool"},
            {"name": "Flash Attention", "param": "--flashattention", "help": "Enable FlashAttention.", "type": "bool"},
            {"name": "No Memory Map", "param": "--nommap", "help": "Disable memory mapping.", "type": "bool"},
            {"name": "Low VRAM Mode", "param": "--lowvram", "help": "Enable low VRAM mode.", "type": "bool"},
            {"name": "QuantKV", "param": "--quantkv", "help": "K/V cache quant ('auto', 'off', or number)."},
            {"name": "BLAS Batch Size", "param": "--blasbatchsize", "help": "BLAS batch size ('auto', 'off', or number)."},
            {"name": "Port", "param": "--port", "help": "Web UI port (default 5000)."},
            {"name": "Default Gen Amount", "param": "--defaultgenamt", "help": "Default tokens to generate."}
        ]

    def edit_base_args_for_tuning_session(self):
        if not self.tuning_in_progress: return
        dialog = ctk.CTkToplevel(self); dialog.title("Edit Base Arguments for This Session")
        dialog.geometry("750x650"); dialog.transient(self); dialog.grab_set() # Increased size
        
        param_definitions = self._get_param_definitions_for_dialog()
        
        full_merged_for_display = self._get_merged_args_for_model(self.current_tuning_model_path)
        display_args_in_dialog = full_merged_for_display.copy()
        display_args_in_dialog.update(self.current_tuning_session_base_args) # Apply session overrides on top for display
        
        temp_arg_widgets_info = self._create_args_dialog_content(dialog, display_args_in_dialog, param_definitions)
        
        def save_session_args_action():
            for param, info_dict in temp_arg_widgets_info.items():
                widget = info_dict["widget"]
                widget_type = info_dict["type"]
                if widget_type == "str_num": # Entry
                    val_str = widget.get().strip()
                    if val_str: self.current_tuning_session_base_args[param] = val_str
                    elif param in self.current_tuning_session_base_args: del self.current_tuning_session_base_args[param] # Unset if empty
                elif widget_type == "bool": # CheckBox
                    if widget.variable.get(): self.current_tuning_session_base_args[param] = True
                    else: # If unchecked, it means False for boolean flags
                        self.current_tuning_session_base_args[param] = False 
                        # Or, if you want unchecking to revert to global/default, then:
                        # if param in self.current_tuning_session_base_args: del self.current_tuning_session_base_args[param]
            self.log_to_console("Base arguments for this tuning session updated.")
            self.update_tuning_display(); dialog.destroy()
            
        button_frame = ctk.CTkFrame(dialog); button_frame.pack(fill="x", pady=10)
        ctk.CTkButton(button_frame, text="Save Session Args", command=save_session_args_action).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Cancel", command=dialog.destroy).pack(side="right", padx=10)
        dialog.wait_window()

    def edit_permanent_model_args(self):
        if not self.tuning_in_progress or not self.current_tuning_model_path:
            messagebox.showwarning("Action Not Available", "Start tuning with a selected model to edit its permanent arguments.")
            return
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"Edit Permanent Args for: {os.path.basename(self.current_tuning_model_path)}")
        dialog.geometry("750x650"); dialog.transient(self); dialog.grab_set() # Increased size
        
        param_definitions = self._get_param_definitions_for_dialog()
        
        # For permanent edits, we display the currently effective args (global + model-specific)
        args_to_display_in_dialog = self._get_merged_args_for_model(self.current_tuning_model_path)
        temp_arg_widgets_info = self._create_args_dialog_content(dialog, args_to_display_in_dialog, param_definitions)
        
        def save_permanent_args_action():
            model_path_key = self.current_tuning_model_path
            if model_path_key not in self.config["model_specific_args"]:
                self.config["model_specific_args"][model_path_key] = {}
            current_model_specifics = self.config["model_specific_args"][model_path_key]
            
            # Get baseline (Core Defaults + Global Config Defaults), NO session args
            global_baseline_args = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
            global_baseline_args.update(self.config.get("default_args", {}))

            for param, info_dict in temp_arg_widgets_info.items():
                widget = info_dict["widget"]
                widget_type = info_dict["type"]
                
                current_global_baseline_for_param = global_baseline_args.get(param)
                
                if widget_type == "str_num": # Entry
                    val_str_from_dialog = widget.get().strip()
                    if not val_str_from_dialog: # Empty input means revert to global baseline
                        if param in current_model_specifics: del current_model_specifics[param]
                    else:
                        # Convert to baseline type for comparison if possible (e.g. bools as strings)
                        global_baseline_str = str(current_global_baseline_for_param) if current_global_baseline_for_param is not None else ""
                        if isinstance(current_global_baseline_for_param, bool): global_baseline_str = str(current_global_baseline_for_param).lower()
                        
                        if val_str_from_dialog.lower() != global_baseline_str:
                            current_model_specifics[param] = val_str_from_dialog
                        elif param in current_model_specifics: # Value matches global, remove specific
                            del current_model_specifics[param]
                elif widget_type == "bool": # CheckBox
                    val_bool_from_dialog = widget.variable.get()
                    global_baseline_bool = False
                    if isinstance(current_global_baseline_for_param, bool): global_baseline_bool = current_global_baseline_for_param
                    elif isinstance(current_global_baseline_for_param, str): global_baseline_bool = current_global_baseline_for_param.lower() == 'true'
                    
                    if val_bool_from_dialog != global_baseline_bool:
                        current_model_specifics[param] = val_bool_from_dialog
                    elif param in current_model_specifics: # Value matches global, remove specific
                        del current_model_specifics[param]

            if not self.config["model_specific_args"][model_path_key]: # Clean up if no specifics left
                del self.config["model_specific_args"][model_path_key]
                
            if self.save_config():
                self.log_to_console(f"Permanent arguments saved for model: {os.path.basename(model_path_key)}")
                self._reinitialize_session_base_args() # Session args should now reflect the new permanent base
                self.update_tuning_display()
            else: self.log_to_console(f"Failed to save permanent arguments for model.")
            dialog.destroy()
            
        button_frame = ctk.CTkFrame(dialog); button_frame.pack(fill="x", pady=10)
        ctk.CTkButton(button_frame, text="Save Permanent Args", command=save_permanent_args_action).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Cancel", command=dialog.destroy).pack(side="right", padx=10)
        dialog.wait_window()

    def select_new_gguf_during_tuning(self):
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
            if not messagebox.askyesno("Process Running", "A KCPP monitoring process might be running. Stop it and select new model?"): return
        self.end_tuning_session(switch_to_model_selection=True); self.browse_model()

    def end_tuning_session(self, switch_to_model_selection=True):
        if not self.tuning_in_progress: return
        self.log_to_console("Ending tuning session.")
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
            self.log_to_console("Stopping active KCPP process from tuning session...")
            success, msg = koboldcpp_core.kill_process(self.kcpp_process_obj.pid, force=True)
            self.log_to_console(f"Kill attempt (PID {self.kcpp_process_obj.pid}): {msg}"); self.kcpp_process_obj = None
        self.tuning_in_progress = False; self.current_tuning_model_path = None
        self.current_tuning_model_analysis = {}; self.current_tuning_session_base_args = {}
        self.last_free_vram_after_load_mb = None
        if switch_to_model_selection: self._show_model_selection_view()

    def launch_and_monitor_for_tuning(self):
        if not self.tuning_in_progress: 
            messagebox.showerror("Error", "Tuning session not active.")
            return
    
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None: 
            messagebox.showwarning("Process Running", "A KoboldCpp process is already being monitored. Please stop it first or wait.")
            return
    
        self.log_to_console(f"Tuning: Launching & Monitoring for OT Level {self.current_tuning_attempt_level}")
        if hasattr(self, 'kcpp_live_output_text') and self.kcpp_live_output_text.winfo_exists():
            self.kcpp_live_output_text.configure(state="normal")
            self.kcpp_live_output_text.delete("1.0", "end")
            self.kcpp_live_output_text.insert("1.0", "Preparing to launch KoboldCpp for monitoring...\n")
            self.kcpp_live_output_text.configure(state="disabled")
    
        self._set_tuning_buttons_state("disabled")
        self.kcpp_success_event.clear()
        self.kcpp_oom_event.clear()
        self.kcpp_output_lines_shared.clear()
        self.last_free_vram_after_load_mb = None
        self.level_of_last_monitored_run = self.current_tuning_attempt_level

        ot_string = koboldcpp_core.generate_overridetensors(self.current_tuning_model_analysis, self.current_tuning_attempt_level)
        args_for_kcpp_list = koboldcpp_core.build_command(
            self.current_tuning_model_path, ot_string, self.current_tuning_model_analysis, self.current_tuning_session_base_args
        )
        
        self.current_command_list_for_db = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, args_for_kcpp_list)
        self.vram_at_decision_for_db, _, _, _ = koboldcpp_core.get_available_vram_mb()
        
        self.kcpp_process_obj, error_msg = koboldcpp_core.launch_process(
            self.current_command_list_for_db, capture_output=True, new_console=False, use_text_mode=False
        )

        if error_msg or not self.kcpp_process_obj:
            self.log_to_console(f"Failed to launch KCPP for monitoring: {error_msg or 'Unknown error'}")
            self._log_to_kcpp_live_output(f"LAUNCH ERROR: {error_msg or 'Unknown error'}\n")
            koboldcpp_core.save_config_to_db(self.db_path, self.current_tuning_model_path, self.current_tuning_model_analysis,
                                           self.vram_at_decision_for_db, self.current_command_list_for_db,
                                           self.level_of_last_monitored_run, "LAUNCH_FAILED_SETUP_GUI")
            self._set_tuning_buttons_state("normal")
            return
        
        self._log_to_kcpp_live_output(f"KoboldCpp process started (PID: {self.kcpp_process_obj.pid}). Monitoring output...\n")
        target_port = self.current_tuning_session_base_args.get("--port", self.config["default_args"].get("--port", "5000"))
        success_pattern = self.config.get("kobold_success_pattern", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["kobold_success_pattern"])
        oom_keywords = self.config.get("oom_error_keywords", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["oom_error_keywords"])
        
        self.kcpp_monitor_thread = threading.Thread(
            target=self._monitor_kcpp_output_thread_target,
            args=(self.kcpp_process_obj, success_pattern, oom_keywords, target_port),
            daemon=True
        )
        
        self.kcpp_monitor_thread.start()
        self.monitor_start_time = time.monotonic()
        self._poll_monitor_status()

    def _log_to_kcpp_live_output(self, text):
        def _log_update():
            if hasattr(self, 'kcpp_live_output_text') and self.kcpp_live_output_text.winfo_exists():
                self.kcpp_live_output_text.configure(state="normal"); self.kcpp_live_output_text.insert("end", text)
                self.kcpp_live_output_text.see("end"); self.kcpp_live_output_text.configure(state="disabled")
        if hasattr(self, 'after'): self.after(0, _log_update)

    def _monitor_kcpp_output_thread_target(self, process, success_regex_config, oom_keywords_config, target_port):
        try:
            for line_bytes in iter(process.stdout.readline, b''):
                if not line_bytes: break
                try: line_decoded = line_bytes.decode('utf-8', errors='replace')
                except UnicodeDecodeError: line_decoded = line_bytes.decode('latin-1', errors='replace')
                self._log_to_kcpp_live_output(line_decoded); line_strip = line_decoded.strip()
                if line_strip:
                    self.kcpp_output_lines_shared.append(line_strip); line_lower = line_strip.lower()
                    if not self.kcpp_success_event.is_set() and not self.kcpp_oom_event.is_set():
                        success_match = re.search(success_regex_config, line_strip, re.IGNORECASE)
                        if success_match:
                            launched_port_str = target_port
                            try: launched_port_str = success_match.group(1)
                            except IndexError: pass
                            if str(launched_port_str) == str(target_port): self.kcpp_success_event.set()
                        if not self.kcpp_success_event.is_set():
                            for keyword in oom_keywords_config:
                                if keyword.lower() in line_lower: self.kcpp_oom_event.set(); break
                if self.kcpp_success_event.is_set() or self.kcpp_oom_event.is_set(): break
        except Exception as e: self._log_to_kcpp_live_output(f"\nError in KCPP monitor thread: {type(e).__name__}: {e}\n")
        finally:
            if process.stdout and not process.stdout.closed:
                try: process.stdout.close()
                except: pass
            self._log_to_kcpp_live_output("\nKCPP output monitoring thread finished.\n")

    def _poll_monitor_status(self):
        timeout_seconds = self.config.get("loading_timeout_seconds", 60); elapsed_time = time.monotonic() - self.monitor_start_time
        process_exited = self.kcpp_process_obj.poll() is not None
        if self.kcpp_success_event.is_set(): self._handle_monitoring_completion("SUCCESS_LOAD_DETECTED_GUI")
        elif self.kcpp_oom_event.is_set(): self._handle_monitoring_completion("OOM_CRASH_DETECTED_GUI")
        elif process_exited: self._handle_monitoring_completion("PREMATURE_EXIT_GUI")
        elif elapsed_time > timeout_seconds: self._handle_monitoring_completion("TIMEOUT_NO_SIGNAL_GUI")
        else: self.after(250, self._poll_monitor_status)

    def _handle_monitoring_completion(self, initial_outcome_key):
        self.log_to_console(f"Monitoring completed. Initial Outcome: {initial_outcome_key}")
        self._log_to_kcpp_live_output(f"\n--- Monitoring Finished: {initial_outcome_key} ---\n")
        
        if initial_outcome_key in ["TIMEOUT_NO_SIGNAL_GUI", "OOM_CRASH_DETECTED_GUI", "PREMATURE_EXIT_GUI"] or \
           "OOM" in initial_outcome_key or "CRASH" in initial_outcome_key:
            if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
                self.log_to_console("Terminating KCPP process due to unfavorable outcome...")
                koboldcpp_core.kill_process(self.kcpp_process_obj.pid, force=True)
                self.kcpp_process_obj = None
        
        self.last_approx_vram_used_kcpp_mb = None
        self.last_free_vram_after_load_mb = None
        final_db_outcome = initial_outcome_key
        
        if initial_outcome_key == "SUCCESS_LOAD_DETECTED_GUI":
            self._log_to_kcpp_live_output("API detected. Waiting a few seconds for VRAM to stabilize...\n")
            time.sleep(5)
            current_free_vram, total_vram, _, _ = koboldcpp_core.get_available_vram_mb()
            self.last_free_vram_after_load_mb = current_free_vram
            
            if total_vram > 0 and hasattr(self, 'vram_at_decision_for_db') and self.vram_at_decision_for_db is not None:
                self.last_approx_vram_used_kcpp_mb = self.vram_at_decision_for_db - current_free_vram
                self.last_approx_vram_used_kcpp_mb = max(0, min(self.last_approx_vram_used_kcpp_mb, total_vram))
                self._log_to_kcpp_live_output(f"VRAM after load: {current_free_vram:.0f}MB free. Approx KCPP usage: {self.last_approx_vram_used_kcpp_mb:.0f}MB\n")
                min_vram_free_target = self.config.get("min_vram_free_after_load_success_mb", 512)
                
                if current_free_vram < min_vram_free_target:
                    self._log_to_kcpp_live_output(f"WARNING: VRAM tight! {current_free_vram:.0f}MB < {min_vram_free_target}MB target.\n")
                    final_db_outcome = "SUCCESS_LOAD_VRAM_TIGHT_GUI"
                else:
                    self._log_to_kcpp_live_output("VRAM usage OK.\n")
                    final_db_outcome = "SUCCESS_LOAD_VRAM_OK_GUI"
            else:
                final_db_outcome = "SUCCESS_LOAD_NO_VRAM_CHECK_GUI"
        
        koboldcpp_core.save_config_to_db(
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

    def _set_tuning_buttons_state(self, state="normal"):
        buttons_to_manage = [
            getattr(self, 'btn_tune_launch_monitor', None), 
            getattr(self, 'btn_tune_skip_launch_direct', None), 
            getattr(self, 'btn_tune_more_gpu', None), 
            getattr(self, 'btn_tune_more_cpu', None), 
            getattr(self, 'btn_tune_edit_args', None), 
            getattr(self, 'btn_tune_edit_model_perm_args', None), 
            getattr(self, 'btn_tune_new_gguf', None), 
            getattr(self, 'btn_tune_quit_tuning', None)
        ]
        for btn in buttons_to_manage:
            if btn and hasattr(btn, 'winfo_exists') and btn.winfo_exists(): 
                btn.configure(state=state)
        
        if state == "normal": # Re-apply conditional disabled states
            if hasattr(self, 'btn_tune_more_gpu') and self.btn_tune_more_gpu.winfo_exists():
                self.btn_tune_more_gpu.configure(state="normal" if self.current_tuning_attempt_level > self.current_tuning_min_level else "disabled")
            if hasattr(self, 'btn_tune_more_cpu') and self.btn_tune_more_cpu.winfo_exists():
                self.btn_tune_more_cpu.configure(state="normal" if self.current_tuning_attempt_level < self.current_tuning_max_level else "disabled")
            if hasattr(self, 'btn_tune_edit_model_perm_args') and self.btn_tune_edit_model_perm_args.winfo_exists():
                 self.btn_tune_edit_model_perm_args.configure(state="normal" if self.tuning_in_progress and self.current_tuning_model_path else "disabled")

    def _present_post_monitoring_choices(self, outcome):
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
            self.log_to_console("Auto-killing monitored KCPP instance after assessment before showing choices.")
            koboldcpp_core.kill_process(self.kcpp_process_obj.pid, force=True); self.kcpp_process_obj = None
        
        frames_to_hide_names = ['tuning_actions_primary_frame', 'tuning_actions_secondary_frame', 
                                'tuning_model_config_frame', 'tuning_actions_navigation_frame']
        for frame_name in frames_to_hide_names:
            frame = getattr(self, frame_name, None)
            if frame and hasattr(frame, 'winfo_exists') and frame.winfo_exists():
                frame.grid_remove()

        if not hasattr(self, 'post_monitor_choices_frame'):
            self.post_monitor_choices_frame = ctk.CTkFrame(self.tuning_mode_frame)
            # Ensure it's placed correctly within tuning_mode_frame's grid
            self.post_monitor_choices_frame.grid(row=3, column=0, rowspan=4, padx=10, pady=5, sticky="news") 
            self.post_monitor_choices_frame.grid_columnconfigure(0, weight=1)
        
        self.post_monitor_choices_frame.grid() # Make sure it's visible

        for widget in self.post_monitor_choices_frame.winfo_children(): widget.destroy() # Clear previous choices

        ctk.CTkLabel(self.post_monitor_choices_frame, text=f"Outcome: {outcome}", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 2), anchor="w", padx=5)
        if self.last_free_vram_after_load_mb is not None:
            ctk.CTkLabel(self.post_monitor_choices_frame, text=f"VRAM After Load: {self.last_free_vram_after_load_mb:.0f} MB free").pack(pady=1, anchor="w", padx=5)
        if self.last_approx_vram_used_kcpp_mb is not None:
            ctk.CTkLabel(self.post_monitor_choices_frame, text=f"Approx. KCPP VRAM Used: {self.last_approx_vram_used_kcpp_mb:.0f} MB").pack(pady=1, anchor="w", padx=5)
        
        vram_status_text = "VRAM Status: Check KCPP Log"
        if "SUCCESS_LOAD_VRAM_OK" in outcome: vram_status_text = "VRAM Usage: OK"
        elif "SUCCESS_LOAD_VRAM_TIGHT" in outcome: vram_status_text = "VRAM Usage: TIGHT"
        elif "SUCCESS_LOAD_NO_VRAM_CHECK" in outcome: vram_status_text = "VRAM Usage: Not Checked"
        ctk.CTkLabel(self.post_monitor_choices_frame, text=vram_status_text).pack(pady=(1, 5), anchor="w", padx=5)

        if "SUCCESS_LOAD_VRAM_OK" in outcome:
            ctk.CTkButton(self.post_monitor_choices_frame, text="✅ Accept & Launch for Use", command=lambda: self._handle_post_monitor_action("launch_for_use", outcome)).pack(fill="x", pady=3, padx=5)
            ctk.CTkButton(self.post_monitor_choices_frame, text="💾 Save as Good & Continue Tuning (Try More GPU)", command=lambda: self._handle_post_monitor_action("save_good_more_gpu", outcome)).pack(fill="x", pady=3, padx=5)
            ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Try More GPU (This Session)", command=lambda: self._handle_post_monitor_action("more_gpu_now", outcome)).pack(fill="x", pady=3, padx=5)
            ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Try More CPU (This Session)", command=lambda: self._handle_post_monitor_action("more_cpu_now", outcome)).pack(fill="x", pady=3, padx=5)
        elif "SUCCESS_LOAD_VRAM_TIGHT" in outcome:
            ctk.CTkButton(self.post_monitor_choices_frame, text="⚠️ Auto-Adjust (More CPU & Continue)", command=lambda: self._handle_post_monitor_action("auto_adjust_cpu", outcome)).pack(fill="x", pady=3, padx=5)
            ctk.CTkButton(self.post_monitor_choices_frame, text="🚀 Launch Anyway (Risky)", command=lambda: self._handle_post_monitor_action("launch_for_use_risky", outcome)).pack(fill="x", pady=3, padx=5)
        elif "OOM" in outcome or "CRASH" in outcome or "PREMATURE_EXIT" in outcome :
            ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Try More CPU & Continue Tuning", command=lambda: self._handle_post_monitor_action("more_cpu_after_fail", outcome)).pack(fill="x", pady=3, padx=5)
        elif "TIMEOUT" in outcome:
            ctk.CTkLabel(self.post_monitor_choices_frame, text="Launch timed out. Manual check or try different settings.").pack(pady=3, anchor="w", padx=5)
            ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Try More CPU (Assume OOM)", command=lambda: self._handle_post_monitor_action("more_cpu_after_fail", outcome)).pack(fill="x", pady=3, padx=5)
        
        ctk.CTkButton(self.post_monitor_choices_frame, text="↩️ Save & Return to Tuning Menu", command=lambda: self._handle_post_monitor_action("return_to_tuning_menu", outcome)).pack(fill="x", pady=3, padx=5)
        self.log_to_console("Presented post-monitoring choices to user.")

    def _return_to_full_tuning_menu(self):
        if hasattr(self, 'post_monitor_choices_frame') and self.post_monitor_choices_frame.winfo_exists():
            self.post_monitor_choices_frame.grid_remove()
        
        frames_to_show_names = ['tuning_actions_primary_frame', 'tuning_actions_secondary_frame', 
                                'tuning_model_config_frame', 'tuning_actions_navigation_frame']
        for frame_name in frames_to_show_names:
            frame = getattr(self, frame_name, None)
            if frame and hasattr(frame, 'winfo_exists') and frame.winfo_exists():
                frame.grid() # Re-grid them to make them visible
        
        self._set_tuning_buttons_state("normal"); self.update_tuning_display()

    def _handle_post_monitor_action(self, action_key, original_db_outcome):
        self.log_to_console(f"User post-monitoring action: {action_key}")

        ot_string_for_next_launch = koboldcpp_core.generate_overridetensors(self.current_tuning_model_analysis, self.current_tuning_attempt_level)
        args_for_next_launch = koboldcpp_core.build_command(self.current_tuning_model_path, ot_string_for_next_launch, self.current_tuning_model_analysis, self.current_tuning_session_base_args)
        command_to_run_for_use = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, args_for_next_launch)

        new_db_outcome_suffix = "" # Suffix to append to original_db_outcome

        if action_key == "launch_for_use":
            new_db_outcome_suffix = "_USER_LAUNCHED"
            self._launch_final_koboldcpp(command_to_run_for_use, original_db_outcome + new_db_outcome_suffix, self.level_of_last_monitored_run)
            self.end_tuning_session()
            return
        if action_key == "launch_for_use_risky":
            new_db_outcome_suffix = "_USER_LAUNCHED_RISKY"
            self._launch_final_koboldcpp(command_to_run_for_use, original_db_outcome + new_db_outcome_suffix, self.level_of_last_monitored_run)
            self.end_tuning_session()
            return

        if action_key == "save_good_more_gpu":
            new_db_outcome_suffix = "_USER_SAVED_GOOD_MORE_GPU"
            if self.current_tuning_attempt_level > self.current_tuning_min_level:
                self.current_tuning_attempt_level -= 1
            else: self.log_to_console("Already at Max GPU, cannot go further.")
        elif action_key == "more_gpu_now":
            new_db_outcome_suffix = "_USER_WANTS_MORE_GPU_NOW"
            if self.current_tuning_attempt_level > self.current_tuning_min_level:
                self.current_tuning_attempt_level -= 1
            else: self.log_to_console("Already at Max GPU, cannot go further.")
        elif action_key in ["more_cpu_now", "auto_adjust_cpu", "more_cpu_after_fail"]:
            if action_key == "auto_adjust_cpu": new_db_outcome_suffix = "_USER_ACCEPTED_AUTO_ADJUST_CPU"
            elif action_key == "more_cpu_after_fail": new_db_outcome_suffix = "_USER_TRIED_MORE_CPU_AFTER_FAIL"
            else: new_db_outcome_suffix = "_USER_WANTS_MORE_CPU_NOW"
            if self.current_tuning_attempt_level < self.current_tuning_max_level:
                self.current_tuning_attempt_level += 1
            else: self.log_to_console("Already at Max CPU, cannot go further.")
        elif action_key == "return_to_tuning_menu":
            new_db_outcome_suffix = "_USER_RETURNED_TO_TUNING_MENU"

        final_db_outcome_to_save = original_db_outcome + new_db_outcome_suffix
        vram_decision_val = self.vram_at_decision_for_db if hasattr(self, 'vram_at_decision_for_db') else 0
        koboldcpp_core.save_config_to_db(
            self.db_path, self.current_tuning_model_path, self.current_tuning_model_analysis,
            vram_decision_val, self.current_command_list_for_db, 
            self.level_of_last_monitored_run, 
            final_db_outcome_to_save, self.last_approx_vram_used_kcpp_mb
        )
        self.load_history()
        self._return_to_full_tuning_menu()

    def _launch_final_koboldcpp(self, command_list, db_outcome_for_update, attempt_level_for_db_record):
        self.log_to_console(f"Launching KoboldCpp for use with command: {' '.join(command_list)}")
        vram_for_db = self.vram_at_decision_for_db if hasattr(self, 'vram_at_decision_for_db') and self.vram_at_decision_for_db is not None else \
                      koboldcpp_core.get_available_vram_mb()[0]
        koboldcpp_core.save_config_to_db(
            self.db_path, self.current_tuning_model_path, self.current_tuning_model_analysis,
            vram_for_db, command_list, attempt_level_for_db_record,
            db_outcome_for_update, self.last_approx_vram_used_kcpp_mb
        )
        self.load_history()
        process, error_msg = koboldcpp_core.launch_process(command_list, capture_output=False, new_console=True)
        if error_msg or not process:
            self.log_to_console(f"Failed to launch KoboldCPP for use: {error_msg or 'Unknown error'}")
            messagebox.showerror("Launch Error", f"Failed to launch KoboldCPP: {error_msg or 'Unknown error'}")
        else:
            self.log_to_console(f"KoboldCPP launched for use in new console (PID: {process.pid}).")
            self.last_process = process; self.process_running = True
            
            if self.config.get("auto_open_webui", True):
                args_dict_for_port = koboldcpp_core.args_list_to_dict(command_list)
                port_to_open = args_dict_for_port.get("--port", self.config.get("default_args", {}).get("--port", "5000"))
                try:
                    port_num = int(port_to_open)
                    if 1 <= port_num <= 65535:
                        import webbrowser
                        self.log_to_console(f"Auto-opening web UI at http://localhost:{port_num} based on setting.")
                        threading.Timer(3.0, lambda: webbrowser.open(f"http://localhost:{port_num}")).start()
                    else: self.log_to_console(f"Invalid port number '{port_to_open}' for auto-opening browser.")
                except ValueError: self.log_to_console(f"Invalid port value '{port_to_open}' for auto-opening browser.")
            else: self.log_to_console("Auto-open web UI is disabled in settings.")

    def skip_tune_and_launch_direct(self):
        if not self.tuning_in_progress: return
        self.log_to_console("User chose to skip further tuning and launch current config directly.")
        if not hasattr(self, 'vram_at_decision_for_db') or self.vram_at_decision_for_db is None:
             self.vram_at_decision_for_db = koboldcpp_core.get_available_vram_mb()[0]
        ot_string = koboldcpp_core.generate_overridetensors(self.current_tuning_model_analysis, self.current_tuning_attempt_level)
        args_list = koboldcpp_core.build_command(self.current_tuning_model_path, ot_string, self.current_tuning_model_analysis, self.current_tuning_session_base_args)
        command_to_run = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, args_list)
        self._launch_final_koboldcpp(command_to_run, "SUCCESS_USER_DIRECT_LAUNCH_GUI", self.current_tuning_attempt_level)
        self.end_tuning_session()

    def setup_settings_tab(self):
        settings_frame = ctk.CTkScrollableFrame(self.tab_settings)
        settings_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        theme_frame = ctk.CTkFrame(settings_frame)
        theme_frame.pack(fill="x", expand=True, padx=10, pady=10)
        ctk.CTkLabel(theme_frame, text="UI Theme:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        theme_var = ctk.StringVar(value=ctk.get_appearance_mode().lower())
        ctk.CTkOptionMenu(theme_frame, values=["dark", "light", "system"], variable=theme_var, command=self.change_theme).grid(row=0, column=1, padx=10, pady=10)

        exe_frame = ctk.CTkFrame(settings_frame)
        exe_frame.pack(fill="x", expand=True, padx=10, pady=10)
        exe_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(exe_frame, text="KoboldCPP Executable:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.exe_path_entry = ctk.CTkEntry(exe_frame, width=400)
        self.exe_path_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(exe_frame, text="Browse", command=self.browse_executable).grid(row=0, column=2, padx=10, pady=10)

        webui_frame = ctk.CTkFrame(settings_frame)
        webui_frame.pack(fill="x", expand=True, padx=10, pady=5)
        ctk.CTkLabel(webui_frame, text="Launcher Behavior:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.auto_open_webui_var = ctk.BooleanVar(value=self.config.get("auto_open_webui", True))
        self.auto_open_webui_checkbox = ctk.CTkCheckBox(webui_frame, text="Auto-Open Web UI After Launch", variable=self.auto_open_webui_var)
        self.auto_open_webui_checkbox.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(settings_frame, text="Global KoboldCpp Default Arguments", font=ctk.CTkFont(size=16, weight="bold")).pack(fill="x", expand=True, padx=10, pady=(20, 10))
        self.settings_widgets = {} # Stores {"--param_name": {"widget": widget_obj, "type": "bool/str_num"}}
        
        param_definitions = [
            {"name": "Threads", "param": "--threads", "help": "CPU threads ('auto' or number)."},
            {"name": "BLAS Threads (nblas)", "param": "--nblas", "help": "BLAS threads ('auto' or number)."},
            {"name": "Context Size", "param": "--contextsize", "help": "Max context tokens (e.g., 4096, 16384)."},
            {"name": "Prompt Limit", "param": "--promptlimit", "help": "Max prompt tokens (<= contextsize)."},
            {"name": "GPU Layers", "param": "--gpulayers", "help": "Layers on GPU. 'off', 0 for CPU. 999 for max."},
            {"name": "Use CUBLAS", "param": "--usecublas", "help": "Enable CUDA BLAS (NVIDIA).", "type": "bool"},
            {"name": "Flash Attention", "param": "--flashattention", "help": "Enable FlashAttention.", "type": "bool"},
            {"name": "No Memory Map", "param": "--nommap", "help": "Disable memory mapping.", "type": "bool"},
            {"name": "Low VRAM Mode", "param": "--lowvram", "help": "Enable low VRAM mode.", "type": "bool"},
            {"name": "QuantKV", "param": "--quantkv", "help": "K/V cache quant ('auto', 'off', or number)."},
            {"name": "BLAS Batch Size", "param": "--blasbatchsize", "help": "BLAS batch size ('auto', 'off', or number)."},
            {"name": "Port", "param": "--port", "help": "Web UI port (default 5000)."},
            {"name": "Default Gen Amount", "param": "--defaultgenamt", "help": "Default tokens to generate."}
        ]

        for setting_def in param_definitions:
            param_name = setting_def["param"]
            core_default_value = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].get(param_name)
            
            frame = ctk.CTkFrame(settings_frame); frame.pack(fill="x", expand=True, padx=10, pady=5)
            frame.grid_columnconfigure(1, weight=0); frame.grid_columnconfigure(2, weight=1) # Label, Widget, HelpText
            
            ctk.CTkLabel(frame, text=f"{setting_def['name']}:", width=160, anchor="w").grid(row=0, column=0, padx=10, pady=(5,0), sticky="w")
            
            widget_instance = None
            widget_type_str = "str_num" # Default for Entry

            is_boolean_param = setting_def.get("type") == "bool" or isinstance(core_default_value, bool)
            
            if is_boolean_param:
                widget_type_str = "bool"
                initial_bool_val = False # Determine initial value for checkbox
                if isinstance(core_default_value, bool): initial_bool_val = core_default_value
                elif isinstance(core_default_value, str): initial_bool_val = core_default_value.lower() == 'true'
                
                var = ctk.BooleanVar(value=initial_bool_val)
                widget_instance = ctk.CTkCheckBox(frame, text="", variable=var)
                widget_instance.grid(row=0, column=1, padx=10, pady=(5,0), sticky="w")
                widget_instance.variable = var # Store var for easy access
            else: # Entry for string/number/auto
                widget_instance = ctk.CTkEntry(frame, width=120) # Fixed width for entry
                if core_default_value is not None:
                    widget_instance.insert(0, str(core_default_value))
                widget_instance.grid(row=0, column=1, padx=10, pady=(5,0), sticky="w")
            
            ctk.CTkLabel(frame, text=setting_def['help'], font=ctk.CTkFont(size=11), text_color="gray", anchor="w", justify="left", wraplength=350).grid(row=0, column=2, padx=10, pady=(5,0), sticky="w")
            self.settings_widgets[param_name] = {"widget": widget_instance, "type": widget_type_str}


        ctk.CTkButton(settings_frame, text="Save Settings", command=self.save_settings_action).pack(pady=20)
        ctk.CTkButton(settings_frame, text="Reset All Settings to Defaults", command=self.reset_config_action, fg_color="#dc3545", hover_color="#c82333").pack(pady=10)

    def browse_executable(self):
        current_exe_dir = os.getcwd()
        if self.koboldcpp_executable and os.path.exists(os.path.dirname(self.koboldcpp_executable)):
            current_exe_dir = os.path.dirname(self.koboldcpp_executable)
        filetypes = [("Executables", "*.exe" if sys.platform == "win32" else "*"), ("Python Files", "*.py"), ("All Files", "*.*")]
        filepath = filedialog.askopenfilename(title="Select KoboldCPP Executable", filetypes=filetypes, initialdir=current_exe_dir)
        if filepath:
            if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
                self.exe_path_entry.delete(0, "end"); self.exe_path_entry.insert(0, filepath)
            self.log_to_console(f"Selected KoboldCPP executable (pending save): {filepath}")

    def setup_history_tab(self):
        history_frame = ctk.CTkFrame(self.tab_history); history_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        history_frame.grid_columnconfigure(0, weight=1); history_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(history_frame, text="Launch History", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.history_text = ctk.CTkTextbox(history_frame); self.history_text.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkButton(history_frame, text="Refresh History", command=self.load_history).grid(row=2, column=0, padx=10, pady=10)
        self.after(1000, self.load_history)

    def reset_config_action(self):
        if messagebox.askyesno("Reset Configuration", "Reset all settings to defaults? This will overwrite your config file (a backup will be attempted)."):
            try:
                config_path = koboldcpp_core.CONFIG_FILE
                if os.path.exists(config_path):
                    shutil.copy2(config_path, config_path + ".backup_" + time.strftime("%Y%m%d_%H%M%S"))
                    self.log_to_console(f"Backed up current config to {config_path}.backup_{time.strftime('%Y%m%d_%H%M%S')}")
                
                self.config = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE.copy()
                # Crucially, ensure default_args is a deep copy for reset
                self.config["default_args"] = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()

                if "model_specific_args" not in self.config: self.config["model_specific_args"] = {}
                if "auto_open_webui" not in self.config:
                    self.config["auto_open_webui"] = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE.get("auto_open_webui", True)

                success, message = koboldcpp_core.save_launcher_config(self.config)
                if success:
                    self.db_path = self.config.get("db_file", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["db_file"])
                    self.default_model_dir = self.config.get("default_gguf_dir", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE.get("default_gguf_dir",""))
                    if not self.default_model_dir or not os.path.isdir(self.default_model_dir): self.default_model_dir = os.getcwd()
                    self.koboldcpp_executable = self.config.get("koboldcpp_executable", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["koboldcpp_executable"])
                    self.load_settings_from_config()
                    self.log_to_console("Configuration reset to defaults and saved."); messagebox.showinfo("Configuration Reset", "Configuration has been reset to defaults.")
                else: self.log_to_console(f"Error saving reset config: {message}"); messagebox.showerror("Error", f"Error saving reset configuration: {message}")
            except Exception as e: self.log_to_console(f"Error during configuration reset: {e}"); messagebox.showerror("Error", f"An error occurred during configuration reset: {e}")

    def launch_direct_defaults(self):
        if not self.current_model_path: messagebox.showwarning("No Model Selected", "Please select a GGUF model first."); return
        if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
            current_exe_from_gui = self.exe_path_entry.get().strip()
            if current_exe_from_gui: self.koboldcpp_executable = current_exe_from_gui
        self.check_koboldcpp_executable()
        if not os.path.exists(self.koboldcpp_executable): messagebox.showerror("Executable Not Found", f"Could not find KoboldCPP: {self.koboldcpp_executable}"); return
        
        self.log_to_console(f"Direct launching KoboldCPP with effective defaults for: {os.path.basename(self.current_model_path)}")
        effective_base_args = self._get_merged_args_for_model(self.current_model_path)
        
        active_args_for_launch = {} # Filter for active args (True bools, non-empty/non-"auto" strings where appropriate)
        for k, v_raw in effective_base_args.items():
            v = v_raw
            if isinstance(v_raw, str): # Convert "true"/"false" strings to bools for consistent handling
                if v_raw.lower() == 'true': v = True
                elif v_raw.lower() == 'false': v = False
            
            if isinstance(v, bool):
                if v: active_args_for_launch[k] = True # Add if True
            elif isinstance(v, str) and v.strip(): # Non-empty strings
                # For specific args like 'nblas', 'auto' means omit, so don't add 'auto' here for build_command's logic
                if k == "--nblas" and v.lower() == "auto":
                    pass # build_command will omit if --nblas is not in the dict or is "auto"
                else:
                    active_args_for_launch[k] = v.strip()
            elif not isinstance(v, str) and v is not None: # Numbers etc.
                active_args_for_launch[k] = v

        args_list = koboldcpp_core.build_command(self.current_model_path, None, self.model_analysis_info or {}, active_args_for_launch)
        full_cmd_list = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, args_list)
        self.vram_at_decision_for_db = koboldcpp_core.get_available_vram_mb()[0]
        self.last_approx_vram_used_kcpp_mb = None
        self._launch_final_koboldcpp(full_cmd_list, "SUCCESS_USER_DIRECT_SETTINGS_GUI", 0)

    def launch_best_remembered(self):
        if not self.current_model_path: messagebox.showwarning("No Model Selected", "Please select a GGUF model first."); return
        if not self.model_analysis_info or not self.model_analysis_info.get('filepath'): messagebox.showwarning("Model Not Analyzed", "Please select and analyze a model first."); return
        if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
            current_exe_from_gui = self.exe_path_entry.get().strip()
            if current_exe_from_gui: self.koboldcpp_executable = current_exe_from_gui
        self.check_koboldcpp_executable()
        if not os.path.exists(self.koboldcpp_executable): messagebox.showerror("Executable Not Found", f"KoboldCPP not found: {self.koboldcpp_executable}"); return
        
        self.log_to_console("Attempting to launch with best remembered config...")
        current_vram, _, _, _ = koboldcpp_core.get_available_vram_mb()
        best_config_from_db = koboldcpp_core.find_best_historical_config(self.db_path, self.model_analysis_info, current_vram)
        
        if best_config_from_db and best_config_from_db.get("args_list"):
            self.log_to_console(f"Found best remembered config (Lvl: {best_config_from_db['attempt_level']}, Outcome: {best_config_from_db['outcome']})")
            historical_args_dict = koboldcpp_core.args_list_to_dict(best_config_from_db["args_list"])
            if "--model" in historical_args_dict: del historical_args_dict["--model"] # Remove model from historical
            historical_ot_string = historical_args_dict.pop("--overridetensors", None) # Extract OT string
            
            base_for_remembered_launch = self._get_merged_args_for_model(self.current_model_path) # Get current model/global defaults
            final_args_for_launch = base_for_remembered_launch.copy()
            final_args_for_launch.update(historical_args_dict) # Override with remembered non-OT args
            
            active_final_args = {} # Filter for active args again
            for k, v_raw in final_args_for_launch.items():
                v = v_raw
                if isinstance(v_raw, str):
                    if v_raw.lower() == 'true': v = True
                    elif v_raw.lower() == 'false': v = False
                if isinstance(v, bool):
                    if v: active_final_args[k] = True
                elif isinstance(v, str) and v.strip():
                    if k == "--nblas" and v.lower() == "auto": pass 
                    else: active_final_args[k] = v.strip()
                elif not isinstance(v, str) and v is not None: active_final_args[k] = v
                
            final_args_list = koboldcpp_core.build_command(self.current_model_path, historical_ot_string, self.model_analysis_info, active_final_args)
            full_cmd_list = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, final_args_list)
            
            self.vram_at_decision_for_db = current_vram
            self.last_approx_vram_used_kcpp_mb = best_config_from_db.get("approx_vram_used_kcpp_mb")
            attempt_level_for_this_launch = best_config_from_db.get("attempt_level",0)
            self._launch_final_koboldcpp(full_cmd_list, "SUCCESS_USER_LAUNCHED_BEST_REMEMBERED_GUI", attempt_level_for_this_launch)
        else:
            self.log_to_console("No suitable remembered config found. Falling back to Settings Defaults launch.")
            messagebox.showinfo("No Remembered Config", "No suitable remembered configuration found. Launching with current settings defaults instead.")
            self.launch_direct_defaults()

    def stop_all_kcpp_processes_forcefully(self):
        self.log_to_console("Attempting to forcefully stop ALL KoboldCpp-related processes...")
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
            self.log_to_console(f"Stopping monitored KCPP process (PID: {self.kcpp_process_obj.pid})..."); koboldcpp_core.kill_process(self.kcpp_process_obj.pid, force=True); self.kcpp_process_obj = None
        if self.last_process and self.last_process.poll() is None:
            self.log_to_console(f"Stopping last direct launch KCPP process (PID: {self.last_process.pid})..."); koboldcpp_core.kill_process(self.last_process.pid, force=True); self.last_process = None
        self.process_running = False
        if self.tuning_in_progress: self.end_tuning_session(switch_to_model_selection=False) # Just end, don't switch view
        
        kobold_exe_basename = os.path.basename(self.koboldcpp_executable if self.koboldcpp_executable else "koboldcpp.exe")
        killed_by_sweep, msg_sweep = koboldcpp_core.kill_processes_by_name(kobold_exe_basename)
        self.log_to_console(f"Sweep for '{kobold_exe_basename}': {msg_sweep}")
        if kobold_exe_basename.lower().endswith(".py"):
            killed_py_sweep, msg_py_sweep = koboldcpp_core.kill_processes_by_name("python", cmdline_substr_filter=kobold_exe_basename)
            self.log_to_console(f"Sweep for 'python' with cmdline '{kobold_exe_basename}': {msg_py_sweep}")
            killed_by_sweep = killed_by_sweep or killed_py_sweep

        if killed_by_sweep: self.log_to_console("Forceful sweep completed. Check system if issues persist.")
        else: self.log_to_console("Forceful sweep attempted. No processes explicitly reported killed by patterns related to current KCPP executable.")


    def change_theme(self, new_theme):
        ctk.set_appearance_mode(new_theme)
        self.config["color_mode"] = new_theme.lower()
        self.save_config()
        self.log_to_console(f"Theme changed to {new_theme}.")

    def log_to_console(self, text):
        def _log():
            if hasattr(self, 'console') and self.console.winfo_exists():
                self.console.configure(state="normal"); timestamp = time.strftime("%H:%M:%S", time.localtime())
                self.console.insert("end", f"[{timestamp}] {text}\n"); self.console.see("end"); self.console.configure(state="disabled")
        if hasattr(self, 'after'): self.after(0, _log)

    def monitor_vram(self):
        while True:
            try:
                free_mb, total_mb, message, gpu_info_dict = koboldcpp_core.get_available_vram_mb()
                used_mb = total_mb - free_mb if total_mb > 0 else 0
                self.after(0, lambda u=used_mb, t=total_mb, msg=message: self.update_vram_display(u, t, msg))
            except Exception: pass
            time.sleep(5)

    def update_vram_display(self, used_mb, total_mb, message_from_core=""):
        if hasattr(self, 'vram_progress') and self.vram_progress.winfo_exists():
            if total_mb > 0:
                percentage = used_mb / total_mb; self.vram_progress.set(percentage)
                self.vram_text.configure(text=f"{used_mb:.0f}MB / {total_mb:.0f}MB ({percentage*100:.1f}%)")
                progress_color = "#28a745" # Green
                if percentage > 0.9: progress_color = "#dc3545" # Red
                elif percentage > 0.7: progress_color = "#ffc107" # Yellow
                self.vram_progress.configure(progress_color=progress_color)
            else: self.vram_progress.set(0); self.vram_text.configure(text=f"VRAM: {message_from_core or 'Info unavailable'}")

    def refresh_vram(self):
        self.log_to_console("Refreshing VRAM info...")
        try:
            free_mb, total_mb, message, gpu_info_dict = koboldcpp_core.get_available_vram_mb()
            used_mb = total_mb - free_mb if total_mb > 0 else 0
            self.update_vram_display(used_mb, total_mb, message); self.log_to_console(f"VRAM Refreshed: {message}")
        except Exception as e: self.log_to_console(f"Error refreshing VRAM: {e}"); self.update_vram_display(0,0, "Error refreshing VRAM")

    def save_settings_action(self):
        if self.save_config(): messagebox.showinfo("Settings Saved", "Global settings were saved successfully!")

    def load_history(self):
        if not hasattr(self, 'history_text') or not self.history_text.winfo_exists(): return
        self.history_text.configure(state="normal"); self.history_text.delete("1.0", "end")
        if not os.path.exists(self.db_path): self.history_text.insert("end", f"No history database found at {self.db_path}.")
        else:
            try:
                rows = koboldcpp_core.get_history_entries(self.db_path, limit=50)
                if not rows: self.history_text.insert("end", "No launch history found.")
                else:
                    self.history_text.insert("end", "Recent Launches (newest first, max 50):\n\n")
                    for i, row_tuple in enumerate(rows):
                        model_name = os.path.basename(row_tuple[0]); size = row_tuple[1] or "N/A"
                        quant = row_tuple[2] or "N/A"; is_moe_val = "Yes" if row_tuple[3] else "No"
                        vram = row_tuple[4] or "N/A"; level = row_tuple[5]
                        outcome = row_tuple[6] or "N/A"
                        timestamp_val = row_tuple[8] if len(row_tuple) > 8 else "N/A"
                        entry = (f"{i+1}. {model_name}\n"
                                 f"   Size: {size}B, Quant: {quant}, MoE: {is_moe_val}\n"
                                 f"   VRAM@L: {vram}MB, OT_Lvl: {level}, Outcome: {outcome}\n"
                                 f"   Time: {timestamp_val}\n\n")
                        self.history_text.insert("end", entry)
            except Exception as e: self.history_text.insert("end", f"Error loading history: {e}")
        self.history_text.configure(state="disabled")

    def browse_model(self):
        initial_dir_to_use = self.config.get("last_used_gguf_dir", self.default_model_dir)
        if not initial_dir_to_use or not os.path.isdir(initial_dir_to_use): initial_dir_to_use = os.getcwd()
        filepath = filedialog.askopenfilename(title="Select GGUF Model File", filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")], initialdir=initial_dir_to_use)
        if filepath:
            self.model_path_entry.delete(0, "end"); self.model_path_entry.insert(0, filepath)
            self.current_model_path = filepath; self.default_model_dir = os.path.dirname(filepath)
            self.config["last_used_gguf_dir"] = self.default_model_dir; self.save_config()
            self.analyze_model_action(filepath)

    def analyze_model_action(self, filepath):
        self.log_to_console(f"Analyzing model: {os.path.basename(filepath)}")
        self.model_analysis_info = koboldcpp_core.analyze_filename(filepath)
        is_moe_display = 'MoE' if self.model_analysis_info.get('is_moe') else 'Dense'
        size_b_display = self.model_analysis_info.get('size_b', "N/A")
        quant_display = self.model_analysis_info.get('quant', "N/A")
        if hasattr(self, 'model_info_label') and self.model_info_label.winfo_exists():
            self.model_info_label.configure(text=f"Model: {is_moe_display}, Size: ~{size_b_display}B, Quant: {quant_display}")
        self.log_to_console(f"Analysis - MoE: {self.model_analysis_info.get('is_moe')}, Size: {size_b_display}B, Quant: {quant_display}")

if __name__ == "__main__":
    app = KoboldLauncherGUI()
    def on_closing():
        kcpp_monitored_running = hasattr(app, 'kcpp_process_obj') and app.kcpp_process_obj and app.kcpp_process_obj.poll() is None
        last_direct_launch_running = app.last_process and app.last_process.poll() is None
        if app.tuning_in_progress or app.process_running or kcpp_monitored_running or last_direct_launch_running:
            if messagebox.askyesno("Exit", "A KoboldCpp process might be running or tuning is in progress. Stop all related processes and exit?"):
                app.stop_all_kcpp_processes_forcefully(); time.sleep(0.5); app.destroy()
        else: app.destroy()
    app.protocol("WM_DELETE_WINDOW", on_closing); app.mainloop()
