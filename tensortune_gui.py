

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

class KoboldLauncherGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("KoboldCpp Smart Launcher (GUI Edition v1.2)")
        self.geometry("950x820") # Increased height slightly for new elements
        self.minsize(850, 720)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        core_init_results = koboldcpp_core.initialize_launcher()
        self.config = core_init_results["config"]
        self.system_info = core_init_results["system_info"]
        self.gpu_info = core_init_results["gpu_info"]
        self.koboldcpp_capabilities = core_init_results.get("koboldcpp_capabilities", {})

        if "model_specific_args" not in self.config:
            self.config["model_specific_args"] = {}
        self.current_model_path = None
        self.process_running = False
        self.model_analysis_info = {}
        self.last_process = None
        self.db_path = self.config["db_file"]
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
        self.last_successful_monitored_run_details_gui = None


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
        self.load_settings_from_config()
        self.setup_main_tab()
        self.setup_history_tab()

        threading.Thread(target=self.monitor_vram, daemon=True).start()
        self.log_to_console(f"KoboldCpp Smart Launcher GUI Initialized. Core config loaded: {core_init_results['config_message']}")
        self.log_to_console(f"Using DB at: {self.db_path}")
        self.log_to_console(f"Initial GPU Info: {self.gpu_info.get('message', 'N/A')}")
        self.log_to_console(f"KCPP Caps: CUDA:{self.koboldcpp_capabilities.get('cuda', False)}, ROCm:{self.koboldcpp_capabilities.get('rocm',False)}")
        self.check_koboldcpp_executable()
        self._show_model_selection_view()
        self.after(100, self._run_first_time_setup_if_needed)


    def setup_main_tab(self):
        self.tab_main.grid_rowconfigure(0, weight=1)
        self.model_selection_frame = ctk.CTkFrame(self.tab_main)
        self.model_selection_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.model_selection_frame.grid_columnconfigure(0, weight=1)
        self.model_selection_frame.grid_rowconfigure(5, weight=1)

        title_label = ctk.CTkLabel(self.model_selection_frame, text="KoboldCpp Model Launcher", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 15), sticky="n")

        model_controls_frame = ctk.CTkFrame(self.model_selection_frame)
        model_controls_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        model_controls_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(model_controls_frame, text="GGUF Model:").grid(row=0, column=0, padx=(10,5), pady=10, sticky="w")
        self.model_path_entry = ctk.CTkEntry(model_controls_frame, width=400)
        self.model_path_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        ctk.CTkButton(model_controls_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=(5,10), pady=10, sticky="e")
        self.model_info_label = ctk.CTkLabel(model_controls_frame, text="No model selected. Analysis includes: Type, Size, Quant, Layers, Est. VRAM.", justify="left", wraplength=600)
        self.model_info_label.grid(row=1, column=0, columnspan=3, padx=10, pady=(0,10), sticky="w")

        vram_frame = ctk.CTkFrame(self.model_selection_frame)
        vram_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        vram_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(vram_frame, text="GPU Status:").grid(row=0, column=0, padx=(10,5), pady=5, sticky="w")
        self.vram_progress = ctk.CTkProgressBar(vram_frame, height=18)
        self.vram_progress.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.vram_progress.set(0)
        self.vram_text = ctk.CTkLabel(vram_frame, text="Scanning...")
        self.vram_text.grid(row=0, column=2, padx=(5,10), pady=5, sticky="e")
        ctk.CTkButton(vram_frame, text="Refresh", width=60, command=self.refresh_vram).grid(row=0, column=3, padx=(5,10), pady=5, sticky="e")

        launch_buttons_frame = ctk.CTkFrame(self.model_selection_frame)
        launch_buttons_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        launch_buttons_frame.grid_columnconfigure((0,1,2), weight=1)
        ctk.CTkButton(launch_buttons_frame, text="Start Auto-Tune / Use OT Strategy", command=self.start_tuning_session, height=35, fg_color="seagreen", hover_color="darkgreen").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(launch_buttons_frame, text="Launch Best Remembered Config", command=self.launch_best_remembered, height=35, fg_color="cornflowerblue", hover_color="royalblue").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(launch_buttons_frame, text="Direct Launch (Settings Defaults)", command=self.launch_direct_defaults, height=35).grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        stop_button_frame = ctk.CTkFrame(self.model_selection_frame)
        stop_button_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=(0,5), sticky="ew")
        stop_button_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(stop_button_frame, text="Stop Any KCPP Processes", command=self.stop_all_kcpp_processes_forcefully, height=35, fg_color="firebrick", hover_color="darkred").grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        console_frame_ms = ctk.CTkFrame(self.model_selection_frame)
        console_frame_ms.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        console_frame_ms.grid_columnconfigure(0, weight=1)
        console_frame_ms.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(console_frame_ms, text="Launcher Log:").grid(row=0, column=0, padx=10, pady=(5,0), sticky="w")
        self.console = ctk.CTkTextbox(console_frame_ms, height=100)
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
        self.tuning_view_vram_text = ctk.CTkLabel(self.tuning_view_vram_frame, text="Scanning...")
        self.tuning_view_vram_text.grid(row=0, column=2, padx=(5,10), pady=5, sticky="e")
        ctk.CTkButton(self.tuning_view_vram_frame, text="Refresh", width=60, command=self.refresh_vram).grid(row=0, column=3, padx=(5,10), pady=5, sticky="e")

        self.last_run_info_frame_tuning = ctk.CTkFrame(self.tuning_mode_frame)
        self.last_run_info_frame_tuning.grid(row=3, column=0, padx=10, pady=2, sticky="ew")
        self.last_run_info_frame_tuning.grid_columnconfigure(0, weight=1)
        self.tuning_last_run_info_label = ctk.CTkLabel(self.last_run_info_frame_tuning, text="Last Monitored Success: None yet in this session.", justify="left", font=ctk.CTkFont(size=11), text_color="gray")
        self.tuning_last_run_info_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")

        ot_strategy_display_frame = ctk.CTkFrame(self.tuning_mode_frame)
        ot_strategy_display_frame.grid(row=4, column=0, padx=10, pady=2, sticky="ew")
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
        proposed_command_frame.grid(row=5, column=0, padx=10, pady=2, sticky="ew")
        proposed_command_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(proposed_command_frame, text="Proposed Command:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.tuning_proposed_command_text = ctk.CTkTextbox(proposed_command_frame, height=100, wrap="word")
        self.tuning_proposed_command_text.grid(row=1, column=0, padx=5, pady=2, sticky="nsew")
        self.tuning_proposed_command_text.configure(state="disabled")

        self.tuning_actions_primary_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.tuning_actions_primary_frame.grid(row=6, column=0, padx=10, pady=(5,2), sticky="ew")
        self.tuning_actions_primary_frame.grid_columnconfigure((0,1), weight=1)
        self.btn_tune_launch_monitor = ctk.CTkButton(self.tuning_actions_primary_frame, text="Launch & Monitor Output", command=self.launch_and_monitor_for_tuning, height=35, fg_color="seagreen", hover_color="darkgreen")
        self.btn_tune_launch_monitor.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.btn_tune_skip_launch_direct = ctk.CTkButton(self.tuning_actions_primary_frame, text="Skip Tune & Launch This Config", command=self.skip_tune_and_launch_direct, height=35)
        self.btn_tune_skip_launch_direct.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.tuning_actions_secondary_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.tuning_actions_secondary_frame.grid(row=7, column=0, padx=10, pady=0, sticky="ew")
        self.tuning_actions_secondary_frame.grid_columnconfigure((0,1), weight=1)
        self.btn_tune_more_gpu = ctk.CTkButton(self.tuning_actions_secondary_frame, text="More GPU (↓ Level)", command=lambda: self.adjust_ot_level(-1))
        self.btn_tune_more_gpu.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.btn_tune_more_cpu = ctk.CTkButton(self.tuning_actions_secondary_frame, text="More CPU (↑ Level)", command=lambda: self.adjust_ot_level(1))
        self.btn_tune_more_cpu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.tuning_edit_args_buttons_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.tuning_edit_args_buttons_frame.grid(row=8, column=0, padx=10, pady=2, sticky="ew")
        self.tuning_edit_args_buttons_frame.grid_columnconfigure((0,1), weight=1)
        self.btn_tune_edit_args = ctk.CTkButton(self.tuning_edit_args_buttons_frame, text="Edit Base Args (This Session)", command=self.edit_base_args_for_tuning_session)
        self.btn_tune_edit_args.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.btn_tune_edit_model_perm_args = ctk.CTkButton(self.tuning_edit_args_buttons_frame, text="Edit Base Args (Permanent for This Model)", command=self.edit_permanent_model_args)
        self.btn_tune_edit_model_perm_args.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.tuning_actions_navigation_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.tuning_actions_navigation_frame.grid(row=9, column=0, padx=10, pady=2, sticky="ew")
        self.tuning_actions_navigation_frame.grid_columnconfigure((0,1,2), weight=1)
        self.btn_tune_new_gguf = ctk.CTkButton(self.tuning_actions_navigation_frame, text="New GGUF Model", command=self.select_new_gguf_during_tuning)
        self.btn_tune_new_gguf.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.btn_tune_history = ctk.CTkButton(self.tuning_actions_navigation_frame, text="View History (This Model)", command=lambda: self.view_history_for_current_model())
        self.btn_tune_history.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.btn_tune_quit_tuning = ctk.CTkButton(self.tuning_actions_navigation_frame, text="End Tuning Session", command=self.end_tuning_session, fg_color="firebrick", hover_color="darkred")
        self.btn_tune_quit_tuning.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.kcpp_output_console_frame = ctk.CTkFrame(self.tuning_mode_frame)
        self.kcpp_output_console_frame.grid(row=10, column=0, padx=10, pady=(5,10), sticky="nsew")
        self.kcpp_output_console_frame.grid_columnconfigure(0, weight=1)
        self.kcpp_output_console_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(self.kcpp_output_console_frame, text="KoboldCpp Output (during monitoring):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.kcpp_live_output_text = ctk.CTkTextbox(self.kcpp_output_console_frame, wrap="char")
        self.kcpp_live_output_text.grid(row=1, column=0, padx=5, pady=2, sticky="nsew")
        self.kcpp_live_output_text.configure(state="disabled")

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
        self.last_successful_monitored_run_details_gui = None

        if hasattr(self, 'tuning_model_name_label') and self.tuning_model_name_label.winfo_exists():
            self.tuning_model_name_label.configure(text=f"Model: {os.path.basename(self.current_tuning_model_path)}")

        current_vram_mb, _, _, _ = koboldcpp_core.get_available_vram_mb()
        is_moe = self.current_tuning_model_analysis.get('is_moe', False)
        estimated_vram_needed_gb = self.current_tuning_model_analysis.get('estimated_vram_gb_full_gpu', 0)
        estimated_vram_needed_mb = estimated_vram_needed_gb * 1024

        if is_moe:
            self.current_tuning_min_level, self.current_tuning_max_level, initial_heuristic_level = -25, 10, -10
        else:
            self.current_tuning_min_level, self.current_tuning_max_level = -17, 9
            size_b = self.current_tuning_model_analysis.get('size_b', 0)
            if isinstance(size_b, (int, float)):
                if size_b >= 30:
                    initial_heuristic_level = -3
                elif size_b >= 20:
                    initial_heuristic_level = -5
                else:
                    initial_heuristic_level = -7
            else:
                initial_heuristic_level = -5

        safety_buffer_mb = self.config.get("vram_safety_buffer_mb", 768)
        min_free_after_load_mb = self.config.get("min_vram_free_after_load_success_mb", 512)
        effective_vram_budget_mb = current_vram_mb - safety_buffer_mb - min_free_after_load_mb

        if estimated_vram_needed_mb > 0 and current_vram_mb > 0 :
            if estimated_vram_needed_mb > effective_vram_budget_mb * 1.1:
                initial_heuristic_level = max(initial_heuristic_level, -3 if not is_moe else -6)
                self.log_to_console(f"Heuristic: Est. VRAM ({estimated_vram_needed_mb:.0f}MB) > budget ({effective_vram_budget_mb:.0f}MB). Adjusting OT towards CPU.")
            elif estimated_vram_needed_mb < effective_vram_budget_mb * 0.7:
                initial_heuristic_level = min(initial_heuristic_level, -12 if not is_moe else -18)
                self.log_to_console(f"Heuristic: Ample VRAM. Adjusting OT towards GPU.")

        best_hist_config = koboldcpp_core.find_best_historical_config(self.db_path, self.current_tuning_model_analysis, current_vram_mb, self.config)

        if best_hist_config and "attempt_level" in best_hist_config:
            self.log_to_console(f"Found historical config. Level: {best_hist_config['attempt_level']}, Outcome: {best_hist_config['outcome']}")
            hist_level, hist_outcome = best_hist_config['attempt_level'], best_hist_config.get('outcome', "")
            approx_hist_vram_used = best_hist_config.get('approx_vram_used_kcpp_mb')

            if approx_hist_vram_used and (approx_hist_vram_used + safety_buffer_mb < current_vram_mb):
                initial_heuristic_level = max(self.current_tuning_min_level, hist_level -1 if hist_level > self.current_tuning_min_level else hist_level)
                self.log_to_console(f"Historical success fits VRAM. Starting near: {initial_heuristic_level}")
            elif hist_outcome.startswith("SUCCESS_LOAD_VRAM_OK") or hist_outcome.startswith("SUCCESS_USER_CONFIRMED") or hist_outcome.endswith("_USER_SAVED_GOOD_GPU_GUI"):
                initial_heuristic_level = max(self.current_tuning_min_level, hist_level -1 if hist_level > self.current_tuning_min_level else hist_level)
            elif hist_outcome.endswith("_USER_AUTO_ADJUST_CPU_GUI") or hist_outcome.endswith("_USER_TRIED_CPU_AFTER_FAIL_GUI") or "OOM" in hist_outcome.upper() or "TIGHT" in hist_outcome.upper():
                 initial_heuristic_level = min(self.current_tuning_max_level, hist_level + 1 if hist_level < self.current_tuning_max_level else hist_level)
            else:
                initial_heuristic_level = hist_level

            remembered_args_list = best_hist_config.get("args_list", [])
            if remembered_args_list:
                remembered_args_dict = koboldcpp_core.args_list_to_dict(remembered_args_list)
                remembered_args_dict.pop("--model", None)
                remembered_args_dict.pop("--overridetensors", None)
                self.current_tuning_session_base_args.update(remembered_args_dict)
                self.log_to_console(f"Applied remembered arguments. OT Level target adjusted to: {initial_heuristic_level}")
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
        if not self.tuning_in_progress:
            return
        self.current_tuning_attempt_level = max(self.current_tuning_min_level, min(self.current_tuning_attempt_level, self.current_tuning_max_level))

        if hasattr(self, 'tuning_model_name_label') and self.tuning_model_name_label.winfo_exists() and self.current_tuning_model_path:
            self.tuning_model_name_label.configure(text=f"Model: {os.path.basename(self.current_tuning_model_path)}")

        if hasattr(self, 'tuning_last_run_info_label') and self.tuning_last_run_info_label.winfo_exists():
            if self.last_successful_monitored_run_details_gui:
                level = self.last_successful_monitored_run_details_gui.get("level", "N/A")
                outcome = self.last_successful_monitored_run_details_gui.get("outcome", "N/A")
                vram_used_last = self.last_successful_monitored_run_details_gui.get("vram_used_mb", "N/A")
                self.tuning_last_run_info_label.configure(text=f"Last Monitored Success: Level {level}, Outcome: {outcome}, Approx VRAM Used: {vram_used_last}MB")
            else:
                self.tuning_last_run_info_label.configure(text="Last Monitored Success: None yet in this session.")

        ot_string = koboldcpp_core.generate_overridetensors(self.current_tuning_model_analysis, self.current_tuning_attempt_level)
        description = koboldcpp_core.get_offload_description(self.current_tuning_model_analysis, self.current_tuning_attempt_level, ot_string)
        gpu_layers = koboldcpp_core.get_gpu_layers_for_level(self.current_tuning_model_analysis, self.current_tuning_attempt_level)
        total_layers = self.current_tuning_model_analysis.get('num_layers', 32)

        if hasattr(self, 'tuning_ot_level_label'):
            self.tuning_ot_level_label.configure(text=f"Level: {self.current_tuning_attempt_level}")
        range_text_suffix = 'SuperMaxCPU' if self.current_tuning_model_analysis.get('is_moe') else 'MaxCPU'
        range_text = f"{self.current_tuning_min_level}=MaxGPU ... {self.current_tuning_max_level}={range_text_suffix}"
        if hasattr(self, 'tuning_ot_range_label'):
            self.tuning_ot_range_label.configure(text=f"Range: {range_text}")
        if hasattr(self, 'tuning_ot_strategy_desc_label'):
            self.tuning_ot_strategy_desc_label.configure(text=f"Strategy: {description}")
        regex_display = ot_string if ot_string else "None (Max GPU layers if --gpulayers is high)"
        if hasattr(self, 'tuning_ot_regex_label'):
            self.tuning_ot_regex_label.configure(text=f"Regex: {regex_display}")
        if hasattr(self, 'tuning_gpu_layers_label'):
            self.tuning_gpu_layers_label.configure(text=f"GPU Layers: {gpu_layers}/{total_layers}")

        args_for_kcpp_list = koboldcpp_core.build_command(self.current_tuning_model_path, ot_string, self.current_tuning_model_analysis, self.current_tuning_session_base_args)
        full_command_list = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, args_for_kcpp_list)
        display_command_str = koboldcpp_core.format_command_for_display(full_command_list)

        if hasattr(self, 'tuning_proposed_command_text'):
            self.tuning_proposed_command_text.configure(state="normal")
            self.tuning_proposed_command_text.delete("1.0", "end")
            self.tuning_proposed_command_text.insert("1.0", display_command_str)
            self.tuning_proposed_command_text.configure(state="disabled")
        if hasattr(self, 'btn_tune_more_gpu'):
            self.btn_tune_more_gpu.configure(state="normal" if self.current_tuning_attempt_level > self.current_tuning_min_level else "disabled")
        if hasattr(self, 'btn_tune_more_cpu'):
            self.btn_tune_more_cpu.configure(state="normal" if self.current_tuning_attempt_level < self.current_tuning_max_level else "disabled")
        self.log_to_console(f"Tuning display updated for OT Level: {self.current_tuning_attempt_level}")


    def _handle_monitoring_completion(self, initial_outcome_key: str):
        self.log_to_console(f"Monitoring completed. Initial Outcome: {initial_outcome_key}")
        self._log_to_kcpp_live_output(f"\n--- Monitoring Finished: {initial_outcome_key} ---\n")

        if initial_outcome_key in ["TIMEOUT_NO_SIGNAL_GUI", "OOM_CRASH_DETECTED_GUI", "PREMATURE_EXIT_GUI"] or \
           "OOM" in initial_outcome_key.upper() or "CRASH" in initial_outcome_key.upper():
            if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
                self.log_to_console("Terminating KCPP process due to unfavorable outcome...")
                koboldcpp_core.kill_process(self.kcpp_process_obj.pid, force=True)
            self.kcpp_process_obj = None
        self.last_approx_vram_used_kcpp_mb = None
        self.last_free_vram_after_load_mb = None
        final_db_outcome = initial_outcome_key

        if initial_outcome_key == "SUCCESS_LOAD_DETECTED_GUI":
            self._log_to_kcpp_live_output("API detected. Waiting for VRAM to stabilize...\n")
            time.sleep(self.config.get("vram_stabilization_wait_s", 3.0))
            current_free_vram, total_vram, _, _ = koboldcpp_core.get_available_vram_mb()
            self.last_free_vram_after_load_mb = current_free_vram
            if total_vram > 0 and self.vram_at_decision_for_db is not None:
                vram_used_by_kcpp = self.vram_at_decision_for_db - current_free_vram
                self.last_approx_vram_used_kcpp_mb = max(0, min(vram_used_by_kcpp, total_vram))
                self._log_to_kcpp_live_output(f"VRAM after load: {current_free_vram:.0f}MB free. Approx KCPP usage: {self.last_approx_vram_used_kcpp_mb:.0f}MB\n")
                min_vram_free_target = self.config.get("min_vram_free_after_load_success_mb", 512)
                if current_free_vram < min_vram_free_target:
                    self._log_to_kcpp_live_output(f"WARNING: VRAM tight! {current_free_vram:.0f}MB < {min_vram_free_target}MB target.\n")
                    final_db_outcome = "SUCCESS_LOAD_VRAM_TIGHT_GUI"
                else:
                    self._log_to_kcpp_live_output("VRAM usage OK.\n")
                    final_db_outcome = "SUCCESS_LOAD_VRAM_OK_GUI"

                self.last_successful_monitored_run_details_gui = {
                    "level": self.level_of_last_monitored_run,
                    "outcome": final_db_outcome,
                    "vram_used_mb": f"{self.last_approx_vram_used_kcpp_mb:.0f}" if self.last_approx_vram_used_kcpp_mb is not None else "N/A"
                }
                self.update_tuning_display()
            else:
                final_db_outcome = "SUCCESS_LOAD_NO_VRAM_CHECK_GUI"

        koboldcpp_core.save_config_to_db(self.db_path, self.current_tuning_model_path, self.current_tuning_model_analysis, self.vram_at_decision_for_db, self.current_command_list_for_db, self.level_of_last_monitored_run, final_db_outcome, self.last_approx_vram_used_kcpp_mb)
        self.load_history()
        self._present_post_monitoring_choices(final_db_outcome)

    def _present_post_monitoring_choices(self, outcome: str):
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
            self.log_to_console("Stopping monitored KCPP instance before showing post-monitoring choices.")
            koboldcpp_core.kill_process(self.kcpp_process_obj.pid, force=True)
        self.kcpp_process_obj = None
        frames_to_hide_names = ['tuning_actions_primary_frame', 'tuning_actions_secondary_frame',
                                'tuning_edit_args_buttons_frame',
                                'tuning_actions_navigation_frame']
        for frame_name in frames_to_hide_names:
            frame_widget = getattr(self, frame_name, None)
            if frame_widget and hasattr(frame_widget, 'winfo_exists') and frame_widget.winfo_exists():
                frame_widget.grid_remove()

        if not hasattr(self, 'post_monitor_choices_frame') or not self.post_monitor_choices_frame.winfo_exists():
            self.post_monitor_choices_frame = ctk.CTkFrame(self.tuning_mode_frame)
            self.post_monitor_choices_frame.grid(row=6, column=0, rowspan=3, padx=10, pady=5, sticky="news")
            self.post_monitor_choices_frame.grid_columnconfigure(0, weight=1)
        else:
            self.post_monitor_choices_frame.grid()

        for widget in self.post_monitor_choices_frame.winfo_children():
            widget.destroy()
        ctk.CTkLabel(self.post_monitor_choices_frame, text=f"Outcome: {outcome}", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 2), anchor="w", padx=5)
        if self.last_free_vram_after_load_mb is not None:
            ctk.CTkLabel(self.post_monitor_choices_frame, text=f"VRAM After Load: {self.last_free_vram_after_load_mb:.0f} MB free").pack(pady=1, anchor="w", padx=5)
        if self.last_approx_vram_used_kcpp_mb is not None:
            ctk.CTkLabel(self.post_monitor_choices_frame, text=f"Approx. KCPP VRAM Used: {self.last_approx_vram_used_kcpp_mb:.0f} MB").pack(pady=1, anchor="w", padx=5)
        vram_status_message = "VRAM Status: Check KCPP Log."
        if "SUCCESS_LOAD_VRAM_OK" in outcome:
            vram_status_message = "VRAM Usage: OK"
        elif "SUCCESS_LOAD_VRAM_TIGHT" in outcome:
            vram_status_message = "VRAM Usage: TIGHT"
        elif "SUCCESS_LOAD_NO_VRAM_CHECK" in outcome:
            vram_status_message = "VRAM Usage: Not Checked"
        ctk.CTkLabel(self.post_monitor_choices_frame, text=vram_status_message).pack(pady=(1, 10), anchor="w", padx=5)

        if "SUCCESS_LOAD_VRAM_OK" in outcome:
            ctk.CTkButton(self.post_monitor_choices_frame, text="✅ Accept & Launch This Config for Use", command=lambda: self._handle_post_monitor_action("launch_for_use", outcome)).pack(fill="x", pady=3, padx=5)
            ctk.CTkButton(self.post_monitor_choices_frame, text="💾 Save as Good, Auto-Adjust for More GPU & Continue Tuning", command=lambda: self._handle_post_monitor_action("save_good_more_gpu", outcome)).pack(fill="x", pady=3, padx=5)
            ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Manually Try More GPU (This Session) & Continue Tuning", command=lambda: self._handle_post_monitor_action("more_gpu_now", outcome)).pack(fill="x", pady=3, padx=5)
            ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Manually Try More CPU (This Session) & Continue Tuning", command=lambda: self._handle_post_monitor_action("more_cpu_now", outcome)).pack(fill="x", pady=3, padx=5)
        elif "SUCCESS_LOAD_VRAM_TIGHT" in outcome:
            ctk.CTkButton(self.post_monitor_choices_frame, text="⚠️ Auto-Adjust for More CPU & Continue Tuning", command=lambda: self._handle_post_monitor_action("auto_adjust_cpu", outcome)).pack(fill="x", pady=3, padx=5)
            ctk.CTkButton(self.post_monitor_choices_frame, text="🚀 Launch This Config Anyway (Risky)", command=lambda: self._handle_post_monitor_action("launch_for_use_risky", outcome)).pack(fill="x", pady=3, padx=5)
        elif "OOM" in outcome or "CRASH" in outcome or "PREMATURE_EXIT" in outcome :
            ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Auto-Adjust for More CPU & Continue Tuning", command=lambda: self._handle_post_monitor_action("more_cpu_after_fail", outcome)).pack(fill="x", pady=3, padx=5)
        elif "TIMEOUT" in outcome:
            ctk.CTkLabel(self.post_monitor_choices_frame, text="Launch timed out.").pack(pady=3, anchor="w", padx=5)
            ctk.CTkButton(self.post_monitor_choices_frame, text="⚙️ Auto-Adjust for More CPU (Assume OOM) & Continue Tuning", command=lambda: self._handle_post_monitor_action("more_cpu_after_fail", outcome)).pack(fill="x", pady=3, padx=5)
        ctk.CTkButton(self.post_monitor_choices_frame, text="↩️ Save Outcome & Return to Tuning Menu (Manual Adjust)", command=lambda: self._handle_post_monitor_action("return_to_tuning_menu", outcome)).pack(fill="x", pady=3, padx=5)
        self.log_to_console("Presented post-monitoring choices to user.")

    def _return_to_full_tuning_menu(self):
        if hasattr(self, 'post_monitor_choices_frame') and self.post_monitor_choices_frame.winfo_exists():
            self.post_monitor_choices_frame.grid_remove()
        frames_to_show_names = ['tuning_actions_primary_frame', 'tuning_actions_secondary_frame',
                                'tuning_edit_args_buttons_frame',
                                'tuning_actions_navigation_frame']
        for frame_name in frames_to_show_names:
            frame_widget = getattr(self, frame_name, None)
            if frame_widget and hasattr(frame_widget, 'winfo_exists') and frame_widget.winfo_exists():
                frame_widget.grid()
        self._set_tuning_buttons_state("normal")
        self.update_tuning_display()

    def _run_first_time_setup_if_needed(self):
        if not self.config.get("first_run_completed", False):
            self.log_to_console("Performing first-time setup...")
            dialog = FirstTimeSetupDialog(self, self.config, self.koboldcpp_executable)
            self.wait_window(dialog)
            if dialog.saved_config:
                self.config = dialog.saved_config
                self.koboldcpp_executable = self.config["koboldcpp_executable"]
                self.default_model_dir = self.config.get("default_gguf_dir", "")
                if hasattr(self, 'exe_path_entry'):
                    self.exe_path_entry.insert(0, self.koboldcpp_executable)
                if hasattr(self, 'auto_open_webui_var'):
                    self.auto_open_webui_var.set(self.config.get("auto_open_webui", True))
                self.load_settings_from_config()
                self.save_config()
                self.log_to_console("First-time setup complete. Configuration updated.")
            else:
                self.log_to_console("First-time setup cancelled or not completed.")
                messagebox.showwarning("Setup Incomplete", "First-time setup was not completed. Please configure the KoboldCpp executable in Settings.")

    def _get_merged_args_for_model(self, model_path):
        merged_args = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
        merged_args.update(self.config.get("default_args", {}))
        if model_path:
            model_specifics = self.config.get("model_specific_args", {}).get(model_path, {})
            merged_args.update(model_specifics)
        return merged_args

    def _reinitialize_session_base_args(self):
        if not self.current_tuning_model_path:
            effective_args_base = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
            effective_args_base.update(self.config.get("default_args", {}))
        else:
            effective_args_base = self._get_merged_args_for_model(self.current_tuning_model_path)
        self.current_tuning_session_base_args = {}
        for k, v_raw in effective_args_base.items():
            v = v_raw
            arg_def = next((d for d in koboldcpp_core.KOBOLDCPP_ARG_DEFINITIONS if d["key"] == k), None)
            is_bool_type = arg_def and arg_def.get("type_hint") in ["bool", "bool_flag"]
            if is_bool_type:
                if isinstance(v_raw, bool):
                    v = v_raw
                elif isinstance(v_raw, str):
                    v = v_raw.lower() == 'true'
                else:
                    v = False
                self.current_tuning_session_base_args[k] = v
            elif v is not None :
                if isinstance(v, str):
                    self.current_tuning_session_base_args[k] = v.strip()
                else:
                    self.current_tuning_session_base_args[k] = v
        model_name_display = os.path.basename(self.current_tuning_model_path) if self.current_tuning_model_path else "No Model"
        self.log_to_console(f"Session base arguments reinitialized for '{model_name_display}'.")

    def check_koboldcpp_executable(self):
        if hasattr(self, 'exe_path_entry') and self.exe_path_entry.winfo_exists():
            current_exe_path_from_gui = self.exe_path_entry.get().strip()
            if current_exe_path_from_gui:
                self.koboldcpp_executable = current_exe_path_from_gui
        current_exe_path_to_check = self.koboldcpp_executable
        config_needs_save = False
        if not current_exe_path_to_check or not os.path.exists(current_exe_path_to_check):
            self.log_to_console(f"Warning: KoboldCPP executable '{current_exe_path_to_check}' not found.")
            executable_basename = os.path.basename(current_exe_path_to_check) if current_exe_path_to_check else ("koboldcpp.exe" if platform.system() == "Windows" else "koboldcpp")
            found_exe_in_path = shutil.which(executable_basename)
            if found_exe_in_path:
                self.log_to_console(f"Found '{executable_basename}' in PATH: {found_exe_in_path}")
                self.koboldcpp_executable = os.path.abspath(found_exe_in_path)
                config_needs_save = True
            elif executable_basename.lower().endswith(".py"):
                try:
                    launcher_dir = os.path.dirname(os.path.abspath(__file__)) if not getattr(sys, 'frozen', False) else os.path.dirname(sys.executable)
                    potential_relative_path = os.path.join(launcher_dir, executable_basename)
                    if os.path.exists(potential_relative_path):
                        self.log_to_console(f"Found '{executable_basename}' relative: {potential_relative_path}")
                        self.koboldcpp_executable = os.path.abspath(potential_relative_path)
                        config_needs_save = True
                    else:
                        self.log_to_console(f"Could not find '{executable_basename}' in PATH or relative.")
                except NameError:
                    self.log_to_console(f"Could not determine launcher script directory for '{executable_basename}'.")
            else:
                self.log_to_console(f"Could not find '{executable_basename}' in PATH.")
        else:
            self.log_to_console(f"KoboldCPP executable '{current_exe_path_to_check}' verified.")
        if config_needs_save:
            if hasattr(self, 'exe_path_entry'):
                self.exe_path_entry.delete(0, "end")
                self.exe_path_entry.insert(0, self.koboldcpp_executable)
            self.config["koboldcpp_executable"] = self.koboldcpp_executable
            self.log_to_console("KCPP exe path updated. Saving config...")
            self.save_config()
        elif current_exe_path_to_check != self.koboldcpp_executable :
            self.config["koboldcpp_executable"] = self.koboldcpp_executable

    def load_settings_from_config(self):
        global_default_args = self.config.get("default_args", {})
        for arg_def in koboldcpp_core.KOBOLDCPP_ARG_DEFINITIONS:
            param_key = arg_def["key"]
            if param_key not in self.settings_widgets:
                continue
            widget_info = self.settings_widgets[param_key]
            widget = widget_info["widget"]
            type_hint = arg_def.get("type_hint", "str")
            val_conf = global_default_args.get(param_key)
            core_def = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].get(param_key)
            final_val = val_conf if val_conf is not None else core_def
            if type_hint in ["bool", "bool_flag"]:
                bool_val = False
                if isinstance(final_val, bool):
                    bool_val = final_val
                elif isinstance(final_val, str):
                    bool_val = final_val.lower() == 'true'
                if hasattr(widget, 'variable'):
                    widget.variable.set(bool_val)
                elif isinstance(widget, ctk.CTkCheckBox):
                    if bool_val:
                        widget.select()
                    else:
                        widget.deselect()
            elif isinstance(widget, ctk.CTkEntry):
                widget.delete(0, "end")
                if final_val is not None:
                    widget.insert(0, str(final_val))
        if hasattr(self, 'exe_path_entry'):
            self.exe_path_entry.delete(0, "end")
            self.exe_path_entry.insert(0, self.koboldcpp_executable)
        if hasattr(self, 'auto_open_webui_var'):
            self.auto_open_webui_var.set(self.config.get("auto_open_webui", True))
        self.log_to_console("Settings tab populated.")

    def save_config(self):
        if hasattr(self, 'exe_path_entry'):
            self.config["koboldcpp_executable"] = self.exe_path_entry.get().strip()
        else:
            self.config["koboldcpp_executable"] = self.koboldcpp_executable
        self.config["default_gguf_dir"] = self.default_model_dir
        self.config["db_file"] = self.db_path
        self.config["color_mode"] = ctk.get_appearance_mode().lower()
        if hasattr(self, 'auto_open_webui_var'):
            self.config["auto_open_webui"] = self.auto_open_webui_var.get()
        current_global_defaults = self.config.get("default_args", {}).copy()
        for arg_def in koboldcpp_core.KOBOLDCPP_ARG_DEFINITIONS:
            param_key = arg_def["key"]
            if param_key not in self.settings_widgets:
                continue
            widget_info = self.settings_widgets[param_key]
            widget = widget_info["widget"]
            type_hint = arg_def.get("type_hint", "str")
            if type_hint in ["bool", "bool_flag"]:
                if hasattr(widget, 'variable'):
                    current_global_defaults[param_key] = widget.variable.get()
                elif isinstance(widget, ctk.CTkCheckBox):
                    current_global_defaults[param_key] = widget.get() == 1
            elif isinstance(widget, ctk.CTkEntry):
                current_global_defaults[param_key] = widget.get().strip()
        self.config["default_args"] = current_global_defaults
        if "model_specific_args" not in self.config:
            self.config["model_specific_args"] = {}
        success, message = koboldcpp_core.save_launcher_config(self.config)
        if success:
            self.log_to_console(message)
            self.koboldcpp_executable = self.config["koboldcpp_executable"]
            self.default_model_dir = self.config.get("default_gguf_dir", "")
            self.db_path = self.config.get("db_file", self.db_path)
        else:
            self.log_to_console(f"Error saving config: {message}")
            messagebox.showerror("Save Error", f"Could not save: {message}")
        return success

    def _show_model_selection_view(self):
        if hasattr(self, 'tuning_mode_frame'):
            self.tuning_mode_frame.grid_remove()
        if hasattr(self, 'model_selection_frame'):
            self.model_selection_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.log_to_console("Switched to Model Selection view.")

    def _show_tuning_mode_view(self):
        if hasattr(self, 'model_selection_frame'):
            self.model_selection_frame.grid_remove()
        if hasattr(self, 'tuning_mode_frame'):
            self.tuning_mode_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.log_to_console("Switched to Tuning Mode view.")

    def adjust_ot_level(self, delta):
        if not self.tuning_in_progress:
            return
        self.current_tuning_attempt_level += delta
        self.update_tuning_display()

    def _create_args_dialog_content(self, parent, current_args, param_defs):
        scrollable = ctk.CTkScrollableFrame(parent, width=700, height=500)
        scrollable.pack(fill="both", expand=True, padx=10, pady=10)
        widgets = {}
        for setting_def in param_defs:
            param = setting_def["key"]
            if param == "--model":
                continue
            val = current_args.get(param)
            type_hint = setting_def.get("type_hint", "str")
            frame = ctk.CTkFrame(scrollable)
            frame.pack(fill="x", pady=2)
            ctk.CTkLabel(frame, text=f"{setting_def['name']}:", width=180, anchor="w").pack(side="left", padx=5)
            if type_hint in ["bool", "bool_flag"]:
                bool_val = False
                if isinstance(val, bool):
                    bool_val = val
                elif isinstance(val, str):
                    bool_val = val.lower() == 'true'
                var = ctk.BooleanVar(value=bool_val)
                widget = ctk.CTkCheckBox(frame, text="", variable=var)
                widget.pack(side="left", padx=5)
                widget.variable = var
                widgets[param] = {"widget": widget, "type_hint": type_hint}
            else:
                entry = ctk.CTkEntry(frame, width=150)
                if val is not None:
                    entry.insert(0, str(val))
                entry.pack(side="left", padx=5, fill="x", expand=False)
                widgets[param] = {"widget": entry, "type_hint": type_hint}
            ctk.CTkLabel(frame, text=setting_def.get("help", ""), font=ctk.CTkFont(size=10), text_color="gray").pack(side="left", padx=5, fill="x", expand=True)
        return widgets

    def _get_param_definitions_for_dialog(self):
        return koboldcpp_core.KOBOLDCPP_ARG_DEFINITIONS

    def edit_base_args_for_tuning_session(self):
        if not self.tuning_in_progress:
            return
        dialog = ctk.CTkToplevel(self)
        dialog.title("Edit Base Args (Session)")
        dialog.geometry("750x650")
        dialog.transient(self)
        dialog.grab_set()
        param_defs = self._get_param_definitions_for_dialog()
        effective_base = self._get_merged_args_for_model(self.current_tuning_model_path)
        display_args = effective_base.copy()
        display_args.update(self.current_tuning_session_base_args)
        widgets_info = self._create_args_dialog_content(dialog, display_args, param_defs)
        def save_action():
            for param, info in widgets_info.items():
                widget = info["widget"]
                type_hint = info["type_hint"]
                if type_hint in ["bool", "bool_flag"]:
                    self.current_tuning_session_base_args[param] = widget.variable.get()
                else:
                    val_str = widget.get().strip()
                    if val_str:
                        self.current_tuning_session_base_args[param] = val_str
                    elif param in self.current_tuning_session_base_args:
                        del self.current_tuning_session_base_args[param]
            self.log_to_console("Session base args updated.")
            self.update_tuning_display()
            dialog.destroy()
        bf = ctk.CTkFrame(dialog)
        bf.pack(fill="x", pady=10)
        ctk.CTkButton(bf, text="Save Session Args", command=save_action).pack(side="left", padx=10)
        ctk.CTkButton(bf, text="Cancel", command=dialog.destroy).pack(side="right", padx=10)
        dialog.wait_window()

    def edit_permanent_model_args(self):
        if not self.tuning_in_progress or not self.current_tuning_model_path:
            messagebox.showwarning("Not Available", "Start tuning with a model.")
            return
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"Edit Permanent Args: {os.path.basename(self.current_tuning_model_path)}")
        dialog.geometry("750x650")
        dialog.transient(self)
        dialog.grab_set()
        param_defs = self._get_param_definitions_for_dialog()
        display_args = self._get_merged_args_for_model(self.current_tuning_model_path)
        widgets_info = self._create_args_dialog_content(dialog, display_args, param_defs)
        def save_action():
            model_key = self.current_tuning_model_path
            if model_key not in self.config["model_specific_args"]:
                self.config["model_specific_args"][model_key] = {}
            model_specifics = self.config["model_specific_args"][model_key]
            global_baseline = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
            global_baseline.update(self.config.get("default_args", {}))
            for param, info in widgets_info.items():
                widget = info["widget"]
                type_hint = info["type_hint"]
                global_val = global_baseline.get(param)
                if type_hint in ["bool", "bool_flag"]:
                    dialog_val = widget.variable.get()
                    global_bool = False
                    if isinstance(global_val, bool):
                        global_bool = global_val
                    elif isinstance(global_val, str):
                        global_bool = global_val.lower() == 'true'
                    if dialog_val != global_bool:
                        model_specifics[param] = dialog_val
                    elif param in model_specifics:
                        del model_specifics[param]
                else:
                    dialog_str = widget.get().strip()
                    if not dialog_str:
                        if param in model_specifics:
                            del model_specifics[param]
                    else:
                        global_str = str(global_val) if global_val is not None else ""
                        if dialog_str != global_str:
                            model_specifics[param] = dialog_str
                        elif param in model_specifics:
                            del model_specifics[param]
            if not self.config["model_specific_args"][model_key]:
                del self.config["model_specific_args"][model_key]
            if self.save_config():
                self.log_to_console(f"Permanent args saved for {os.path.basename(model_key)}")
                self._reinitialize_session_base_args()
                self.update_tuning_display()
            else:
                self.log_to_console("Failed to save permanent args.")
            dialog.destroy()
        bf = ctk.CTkFrame(dialog)
        bf.pack(fill="x", pady=10)
        ctk.CTkButton(bf, text="Save Permanent Args", command=save_action).pack(side="left", padx=10)
        ctk.CTkButton(bf, text="Cancel", command=dialog.destroy).pack(side="right", padx=10)
        dialog.wait_window()

    def select_new_gguf_during_tuning(self):
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
            if not messagebox.askyesno("Process Running", "KCPP monitoring process might be running. Stop it and select new model?"):
                return
            self.log_to_console("Stopping monitored KCPP.")
            koboldcpp_core.kill_process(self.kcpp_process_obj.pid, force=True)
            self.kcpp_process_obj = None
        self.end_tuning_session(switch_to_model_selection=True)
        self.browse_model()

    def end_tuning_session(self, switch_to_model_selection=True):
        if not self.tuning_in_progress:
            return
        self.log_to_console("Ending tuning session.")
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
            self.log_to_console(f"Stopping KCPP (PID: {self.kcpp_process_obj.pid}) from tuning...")
            koboldcpp_core.kill_process(self.kcpp_process_obj.pid, force=True)
            self.kcpp_process_obj = None
        self.tuning_in_progress = False
        self.current_tuning_model_path = None
        self.current_tuning_model_analysis = {}
        self.current_tuning_session_base_args = {}
        self.last_free_vram_after_load_mb = None
        self.last_approx_vram_used_kcpp_mb = None
        self.last_successful_monitored_run_details_gui = None
        self._return_to_full_tuning_menu()
        if switch_to_model_selection:
            self._show_model_selection_view()

    def launch_and_monitor_for_tuning(self):
        if not self.tuning_in_progress:
            messagebox.showerror("Error", "Tuning session not active.")
            return
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
            messagebox.showwarning("Process Running", "A KCPP process is already being monitored.")
            return
        self.log_to_console(f"Tuning: Launch & Monitor OT Level {self.current_tuning_attempt_level}")
        if hasattr(self, 'kcpp_live_output_text'):
            self.kcpp_live_output_text.configure(state="normal")
            self.kcpp_live_output_text.delete("1.0", "end")
            self.kcpp_live_output_text.insert("1.0", "Preparing KCPP...\n")
            self.kcpp_live_output_text.configure(state="disabled")
        self._set_tuning_buttons_state("disabled")
        self.kcpp_success_event.clear()
        self.kcpp_oom_event.clear()
        self.kcpp_output_lines_shared.clear()
        self.last_free_vram_after_load_mb = None
        self.last_approx_vram_used_kcpp_mb = None
        self.level_of_last_monitored_run = self.current_tuning_attempt_level
        ot_string = koboldcpp_core.generate_overridetensors(self.current_tuning_model_analysis, self.current_tuning_attempt_level)
        args_list = koboldcpp_core.build_command(self.current_tuning_model_path, ot_string, self.current_tuning_model_analysis, self.current_tuning_session_base_args)
        self.current_command_list_for_db = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, args_list)
        self.vram_at_decision_for_db, _, _, _ = koboldcpp_core.get_available_vram_mb()
        self.kcpp_process_obj, err_msg = koboldcpp_core.launch_process(self.current_command_list_for_db, capture_output=True, new_console=False, use_text_mode=False)
        if err_msg or not self.kcpp_process_obj:
            self.log_to_console(f"Failed to launch KCPP: {err_msg or 'Unknown'}")
            self._log_to_kcpp_live_output(f"LAUNCH ERROR: {err_msg or 'Unknown'}\n")
            koboldcpp_core.save_config_to_db(self.db_path, self.current_tuning_model_path, self.current_tuning_model_analysis, self.vram_at_decision_for_db, self.current_command_list_for_db, self.level_of_last_monitored_run, "LAUNCH_FAILED_SETUP_GUI", None)
            self._set_tuning_buttons_state("normal")
            return
        self._log_to_kcpp_live_output(f"KCPP process started (PID: {self.kcpp_process_obj.pid}). Monitoring...\n")
        effective_args = {**self.config.get("default_args", {}), **self.current_tuning_session_base_args}
        target_port = effective_args.get("--port", "5000")
        success_pattern = self.config.get("kobold_success_pattern", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["kobold_success_pattern"])
        oom_keywords = self.config.get("oom_error_keywords", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["oom_error_keywords"])
        self.kcpp_monitor_thread = threading.Thread(target=self._monitor_kcpp_output_thread_target, args=(self.kcpp_process_obj, success_pattern, oom_keywords, target_port), daemon=True)
        self.kcpp_monitor_thread.start()
        self.monitor_start_time = time.monotonic()
        self._poll_monitor_status()

    def _log_to_kcpp_live_output(self, text_line: str):
        def _update():
            if hasattr(self, 'kcpp_live_output_text') and self.kcpp_live_output_text.winfo_exists():
                self.kcpp_live_output_text.configure(state="normal")
                self.kcpp_live_output_text.insert("end", text_line)
                self.kcpp_live_output_text.see("end")
                self.kcpp_live_output_text.configure(state="disabled")
        if hasattr(self, 'after'):
            self.after(0, _update)

    def _monitor_kcpp_output_thread_target(self, process, success_regex, oom_keywords, target_port):
        try:
            for line_bytes in iter(process.stdout.readline, b''):
                if not line_bytes:
                    break
                try:
                    line_decoded = line_bytes.decode('utf-8', errors='replace')
                except UnicodeDecodeError:
                    line_decoded = line_bytes.decode('latin-1', errors='replace')
                self._log_to_kcpp_live_output(line_decoded)
                line_strip = line_decoded.strip().lower()
                if line_strip:
                    self.kcpp_output_lines_shared.append(line_decoded.strip())
                    if not self.kcpp_success_event.is_set() and not self.kcpp_oom_event.is_set():
                        match = re.search(success_regex, line_decoded.strip(), re.IGNORECASE)
                        if match:
                            port = target_port
                            try:
                                port = match.group(1)
                            except IndexError:
                                pass
                            if str(port) == str(target_port):
                                self.kcpp_success_event.set()
                        if not self.kcpp_success_event.is_set():
                            for kw in oom_keywords:
                                if kw.lower() in line_strip:
                                    self.kcpp_oom_event.set()
                                    break
                if self.kcpp_success_event.is_set() or self.kcpp_oom_event.is_set():
                    break
        except Exception as e:
            self._log_to_kcpp_live_output(f"\nError in KCPP monitor: {type(e).__name__}: {e}\n")
        finally:
            if process.stdout and not process.stdout.closed:
                try:
                    process.stdout.close()
                except:
                    pass
            self._log_to_kcpp_live_output("\nKCPP output monitoring finished.\n")

    def _poll_monitor_status(self):
        timeout = self.config.get("loading_timeout_seconds", 60)
        elapsed = time.monotonic() - self.monitor_start_time
        exited = False
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is not None:
            if not self.kcpp_success_event.is_set() and not self.kcpp_oom_event.is_set():
                exited = True
        if self.kcpp_success_event.is_set():
            self._handle_monitoring_completion("SUCCESS_LOAD_DETECTED_GUI")
        elif self.kcpp_oom_event.is_set():
            self._handle_monitoring_completion("OOM_CRASH_DETECTED_GUI")
        elif exited:
            self._handle_monitoring_completion("PREMATURE_EXIT_GUI")
        elif elapsed > timeout:
            self._handle_monitoring_completion("TIMEOUT_NO_SIGNAL_GUI")
        else:
            self.after(250, self._poll_monitor_status)

    def _set_tuning_buttons_state(self, state="normal"):
        buttons = [getattr(self, name, None) for name in ['btn_tune_launch_monitor', 'btn_tune_skip_launch_direct', 'btn_tune_more_gpu', 'btn_tune_more_cpu', 'btn_tune_edit_args', 'btn_tune_edit_model_perm_args', 'btn_tune_new_gguf', 'btn_tune_quit_tuning']]
        for btn in buttons:
            if btn and hasattr(btn, 'winfo_exists') and btn.winfo_exists():
                btn.configure(state=state)
        if state == "normal" and self.tuning_in_progress:
            if hasattr(self, 'btn_tune_more_gpu'):
                self.btn_tune_more_gpu.configure(state="normal" if self.current_tuning_attempt_level > self.current_tuning_min_level else "disabled")
            if hasattr(self, 'btn_tune_more_cpu'):
                self.btn_tune_more_cpu.configure(state="normal" if self.current_tuning_attempt_level < self.current_tuning_max_level else "disabled")
            if hasattr(self, 'btn_tune_edit_model_perm_args'):
                self.btn_tune_edit_model_perm_args.configure(state="normal" if self.current_tuning_model_path else "disabled")

    def _handle_post_monitor_action(self, action_key, original_outcome):
        self.log_to_console(f"User action: '{action_key}' for outcome '{original_outcome}'")
        ot_string = koboldcpp_core.generate_overridetensors(self.current_tuning_model_analysis, self.current_tuning_attempt_level)
        args_list = koboldcpp_core.build_command(self.current_tuning_model_path, ot_string, self.current_tuning_model_analysis, self.current_tuning_session_base_args)
        cmd_to_run = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, args_list)
        db_suffix = "_GUI"
        if action_key == "launch_for_use":
            db_suffix = "_USER_ACCEPTED_TUNED_GUI"
        elif action_key == "launch_for_use_risky":
            db_suffix = "_USER_LAUNCHED_RISKY_GUI"
        elif action_key == "save_good_more_gpu":
            db_suffix = "_USER_SAVED_GOOD_GPU_GUI"
        elif action_key == "more_gpu_now":
            db_suffix = "_USER_WANTS_MORE_GPU_GUI"
        elif action_key == "auto_adjust_cpu":
            db_suffix = "_USER_AUTO_ADJUST_CPU_GUI"
        elif action_key == "more_cpu_after_fail":
            db_suffix = "_USER_TRIED_CPU_FAIL_GUI"
        elif action_key == "more_cpu_now":
            db_suffix = "_USER_WANTS_MORE_CPU_GUI"
        elif action_key == "return_to_tuning_menu":
            db_suffix = "_USER_RETURNED_MENU_GUI"

        if action_key in ["launch_for_use", "launch_for_use_risky"]:
            self._launch_final_koboldcpp(cmd_to_run, original_outcome + db_suffix, self.level_of_last_monitored_run)
            self.end_tuning_session()
            return
        if action_key in ["save_good_more_gpu", "more_gpu_now"]:
            if self.current_tuning_attempt_level > self.current_tuning_min_level:
                self.current_tuning_attempt_level -= 1
            else:
                self.log_to_console("At Max GPU.")
        elif action_key in ["auto_adjust_cpu", "more_cpu_after_fail", "more_cpu_now"]:
            if self.current_tuning_attempt_level < self.current_tuning_max_level:
                self.current_tuning_attempt_level += 1
            else:
                self.log_to_console("At Max CPU.")
        final_db_outcome = original_outcome + db_suffix
        vram_val = self.vram_at_decision_for_db if self.vram_at_decision_for_db is not None else 0
        koboldcpp_core.save_config_to_db(self.db_path, self.current_tuning_model_path, self.current_tuning_model_analysis, vram_val, self.current_command_list_for_db, self.level_of_last_monitored_run, final_db_outcome, self.last_approx_vram_used_kcpp_mb)
        self.load_history()
        self._return_to_full_tuning_menu()

    def _launch_final_koboldcpp(self, command_list, db_outcome, attempt_level):
        if hasattr(self, 'exe_path_entry'):
            self.koboldcpp_executable = self.exe_path_entry.get().strip()
        self.check_koboldcpp_executable()
        if not os.path.exists(self.koboldcpp_executable):
            messagebox.showerror("Not Found", f"KCPP exe not found: {self.koboldcpp_executable}")
            return
        self.log_to_console(f"Launching KCPP. Outcome: {db_outcome}")
        vram_now, _, _, _ = koboldcpp_core.get_available_vram_mb()
        model_p = self.current_tuning_model_path or self.current_model_path
        model_a = self.current_tuning_model_analysis or self.model_analysis_info
        if command_list[0].lower() == sys.executable.lower() and command_list[1].lower().endswith(".py"):
            command_list[1] = self.koboldcpp_executable
        else:
            command_list[0] = self.koboldcpp_executable
        koboldcpp_core.save_config_to_db(self.db_path, model_p, model_a, vram_now, command_list, attempt_level, db_outcome, self.last_approx_vram_used_kcpp_mb)
        self.load_history()
        process, err_msg = koboldcpp_core.launch_process(command_list, capture_output=False, new_console=True)
        if err_msg or not process:
            self.log_to_console(f"Failed to launch KCPP: {err_msg or 'Unknown'}")
            messagebox.showerror("Launch Error", f"Failed to launch KCPP: {err_msg or 'Unknown'}")
            koboldcpp_core.save_config_to_db(self.db_path, model_p, model_a, vram_now, command_list, attempt_level, "LAUNCH_FOR_USE_FAILED_GUI", self.last_approx_vram_used_kcpp_mb)
            self.load_history()
        else:
            self.log_to_console(f"KCPP launched (PID: {process.pid}).")
            self.last_process = process
            self.process_running = True
            if self.config.get("auto_open_webui", True):
                args_dict = koboldcpp_core.args_list_to_dict(command_list)
                port = args_dict.get("--port", self.config.get("default_args", {}).get("--port", "5000"))
                try:
                    port_num = int(port)
                    if 1 <= port_num <= 65535:
                        url = f"http://localhost:{port_num}"
                        self.log_to_console(f"Auto-opening {url}.")
                        threading.Timer(3.0, lambda: webbrowser.open(url)).start()
                    else:
                        self.log_to_console(f"Invalid port '{port}' for auto-open.")
                except ValueError:
                    self.log_to_console(f"Invalid port value '{port}' for auto-open.")
            else:
                self.log_to_console("Auto-open web UI disabled.")

    def skip_tune_and_launch_direct(self):
        if not self.tuning_in_progress:
            messagebox.showwarning("Not Tuning", "Tuning session not active.")
            return
        self.log_to_console("User skipping tune, launching current config.")
        if not hasattr(self, 'vram_at_decision_for_db') or self.vram_at_decision_for_db is None:
            self.vram_at_decision_for_db, _,_,_ = koboldcpp_core.get_available_vram_mb()
        ot_string = koboldcpp_core.generate_overridetensors(self.current_tuning_model_analysis, self.current_tuning_attempt_level)
        args_list = koboldcpp_core.build_command(self.current_tuning_model_path, ot_string, self.current_tuning_model_analysis, self.current_tuning_session_base_args)
        cmd_to_run = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, args_list)
        self._launch_final_koboldcpp(cmd_to_run, "SUCCESS_USER_DIRECT_LAUNCH_GUI", self.current_tuning_attempt_level)
        self.end_tuning_session()

    def setup_settings_tab(self):
        sf = ctk.CTkScrollableFrame(self.tab_settings)
        sf.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        tf = ctk.CTkFrame(sf)
        tf.pack(fill="x", expand=True, padx=10, pady=10)
        ctk.CTkLabel(tf, text="UI Theme:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        theme_var = ctk.StringVar(value=ctk.get_appearance_mode().lower())
        ctk.CTkOptionMenu(tf, values=["dark", "light", "system"], variable=theme_var, command=self.change_theme).grid(row=0, column=1, padx=10, pady=10)
        ef = ctk.CTkFrame(sf)
        ef.pack(fill="x", expand=True, padx=10, pady=10)
        ef.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(ef, text="KCPP Exe:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.exe_path_entry = ctk.CTkEntry(ef, width=400)
        self.exe_path_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(ef, text="Browse", command=self.browse_executable).grid(row=0, column=2, padx=10, pady=10)
        wf = ctk.CTkFrame(sf)
        wf.pack(fill="x", expand=True, padx=10, pady=5)
        ctk.CTkLabel(wf, text="Launcher Behavior:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.auto_open_webui_var = ctk.BooleanVar(value=self.config.get("auto_open_webui", True))
        self.auto_open_webui_checkbox = ctk.CTkCheckBox(wf, text="Auto-Open Web UI", variable=self.auto_open_webui_var)
        self.auto_open_webui_checkbox.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        ctk.CTkLabel(sf, text="Global KCPP Defaults", font=ctk.CTkFont(size=16, weight="bold")).pack(fill="x", expand=True, padx=10, pady=(20,10))
        self.settings_widgets = {}
        for sd in koboldcpp_core.KOBOLDCPP_ARG_DEFINITIONS:
            pk = sd["key"]
            if pk == "--model":
                continue
            core_def_val = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].get(pk)
            type_hint = sd.get("type_hint", "str")
            frame = ctk.CTkFrame(sf)
            frame.pack(fill="x", expand=True, padx=10, pady=5)
            frame.grid_columnconfigure(1, weight=0)
            frame.grid_columnconfigure(2, weight=1)
            ctk.CTkLabel(frame, text=f"{sd['name']}:", width=180, anchor="w").grid(row=0, column=0, padx=10, pady=(5,0), sticky="w")
            widget_inst = None
            if type_hint in ["bool", "bool_flag"]:
                init_bool = False
                if isinstance(core_def_val, bool):
                    init_bool = core_def_val
                elif isinstance(core_def_val, str):
                    init_bool = core_def_val.lower() == 'true'
                var = ctk.BooleanVar(value=init_bool)
                widget_inst = ctk.CTkCheckBox(frame, text="", variable=var)
                widget_inst.grid(row=0, column=1, padx=10, pady=(5,0), sticky="w")
                widget_inst.variable = var
            else:
                widget_inst = ctk.CTkEntry(frame, width=120)
                if core_def_val is not None:
                    widget_inst.insert(0, str(core_def_val))
                widget_inst.grid(row=0, column=1, padx=10, pady=(5,0), sticky="w")
            ctk.CTkLabel(frame, text=sd['help'], font=ctk.CTkFont(size=11), text_color="gray", anchor="w", justify="left", wraplength=350).grid(row=0, column=2, padx=10, pady=(5,0), sticky="w")
            self.settings_widgets[pk] = {"widget": widget_inst, "type_hint": type_hint}
        ctk.CTkButton(sf, text="Save Settings", command=self.save_settings_action).pack(pady=20)
        ctk.CTkButton(sf, text="Reset All to Defaults", command=self.reset_config_action, fg_color="#dc3545", hover_color="#c82333").pack(pady=10)

    def browse_executable(self):
        curr_dir = os.getcwd()
        curr_exe = self.exe_path_entry.get().strip() if hasattr(self, 'exe_path_entry') else self.koboldcpp_executable
        if curr_exe and os.path.exists(os.path.dirname(curr_exe)):
            curr_dir = os.path.dirname(curr_exe)
        types = [("Executables", "*.exe" if sys.platform == "win32" else "*"), ("Py Scripts", "*.py"), ("All", "*.*")]
        fp = filedialog.askopenfilename(title="Select KCPP Exe/Script", filetypes=types, initialdir=curr_dir)
        if fp and hasattr(self, 'exe_path_entry'):
            self.exe_path_entry.delete(0, "end")
            self.exe_path_entry.insert(0, fp)
            self.log_to_console(f"Selected KCPP exe (pending save): {fp}")

    def setup_history_tab(self):
        hf = ctk.CTkFrame(self.tab_history)
        hf.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        hf.grid_columnconfigure(0,weight=1)
        hf.grid_rowconfigure(1,weight=1)
        ctk.CTkLabel(hf, text="Launch History", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0,column=0,padx=10,pady=10,sticky="w")
        self.history_text = ctk.CTkTextbox(hf)
        self.history_text.grid(row=1,column=0,padx=10,pady=10,sticky="nsew")
        self.history_text.configure(state="disabled")
        ctk.CTkButton(hf, text="Refresh History", command=self.load_history).grid(row=2,column=0,padx=10,pady=10)
        self.after(500, self.load_history)

    def reset_config_action(self):
        if messagebox.askyesno("Reset Config", "Reset all settings to defaults? This will overwrite your config (backup attempted)."):
            try:
                core_conf_path = koboldcpp_core.CONFIG_FILE
                if os.path.exists(core_conf_path):
                    shutil.copy2(core_conf_path, core_conf_path + ".backup_" + time.strftime("%Y%m%d_%H%M%S"))
                    self.log_to_console("Backed up config.")
                self.config = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE.copy()
                self.config["default_args"] = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["default_args"].copy()
                db_basename = koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["db_file"]
                data_dir = koboldcpp_core._get_user_app_data_dir()
                self.config["db_file"] = os.path.join(data_dir, db_basename)
                os.makedirs(os.path.dirname(self.config["db_file"]), exist_ok=True)
                for k,v in koboldcpp_core.DEFAULT_CONFIG_TEMPLATE.items():
                    if isinstance(v,dict) and k not in ["default_args", "db_file"]:
                        self.config[k] = v.copy()
                self.config["first_run_completed"] = False
                self.config["first_run_intro_shown"] = False
                koboldcpp_core.save_launcher_config(self.config)
                reinit_res = koboldcpp_core.initialize_launcher()
                self.config = reinit_res["config"]
                self.system_info = reinit_res["system_info"]
                self.gpu_info = reinit_res["gpu_info"]
                self.koboldcpp_executable = self.config.get("koboldcpp_executable")
                self.default_model_dir = self.config.get("default_gguf_dir", "")
                self.db_path = self.config["db_file"]
                self.load_settings_from_config()
                self.log_to_console(f"Config reset. DB: {self.db_path}")
                messagebox.showinfo("Config Reset", "Config reset. First-time setup may re-run.")
                self.after(100, self._run_first_time_setup_if_needed)
            except Exception as e:
                self.log_to_console(f"Error resetting config: {e}")
                messagebox.showerror("Error", f"Error resetting: {e}")

    def launch_direct_defaults(self):
        if not self.current_model_path:
            messagebox.showwarning("No Model", "Select GGUF model.")
            return
        self.check_koboldcpp_executable()
        if not os.path.exists(self.koboldcpp_executable):
            messagebox.showerror("Not Found", f"KCPP not found: {self.koboldcpp_executable}")
            return
        self.log_to_console(f"Direct launch: {os.path.basename(self.current_model_path)}")
        eff_args = self._get_merged_args_for_model(self.current_model_path)
        analysis = self.model_analysis_info if (self.model_analysis_info and 'filepath' in self.model_analysis_info) else koboldcpp_core.analyze_filename(self.current_model_path)
        args_list = koboldcpp_core.build_command(self.current_model_path, None, analysis, eff_args)
        cmd_list = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, args_list)
        self.vram_at_decision_for_db,_,_,_ = koboldcpp_core.get_available_vram_mb()
        self.last_approx_vram_used_kcpp_mb = None
        self._launch_final_koboldcpp(cmd_list, "SUCCESS_USER_DIRECT_SETTINGS_GUI", 0)

    def launch_best_remembered(self):
        if not self.current_model_path:
            messagebox.showwarning("No Model", "Select GGUF model.")
            return
        if not self.model_analysis_info or not self.model_analysis_info.get('filepath'):
            self.analyze_model_action(self.current_model_path)
        if not self.model_analysis_info or not self.model_analysis_info.get('filepath'):
            messagebox.showwarning("Not Analyzed", "Could not analyze.")
            return
        self.check_koboldcpp_executable()
        if not os.path.exists(self.koboldcpp_executable):
            messagebox.showerror("Not Found", f"KCPP not found: {self.koboldcpp_executable}")
            return
        self.log_to_console("Launching best remembered...")
        vram,_,_,_ = koboldcpp_core.get_available_vram_mb()
        best_conf = koboldcpp_core.find_best_historical_config(self.db_path, self.model_analysis_info, vram, self.config)
        if best_conf and best_conf.get("args_list"):
            self.log_to_console(f"Found best remembered (Lvl: {best_conf['attempt_level']}, Outcome: {best_conf['outcome']})")
            hist_args_dict = koboldcpp_core.args_list_to_dict(best_conf["args_list"])
            hist_ot = hist_args_dict.pop("--overridetensors", None)
            base_args = self._get_merged_args_for_model(self.current_model_path)
            final_args = base_args.copy()
            final_args.update(hist_args_dict)
            final_list = koboldcpp_core.build_command(self.current_model_path, hist_ot, self.model_analysis_info, final_args)
            cmd_list = koboldcpp_core.get_command_to_run(self.koboldcpp_executable, final_list)
            self.vram_at_decision_for_db = vram
            self.last_approx_vram_used_kcpp_mb = best_conf.get("approx_vram_used_kcpp_mb")
            level = best_conf.get("attempt_level",0)
            self._launch_final_koboldcpp(cmd_list, "SUCCESS_USER_LAUNCHED_BEST_REMEMBERED_GUI", level)
        else:
            self.log_to_console("No remembered config. Fallback to defaults.")
            messagebox.showinfo("Not Found", "No remembered config. Launching with defaults.")
            self.launch_direct_defaults()

    def stop_all_kcpp_processes_forcefully(self):
        self.log_to_console("Force stop ALL KCPP-related processes...")
        if self.kcpp_process_obj and self.kcpp_process_obj.poll() is None:
            self.log_to_console(f"Stopping monitored KCPP (PID: {self.kcpp_process_obj.pid})...")
            koboldcpp_core.kill_process(self.kcpp_process_obj.pid, force=True)
            self.kcpp_process_obj = None
        if self.last_process and self.last_process.poll() is None:
            self.log_to_console(f"Stopping last direct KCPP (PID: {self.last_process.pid})...")
            koboldcpp_core.kill_process(self.last_process.pid, force=True)
            self.last_process = None
        self.process_running = False
        if self.tuning_in_progress:
            self.end_tuning_session(switch_to_model_selection=False)
        exe_sweep = self.config.get("koboldcpp_executable", koboldcpp_core.DEFAULT_CONFIG_TEMPLATE["koboldcpp_executable"])
        exe_basename = os.path.basename(exe_sweep) if exe_sweep else ""
        killed = False
        if exe_basename:
            killed, msg = koboldcpp_core.kill_processes_by_name(exe_basename)
            self.log_to_console(f"Sweep for '{exe_basename}': {msg}")
            if exe_basename.lower().endswith(".py"):
                killed_py, msg_py = koboldcpp_core.kill_processes_by_name("python", cmdline_substr_filter=exe_basename)
                self.log_to_console(f"Sweep for python '{exe_basename}': {msg_py}")
                killed = killed or killed_py
        if killed:
            self.log_to_console("Force sweep done. Processes may have been terminated.")
        else:
            self.log_to_console("Force sweep attempted. No processes explicitly killed.")
        self.refresh_vram()

    def change_theme(self, theme):
        ctk.set_appearance_mode(theme.lower())
        self.config["color_mode"] = theme.lower()
        self.save_config()
        self.log_to_console(f"Theme: {theme}.")

    def log_to_console(self, text):
        def _log():
            if hasattr(self,'console') and self.console.winfo_exists():
                self.console.configure(state="normal")
                ts = time.strftime("%H:%M:%S")
                self.console.insert("end",f"[{ts}] {text}\n")
                self.console.see("end")
                self.console.configure(state="disabled")
        if hasattr(self,'after'):
            self.after(0,_log)
        else:
            print(f"LOG: {text}")

    def monitor_vram(self):
        while True:
            try:
                free_mb, total_mb, message_from_core, gpu_info_dict = koboldcpp_core.get_available_vram_mb()
                used_mb = total_mb - free_mb if total_mb > 0 else 0.0
                if hasattr(self, 'after'):
                    self.after(0, lambda u=used_mb, t=total_mb, msg=message_from_core: self.update_vram_display(u, t, msg))
            except Exception as e:
                print(f"Error in VRAM monitor thread: {e}")
                if hasattr(self, 'after'):
                    self.after(0, lambda: self.update_vram_display(0,0, "VRAM monitor error"))
            time.sleep(5)
            
    def update_vram_display(self, used_mb: float, total_mb: float, message_from_core: str = ""):
        if hasattr(self, 'vram_progress') and self.vram_progress.winfo_exists():
            if total_mb > 0:
                percentage = used_mb / total_mb
                self.vram_progress.set(percentage)
                self.vram_text.configure(text=f"{message_from_core}")

                progress_color = "#28a745"
                if percentage > 0.9:
                    progress_color = "#dc3545"
                elif percentage > 0.7:
                    progress_color = "#ffc107"
                self.vram_progress.configure(progress_color=progress_color)
            else:
                self.vram_progress.set(0)
                self.vram_text.configure(text=f"{message_from_core if message_from_core else 'VRAM info unavailable'}")

        if hasattr(self, 'tuning_view_vram_progress') and self.tuning_view_vram_progress.winfo_exists():
            if total_mb > 0:
                percentage = used_mb / total_mb
                self.tuning_view_vram_progress.set(percentage)
                if hasattr(self, 'tuning_view_vram_text') and self.tuning_view_vram_text.winfo_exists():
                    self.tuning_view_vram_text.configure(text=f"{message_from_core}")

                progress_color = "#28a745"
                if percentage > 0.9:
                    progress_color = "#dc3545"
                elif percentage > 0.7:
                    progress_color = "#ffc107"
                self.tuning_view_vram_progress.configure(progress_color=progress_color)
            else:
                self.tuning_view_vram_progress.set(0)
                if hasattr(self, 'tuning_view_vram_text') and self.tuning_view_vram_text.winfo_exists():
                    self.tuning_view_vram_text.configure(text=f"{message_from_core if message_from_core else 'VRAM info unavailable'}")
                    
    def refresh_vram(self):
        self.log_to_console("Refreshing VRAM...")
        try:
            free,total,msg,_ = koboldcpp_core.get_available_vram_mb()
            used = total-free if total>0 else 0.0
            self.update_vram_display(used,total,msg)
            self.log_to_console(f"VRAM Refreshed: {msg}")
        except Exception as e:
            self.log_to_console(f"Error VRAM refresh: {e}")
            self.update_vram_display(0,0,"Error VRAM refresh")

    def save_settings_action(self):
        if self.save_config():
            messagebox.showinfo("Settings Saved", "Global settings saved!")

    def load_history(self, model_filter=None):
        if not hasattr(self,'history_text') or not self.history_text.winfo_exists():
            return
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0","end")
        if not os.path.exists(self.db_path):
            self.history_text.insert("end",f"No history DB at {self.db_path}.")
        else:
            try:
                all_rows = koboldcpp_core.get_history_entries(self.db_path,limit=200)
                rows_disp = [r for r in all_rows if r[0]==model_filter] if model_filter else all_rows
                title = f"Launches for {os.path.basename(model_filter)} (max 50):\n\n" if model_filter else "Global Recent Launches (max 50):\n\n"
                if not rows_disp and model_filter:
                    title = f"No history for {os.path.basename(model_filter)}.\n\n"
                self.history_text.insert("end",title)
                for i,row in enumerate(rows_disp[:50]):
                    name = os.path.basename(row[0])
                    sz = f"{row[1]:.1f}" if isinstance(row[1],float) else (str(row[1]) if row[1] else "N/A")
                    q = str(row[2]) if row[2] else "N/A"
                    moe = "Y" if row[3] else "N"
                    vl = str(row[4]) if row[4] is not None else "N/A"
                    lvl = str(row[5]) if row[5] is not None else "N/A"
                    out = str(row[6]) if row[6] else "N/A"
                    vu = str(row[7]) if row[7] is not None else "N/A"
                    ts_obj = row[8]
                    ts = ts_obj.strftime('%y-%m-%d %H:%M') if isinstance(ts_obj,koboldcpp_core.datetime) else str(ts_obj)
                    entry = (f"{i+1}. {name}\n   Sz:{sz}B, Q:{q}, MoE:{moe}\n   VRAM@L:{vl}MB, Lvl:{lvl}, Outcome:{out}\n   KCPP VRAM:{vu}MB, Time:{ts}\n\n")
                    self.history_text.insert("end",entry)
                if not rows_disp and not model_filter:
                    self.history_text.insert("1.0","No launch history.")
            except Exception as e:
                self.history_text.insert("end",f"Error loading history: {e}")
        self.history_text.configure(state="disabled")

    def browse_model(self):
        initial_dir_to_use = self.config.get("last_used_gguf_dir", self.default_model_dir)
        if not initial_dir_to_use or not os.path.isdir(initial_dir_to_use):
            initial_dir_to_use = os.getcwd()

        filepath = filedialog.askopenfilename(
            title="Select GGUF Model File",
            filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")],
            initialdir=initial_dir_to_use
        )
        if filepath:
            self.model_path_entry.delete(0, "end")
            self.model_path_entry.insert(0, filepath)
            self.current_model_path = os.path.abspath(filepath)
            self.default_model_dir = os.path.dirname(self.current_model_path)
            self.config["last_used_gguf_dir"] = self.default_model_dir
            self.save_config()
            self.analyze_model_action(self.current_model_path)
            
    def analyze_model_action(self, filepath):
        self.log_to_console(f"Analyzing: {os.path.basename(filepath)}")
        self.model_analysis_info = koboldcpp_core.analyze_filename(filepath)
        moe= 'MoE' if self.model_analysis_info.get('is_moe') else 'Dense'
        sz=self.model_analysis_info.get('size_b',"N/A")
        q=self.model_analysis_info.get('quant',"N/A")
        layers=self.model_analysis_info.get('num_layers',"N/A")
        vram_est=self.model_analysis_info.get('estimated_vram_gb_full_gpu',"N/A")
        info=f"Type:{moe}, Size:~{sz}B, Q:{q}, Layers:{layers}, Est.FullVRAM:{vram_est}GB"
        if hasattr(self,'model_info_label'):
            self.model_info_label.configure(text=info)
        self.log_to_console(f"Analysis - {info}")

    def view_history_for_current_model(self): # Added this method stub
        if self.current_tuning_model_path:
            self.log_to_console(f"Viewing history for: {os.path.basename(self.current_tuning_model_path)}")
            self.load_history(model_filter=self.current_tuning_model_path)
            self.tabview.set("History")
        else:
            messagebox.showwarning("No Model", "No model is currently active in the tuning session.")


class FirstTimeSetupDialog(ctk.CTkToplevel):
    def __init__(self, parent, current_config, current_exe_path):
        super().__init__(parent)
        self.title("First-Time Setup")
        self.geometry("600x350")
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.parent_app = parent
        self.config_to_update = current_config
        self.saved_config = None
        ctk.CTkLabel(self, text="Welcome to KoboldCpp Smart Launcher!", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10,5))
        ctk.CTkLabel(self, text="Please verify the path to your KoboldCpp executable.").pack(pady=5)
        exe_frame = ctk.CTkFrame(self)
        exe_frame.pack(fill="x", padx=20, pady=10)
        exe_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(exe_frame, text="KoboldCpp Executable:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.exe_entry = ctk.CTkEntry(exe_frame, width=300)
        self.exe_entry.insert(0, current_exe_path)
        self.exe_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(exe_frame, text="Browse", command=self._browse_exe).grid(row=0, column=2, padx=5, pady=5)
        ctk.CTkLabel(self, text="Optionally, set a default directory for your GGUF models.").pack(pady=5)
        gguf_frame = ctk.CTkFrame(self)
        gguf_frame.pack(fill="x", padx=20, pady=10)
        gguf_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(gguf_frame, text="Default GGUF Directory:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.gguf_dir_entry = ctk.CTkEntry(gguf_frame, width=300)
        self.gguf_dir_entry.insert(0, self.config_to_update.get("default_gguf_dir", os.getcwd()))
        self.gguf_dir_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(gguf_frame, text="Browse", command=self._browse_gguf_dir).grid(row=0, column=2, padx=5, pady=5)
        self.auto_open_webui_var_dialog = ctk.BooleanVar(value=self.config_to_update.get("auto_open_webui", True))
        ctk.CTkCheckBox(self, text="Auto-Open Web UI After Successful Launch", variable=self.auto_open_webui_var_dialog).pack(pady=10)
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=20)
        ctk.CTkButton(button_frame, text="Save and Continue", command=self._on_save).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Cancel Setup", command=self._on_cancel, fg_color="gray").pack(side="left", padx=10)

    def _browse_exe(self):
        fp = filedialog.askopenfilename(title="Select KCPP Exe/Script", filetypes=[("Executables", "*.exe" if sys.platform=="win32" else "*"),("Py Scripts","*.py"),("All","*.*")])
        if fp:
            self.exe_entry.delete(0,"end")
            self.exe_entry.insert(0,fp)

    def _browse_gguf_dir(self):
        dp = filedialog.askdirectory(title="Select Default GGUF Dir")
        if dp:
            self.gguf_dir_entry.delete(0,"end")
            self.gguf_dir_entry.insert(0,dp)

    def _on_save(self):
        exe_p = self.exe_entry.get().strip()
        gguf_d = self.gguf_dir_entry.get().strip()
        valid_exe = False
        if exe_p and (os.path.exists(exe_p) or shutil.which(exe_p)):
            valid_exe = True
        elif exe_p and exe_p.lower().endswith(".py"):
            try:
                launcher_d = os.path.dirname(os.path.abspath(__file__)) if not getattr(sys,'frozen',False) else os.path.dirname(sys.executable)
                if os.path.exists(os.path.join(launcher_d, exe_p)): # Corrected indentation for this line
                    valid_exe = True
            except NameError:
                pass
        if not valid_exe:
            messagebox.showerror("Invalid Path","KCPP exe path invalid.",parent=self)
            return
        if gguf_d and not os.path.isdir(gguf_d):
            messagebox.showwarning("Invalid Dir","Default GGUF dir invalid.",parent=self)
            gguf_d=""
        
        self.config_to_update["koboldcpp_executable"] = os.path.abspath(exe_p) if os.path.exists(exe_p) else (shutil.which(exe_p) or exe_p)
        self.config_to_update["default_gguf_dir"] = gguf_d
        self.config_to_update["auto_open_webui"] = self.auto_open_webui_var_dialog.get()
        self.config_to_update["first_run_completed"] = True
        self.config_to_update["first_run_intro_shown"] = True
        self.saved_config = self.config_to_update
        self.destroy()

    def _on_cancel(self):
        self.saved_config = None
        self.destroy()

if __name__ == "__main__":
    app = KoboldLauncherGUI()
    def on_closing_gui():
        kcpp_mon_running = hasattr(app,'kcpp_process_obj') and app.kcpp_process_obj and app.kcpp_process_obj.poll() is None
        last_direct_running = app.last_process and app.last_process.poll() is None
        kcpp_running = app.process_running or kcpp_mon_running or last_direct_running
        if kcpp_running:
            if messagebox.askyesno("Exit", "KCPP process might be running or tuning active.\nStop all & exit?"):
                app.stop_all_kcpp_processes_forcefully()
                time.sleep(0.5)
                app.destroy()
        else:
            app.destroy()
    app.protocol("WM_DELETE_WINDOW", on_closing_gui)
    app.mainloop()
