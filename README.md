# TensorTune v1.0.0

TensorTune is a user-friendly toolkit for efficiently running KoboldCpp, intelligently managing VRAM usage through optimized tensor offload strategies to maximize performance on consumer GPUs. Available in both GUI and CLI versions.

![TensorTune GUI](https://raw.githubusercontent.com/Viceman256/TensorTune/main/screenshots/GUI1.png) 

## 🚀 Performance Examples with TensorTune

Here are some real-world examples of performance improvements using tensor offloading:

### Example 1: QwQ Merge on 12GB VRAM GPU

**Traditional Layer Offloading:**
```bash
python koboldcpp.py --threads 6 --usecublas --contextsize 40960 --flashattention --port 5000 --model MODELNAME.gguf --gpulayers 59 --quantkv 1


Tokens per second: 3.95 t/s

With TensorTune's Smart Tensor Offloading:

      
python koboldcpp.py --threads 10 --usecublas --contextsize 40960 --flashattention --port 5000 --model MODELNAME.gguf --gpulayers 65 --quantkv 1 --overridetensors "\.[13579]\.ffn_up|\.[1-3][13579]\.ffn_up=CPU"


Tokens per second: 10.61 t/s

Result: 168% speed improvement while using the same amount of VRAM!
Example 2: Qwen3-30B A3B on RTX 4060 Ti (16GB)

Traditional Layer Offloading:

      
# Offloading 30 layers
python koboldcpp.py --model Qwen3-30B-A3B-Q4_K_M.gguf --usecublas --gpulayers 30


Tokens per second: 10 t/s

With TensorTune's Smart Tensor Offloading:

      
# All layers on GPU with tensor offloading
python koboldcpp.py --model Qwen3-30B-A3B-Q4_K_M.gguf --usecublas --gpulayers 99 --overridetensors "blk\.([0-9]*[02468])\.ffn_.*_exps\.=CPU"


Tokens per second: 15 t/s

Result: 50% speed improvement with better model quality!
Example 3: Gemma3-27B on 16GB VRAM GPU

Traditional Layer Offloading:

      
# Baseline with 46 layers offloaded
python koboldcpp.py --model gemma3-27b-IQ4_XS.gguf --contextsize 16384 --flashattention --gpulayers 46


Tokens per second: 6.86 t/s

With TensorTune's Smart Tensor Offloading:

      
# All layers on GPU with selective tensor offloading
python koboldcpp.py --model gemma3-27b-IQ4_XS.gguf --contextsize 16384 --flashattention --gpulayers 99 --overridetensors "\.(5[3-9]|6[0-3])\.(ffn_*)=CPU"



Tokens per second: 10.4 t/s

Result: 52% speed improvement with the same VRAM usage!
Example 4: Qwen3-235B MoE on Dual Xeon with 60GB VRAM

Traditional Layer Offloading:

      
# Standard approach
python koboldcpp.py --model Qwen3-235B-IQ4_XS.gguf --contextsize 32768 --flashattention --gpulayers 95



Tokens per second: 2.9 t/s

With TensorTune's Smart Tensor Offloading:

      
# Using tensor-specific offloading
python koboldcpp.py --model Qwen3-235B-IQ4_XS.gguf --contextsize 32768 --flashattention --gpulayers 95 --overridetensors "([4-9]+).ffn_.*_exps.=CPU"

    

Tokens per second: 4.2 t/s

Result: 45% speed improvement while maintaining all 95 layers on GPU!
📋 Features

    Intelligent Auto-Tuning: Automatically finds optimal tensor offload strategies.

    Multiple Launch Options: Direct launch, best remembered config, or full auto-tuning.

    VRAM Monitoring: Real-time display of GPU VRAM usage, with support for vendor-specific selection and manual override.

    Configuration Management: Global, model-specific, and session-based settings. Configurable data/config paths.

    Database-Backed History: Remembers what worked best for each model.

    Process Management: Easily start and stop KoboldCpp instances.

    Both CLI and GUI: Choose the interface that suits your preference.

    Dynamically Adjusts GPU Layers: Coordinates GPU layer counts with tensor offload levels for maximum performance.

    Export/Import Settings: Backup and share your TensorTune configurations.

🔧 Installation of TensorTune
Quick Start

    Ensure you have Python 3.7+ installed.

    Download or clone this repository:

          
    git clone https://github.com/Viceman256/TensorTune.git
    cd TensorTune

        

Run the installation script:

      
python tensortune_install.py

    

IGNORE_WHEN_COPYING_START

    Use code with caution. Bash
    IGNORE_WHEN_COPYING_END

    The script will:

        Check for required dependencies.

        Install necessary Python packages from requirements.txt.

        Prompt for your KoboldCpp installation path if not found.

        Create convenient launch scripts (launch_cli.sh/.py, launch_gui.sh/.py).

Manual Installation

If you prefer to install manually:

    Install required dependencies:

          
    pip install -r requirements.txt

        

    IGNORE_WHEN_COPYING_START

Use code with caution. Bash
IGNORE_WHEN_COPYING_END

Ensure all core files are in the same directory:

    tensortune_core.py

    tensortune_cli.py

    tensortune_gui.py

    tensortune_examples.py (Optional, for viewing examples)

    tensortune_install.py (Optional, for easier setup)

    requirements.txt

Run TensorTune:

      
# For GUI
python tensortune_gui.py

# For CLI
python tensortune_cli.py

    

IGNORE_WHEN_COPYING_START

    Use code with caution. Bash
    IGNORE_WHEN_COPYING_END

📚 Using TensorTune

TensorTune automatically creates and manages its configuration files and history database in your user-specific application data/configuration directory (e.g., ~/.config/TensorTune on Linux, AppData/Roaming/TensorTune on Windows).
GUI Version

The application has three main tabs:
Main Tab: Tune & Launch

    Select your GGUF model.

    Monitor VRAM usage (reflects selected GPU and override settings).

    Choose a launch method:

        Auto-Tune / Use OT Strategy (recommended).

        Launch Best Remembered Config.

        Direct Launch with Settings Defaults.

    Stop any running KoboldCpp processes.

Auto-Tuning Session Mode

    View and adjust the current tensor offload strategy (OT Level).

    Launch and monitor KoboldCpp for testing. Output is displayed in the GUI.

    Make adjustments based on results (more GPU/CPU offload).

    Save successful configurations to the history database.

    Edit session or permanent base arguments for the model.

Settings Tab

    Set KoboldCpp executable path.

    Configure global default arguments for KoboldCpp.

    GPU Management: Select target GPU type and ID for VRAM monitoring.

    VRAM Override: Manually set the total VRAM budget for launcher calculations.

    Choose UI theme (Dark/Light/System).

    Manage Model-Specific argument overrides.

    Export/Import TensorTune settings.

    Reset all settings to default.

History Tab

    View past launch attempts.

    See which configurations worked best, filtered by model or globally.

CLI Version

The CLI offers an interactive text-based interface with similar functionality:

    Select a GGUF model.

    Choose between auto-tuning or direct launch.

    Follow the prompts to adjust settings and launch KoboldCpp.

    View launch history and performance statistics.

    Access "Launcher Settings" to configure paths, GPU targets, VRAM overrides, and default arguments.

📝 Understanding Tensor Offloading

Traditional layer offloading moves entire transformer layers between CPU and GPU. Tensor offloading is more granular - it selectively keeps specific tensors (like FFN up/down/gate) on CPU while keeping smaller, computation-intensive tensors (like attention) on GPU.

TensorTune helps you find the optimal balance by:

    Analyzing your model's characteristics (size, layers, MoE).

    Testing different offloading patterns (OT Levels).

    Monitoring VRAM usage and load success/failure.

    Remembering what works best for each model based on your hardware and available VRAM.

🔍 Troubleshooting

    KoboldCpp Not Found: Set the correct path in the Settings tab (GUI) or Launcher Settings (CLI).

    Python Errors: Ensure you've installed all dependencies from requirements.txt.

    VRAM Info Not Displaying: Install pynvml for NVIDIA GPU monitoring, ensure AMD/Intel drivers and necessary libraries (pyadlx, rocm-smi, pyze) are correctly set up if applicable. Check GPU selection settings.

    Auto-Tuning Fails: Try reducing context size or other memory-intensive settings. Manually adjust the OT Level towards more CPU offload.

🙏 Acknowledgments

This project was inspired by the tensor offloading technique shared on Reddit. TensorTune aims to make this optimization method accessible and manageable for more users.
📜 License

This project is licensed under the MIT License - see the LICENSE.md file for details.
📸 Screenshots


CLI Example 1

![alt text](https://raw.githubusercontent.com/Viceman256/TensorTune/main/screenshots/cli1.png)

CLI Example 2

![alt text](https://raw.githubusercontent.com/Viceman256/TensorTune/main/screenshots/cli2.png)

GUI Example 1 (Main Tab)

![alt text](https://raw.githubusercontent.com/Viceman256/TensorTune/main/screenshots/GUI1.png)

GUI Example 2 (Tuning Session)

![alt text](https://raw.githubusercontent.com/Viceman256/TensorTune/main/screenshots/GUI2.png)

GUI Example 3 (Settings Tab)

![alt text](https://raw.githubusercontent.com/Viceman256/TensorTune/main/screenshots/GUI3.png)

GUI Example 4 (History Tab)

![alt text](https://raw.githubusercontent.com/Viceman256/TensorTune/main/screenshots/GUI4.png)

Latest Release: TensorTune v1.0.0

Key changes in this version include:
* Rebranding to TensorTune.
* Improved configuration management and GPU/VRAM handling.
* Enhanced auto-tuning heuristics.
* GUI and CLI updates for better user experience.

For a full list of changes, see the [v1.0.0 Release Notes](https://github.com/Viceman256/TensorTune/releases/tag/v1.0.0) or the [CHANGELOG.md](CHANGELOG.md).

🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
