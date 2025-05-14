# KoboldCpp-Smart-Launcher-v1.0.0
KoboldCpp Smart Launcher with GPU Layer and Tensor Override Tuning

A user-friendly toolkit for efficiently running KoboldCpp with optimized tensor offload strategies. Available in both GUI and CLI versions, this launcher intelligently manages VRAM usage to maximize performance on consumer GPUs.

![KoboldCpp Smart Launcher GUI](https://github.com/Viceman256/KoboldCpp-Smart-Launcher-v1.0.0/blob/main/screenshots/GUI1.png)

## 🚀 Performance Examples

Here are some real-world examples of performance improvements using tensor offloading:

### Example 1: Qwen3-32B on RTX 3060 (12GB)

**Traditional Layer Offloading:**
```bash
python koboldcpp.py --model Qwen3-32B-Q4_K_M.gguf --threads 8 --usecublas --flashattention --gpulayers 38 --quantkv 1
```
`Tokens per second: 3.98 t/s`

**With Smart Tensor Offloading:**
```bash
python koboldcpp.py --model Qwen3-32B-Q4_K_M.gguf --threads 8 --usecublas --flashattention --gpulayers 65 --quantkv 1 --overridetensors "\.([0-9]+)\.ffn_(up|gate)=CPU"
```
`Tokens per second: 10.61 t/s`

*Result: 166% speed improvement while maintaining VRAM usage!*

### Example 2: Gemma3-27B on RTX 2060 (6GB)

**Traditional Layer Offloading:**
```bash
python koboldcpp.py --model gemma3-27b-IQ4_XS.gguf --threads 6 --usecublas --flashattention --gpulayers 16 --contextsize 16384
```
`Tokens per second: 6.86 t/s`

**With Smart Tensor Offloading:**
```bash
python koboldcpp.py --model gemma3-27b-IQ4_XS.gguf --threads 6 --usecublas --flashattention --gpulayers 99 --contextsize 16384 --overridetensors "\.(5[3-9]|6[0-3])\.(ffn_*)=CPU"
```
`Tokens per second: 10.4 t/s`

*Result: 52% speed improvement with the same VRAM usage!*

### Example 3: QwQ Merge on 12GB VRAM GPU

**Traditional Layer Offloading:**
```bash
python koboldcpp.py --threads 6 --usecublas --contextsize 40960 --flashattention --model QwQ_Merge.gguf --gpulayers 59 --quantkv 1
```
`Tokens per second: 3.95 t/s`

**With Smart Tensor Offloading:**
```bash
python koboldcpp.py --threads 10 --usecublas --contextsize 40960 --flashattention --model QwQ_Merge.gguf --gpulayers 65 --quantkv 1 --overridetensors "\.[13579]\.ffn_up|\.[1-3][13579]\.ffn_up=CPU"
```
`Tokens per second: 10.61 t/s`

*Result: Over 160% speed improvement while maintaining VRAM usage!*

### Example 4: Qwen3-235B MoE on 48GB VRAM GPU

**With Smart Tensor Offloading:**
```bash
python koboldcpp.py --model Qwen3-235B-IQ3_M.gguf --gpulayers 99 --contextsize 32768 --flashattention --usecublas --overridetensors "([4-9]+).ffn_.*_exps.=CPU"
```
`Tokens per second: 7.6 t/s`

*Result: Enables running an otherwise impossible model size on a single GPU!*

## 📋 Features

- **Intelligent Auto-Tuning:** Automatically finds optimal tensor offload strategies
- **Multiple Launch Options:** Direct launch, best remembered config, or full auto-tuning
- **VRAM Monitoring:** Real-time display of GPU VRAM usage
- **Configuration Management:** Global, model-specific, and session-based settings
- **Database-Backed History:** Remembers what worked best for each model
- **Process Management:** Easily start and stop KoboldCpp instances
- **Both CLI and GUI:** Choose the interface that suits your preference
- **Intelligent Auto-Tuning:**
  - Automatically finds optimal tensor offload strategies
  - Dynamically adjusts GPU layer counts based on offload level
  - Coordinates both strategies for maximum performance

## 🔧 Installation

### Quick Start

1. Ensure you have Python 3.7+ installed
2. Download this repository
3. Run the installation script:

```bash
python install.py
```

The script will:
- Check for required dependencies
- Install necessary Python packages
- Look for your KoboldCpp installation
- Create convenient launch scripts

### Manual Installation

If you prefer to install manually:

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Place all three core files in the same directory:
   - `koboldcpp_core.py`
   - `koboldcpp_launcher.py`
   - `koboldcpp_launcher_gui.py`

3. Run the launcher:
```bash
# For GUI
python koboldcpp_launcher_gui.py

# For CLI
python koboldcpp_launcher.py
```

## 📚 Using the Launcher

### GUI Version

The application has three main tabs:

#### Main Tab: Tune & Launch
- Select your GGUF model
- Monitor VRAM usage
- Choose a launch method:
  - Auto-Tune / Use OT Strategy (recommended)
  - Launch Best Remembered Config
  - Direct Launch with Settings Defaults
- Stop any running KoboldCpp processes

#### Auto-Tuning Session Mode
- View and adjust the current tensor offload strategy
- Launch and monitor KoboldCpp for testing
- Make adjustments based on results (more GPU/CPU)
- Save successful configurations
- Edit session or permanent arguments

#### Settings Tab
- Set KoboldCpp executable path
- Configure global default arguments
- Choose UI theme

#### History Tab
- View past launch attempts
- See which configurations worked best

### CLI Version

The CLI offers an interactive text-based interface with similar functionality:

1. Select a GGUF model
2. Choose between auto-tuning or direct launch
3. Follow the prompts to adjust settings and launch KoboldCpp
4. View launch history and performance statistics

## 📝 Understanding Tensor Offloading

Traditional layer offloading moves entire transformer layers between CPU and GPU. Tensor offloading is more granular - it selectively keeps specific tensors (like FFN up/down/gate) on CPU while keeping smaller, computation-intensive tensors on GPU.

The launcher helps you find the optimal balance by:
1. Analyzing your model
2. Testing different offloading patterns
3. Monitoring VRAM usage and load success
4. Remembering what works best for each model

## 🔍 Troubleshooting

- **KoboldCpp Not Found**: Set the correct path in the Settings tab
- **Python Errors**: Ensure you've installed all dependencies
- **VRAM Info Not Displaying**: Install pynvml for NVIDIA GPU monitoring
- **Auto-Tuning Fails**: Try reducing context size or other memory-intensive settings

## 🙏 Acknowledgments

This project was inspired by the [tensor offloading technique](https://www.reddit.com/r/LocalLLaMA/comments/1ki3sze/running_qwen3_235b_on_a_single_3060_12gb_6_ts/) shared on Reddit. It aims to make this optimization method accessible to more users.

## 📜 License

This project is licensed under the MIT License - see the LICENSE.md file for details.

---

## 📸 Screenshots

CLI Example 1
![CLI Example 1](https://github.com/Viceman256/KoboldCpp-Smart-Launcher-v1.0.0/blob/main/screenshots/cli1.png)

CLI Example 2
![CLI Example 2](https://github.com/Viceman256/KoboldCpp-Smart-Launcher-v1.0.0/blob/main/screenshots/cli2.png)

GUI Example 1
![GUI Example 1](https://github.com/Viceman256/KoboldCpp-Smart-Launcher-v1.0.0/blob/main/screenshots/GUI1.png)

GUI Example 2
![GUI Example 2](https://github.com/Viceman256/KoboldCpp-Smart-Launcher-v1.0.0/blob/main/screenshots/GUI2.png)

GUI Example 3
![GUI Example 3](https://github.com/Viceman256/KoboldCpp-Smart-Launcher-v1.0.0/blob/main/screenshots/GUI3.png)

GUI Example 4
![GUI Example 4](https://github.com/Viceman256/KoboldCpp-Smart-Launcher-v1.0.0/blob/main/screenshots/GUI4.png)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
