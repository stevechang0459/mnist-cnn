# PyTorch MNIST CNN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch-based Convolutional Neural Network (CNN) project for MNIST digit classification, featuring a fully interactive PyQt5 graphical interface. This project is specifically optimized for Edge AI environments, supporting **Native Intel Arc GPU (XPU)** acceleration for training and **OpenVINO IR / ONNX** export for high-performance inference.

## ✨ Key Features

* **Interactive PyQt5 GUI:** Dual-tab interface for seamless switching between Model Training and Real-time Inference.
* **Intel XPU Acceleration:** Native support for Intel discrete GPUs (e.g., Arc series) during the training loop without needing external plugins.
* **Edge Deployment Ready:** One-click model export to ONNX (`.onnx`) and OpenVINO Intermediate Representation (`.xml` / `.bin`).
* **Advanced Inference Canvas:** Features a drawing canvas with Auto Bounding Box Alignment and real-time Softmax probability distribution visualizer.
* **Customizable Training:** Toggle data augmentation (Random Affine) and adjust hyperparameters (Epochs, Learning Rate) directly from the UI.

## 🚀 Environment Setup (Recommended)

To avoid DLL conflicts and ensure perfect compatibility with hardware accelerators (Intel XPU / NVIDIA GPU) and OpenVINO, please strictly follow the installation steps below.

### Option 1: Anaconda (Windows)

Step 1: Prepare a clean environment

```shell
conda activate base
conda env remove -n mnist-cnn -y
conda create -n mnist-cnn python=3.11 -y
conda activate mnist-cnn
```

Step 2: Choose your hardware backend (Pick ONLY ONE)

```shell
# Option A: Install PyTorch with NVIDIA CUDA 12.1 backend (For NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

OR

```shell
# Option B: Install PyTorch with specific Intel XPU backend (For Intel GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

Step 3: Install Edge AI runtimes and GUI dependencies

```shell
pip install onnx onnxruntime onnxscript
pip install openvino
pip install PyQt5 matplotlib
```

### Option 2: Python Virtual Environment (Ubuntu 24.04)

Step 1: Prepare a clean environment

```shell
sudo apt install python3.12-venv python-is-python3 -y
python -m venv .venv
source .venv/bin/activate
```

Step 2: Choose your hardware backend (Pick ONLY ONE)

```shell
# Option A: Install PyTorch with NVIDIA CUDA 12.1 backend (For NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

OR

```shell
# Option B: Install PyTorch with specific Intel XPU backend (For Intel GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

Step 3: Install Edge AI runtimes and GUI dependencies

```shell
pip install onnx onnxruntime onnxscript
pip install openvino
pip install PyQt5 matplotlib
```

> [!Note]
> Alternatively, install via requirements file if provided in the repository:
>
> ```shell
> pip install -r examples/requirements.txt
> ```

## 🎮 How to Run

Once the environment setup is complete, you can launch the interactive GUI application directly from your terminal.

```shell
# Step 1: Ensure your environment is activated
# For Conda (Windows):
conda activate mnist-cnn

# For venv (Ubuntu 24.04):
source .venv/bin/activate

# Step 2: Launch the main GUI application
python gui_app.py
```

## 🖥️ Application Screenshots

### Model Training & Export

![training tab](https://hackmd.io/_uploads/r18dBRsMMx.png)

### Real-Time Inference & Canvas

![inference tab](https://hackmd.io/_uploads/BJdNLAsMfg.png)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
