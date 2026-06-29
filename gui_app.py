# gui_app.py
import os
import sys
import random
import numpy as np
import copy
import subprocess
import json

import torch
import torch.nn as nn
import torch.onnx

from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget,
                             QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QTextEdit, QProgressBar, QGroupBox, QFormLayout,
                             QLineEdit, QMessageBox, QCheckBox, QComboBox, QSizePolicy)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QPoint, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QColor

from model import SimpleCNN
from dataset import get_mnist_loaders

# Dynamically attempt to load OpenVINO for IR conversion and inference support
try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

# Dynamically attempt to load ONNX Runtime for multi-backend inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import dx_engine
    DEEPX_AVAILABLE = True
except ImportError:
    DEEPX_AVAILABLE = False

# Define the local path for saving and loading PyTorch weights
WEIGHT_PATH = "models/mnist_cnn_weights.pth"

class TrainingThread(QThread):
    """
    Worker thread to handle PyTorch model training asynchronously.
    This prevents the intensive training loop from freezing the main PyQt GUI event loop.
    """
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool)

    def __init__(self, model, device, train_loader, epochs, lr, weight_path, export_onnx=False, export_ir=False):
        super().__init__()
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.epochs = epochs
        self.lr = lr
        self.weight_path = weight_path
        self.export_onnx = export_onnx
        self.export_ir = export_ir

    def run(self):
        try:
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

            self.log_signal.emit("Starting background training thread...")
            for epoch in range(self.epochs):
                self.model.train()
                running_loss = 0.0

                for images, labels in self.train_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                avg_loss = running_loss / len(self.train_loader)
                self.log_signal.emit(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

                # Calculate and emit training progress percentage to update the GUI
                progress = int(((epoch + 1) / self.epochs) * 100)
                self.progress_signal.emit(progress)

            self.log_signal.emit(f"Saving trained weights to {self.weight_path}...")
            # Save the state_dict locally
            os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.weight_path)

            # --- Post-Training Export Sequence ---
            if self.export_onnx or self.export_ir:
                self.model.eval()
                # Generate a dummy tensor required to trace the model's computational graph (Batch=1, Channel=1, 28x28)
                dummy_input = torch.randn(1, 1, 28, 28).to(self.device)

                if self.export_onnx:
                    onnx_path = self.weight_path.replace('.pth', '.onnx')
                    self.log_signal.emit(f"Compiling computational graph to ONNX format: {onnx_path}...")
                    torch.onnx.export(self.model, dummy_input, onnx_path,
                                      input_names=['input'], output_names=['output'],
                                      # Allow dynamic batch sizing for flexible downstream inference pipelines
                                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
                    self.log_signal.emit("ONNX compilation successful.")

                    if DEEPX_AVAILABLE:
                        dxnn_dir = os.path.dirname(self.weight_path)
                        json_path = self.weight_path.replace('.pth', '.json')
                        calibration_dir = os.path.abspath(os.path.join(dxnn_dir, "calibration_dataset"))

                        # Preprocessing pipeline configuration specifically for the DEEPX dxcom compiler
                        dxnn_config = {
                            "model_name": "mnist_cnn_weights",
                            "model_type": "ONNX",
                            "inputs": {
                                "input": [1, 1, 28, 28]     # Note: DEEPX strictly requires the batch size to be fixed to 1
                            },
                            "calibration_num": 100,
                            "calibration_method": "ema",    # Recommended to use the EMA algorithm to improve quantization accuracy
                            "default_loader": {
                                "dataset_path": calibration_dir,
                                "file_extensions": ["png", "jpg", "jpeg"],
                                "preprocessings": [
                                    # OpenCV reads images in [28, 28, 3] BGR format by default
                                    { "convertColor": { "form": "BGR2GRAY" } }, # Convert to [28, 28] grayscale (Note: the channel dimension is squeezed)
                                    {"resize": {"width": 28, "height": 28}},
                                    { "div": { "x": 255.0 } },
                                    { "normalize": { "mean": [0.5], "std": [0.5] } },
                                    # CRITICAL: Must execute expandDim twice consecutively to restore [28, 28] back to the ONNX-expected [1, 1, 28, 28]
                                    { "expandDim": { "axis": 0 } },  # First layer: Restore the Channel dimension -> [1, 28, 28]
                                    { "expandDim": { "axis": 0 } }   # Second layer: Restore the Batch dimension -> [1, 1, 28, 28]
                                ]
                            }
                        }

                        with open(json_path, 'w') as f:
                            json.dump(dxnn_config, f, indent=4)

                        self.log_signal.emit(f"Generated dxcom compilation config: {json_path}")

                        try:
                            compile_cmd = [
                                "dxcom",
                                "-m", onnx_path,
                                "-c", json_path,
                                "-o", dxnn_dir,
                                "--gen_log"
                            ]
                            cmd_str = " ".join(compile_cmd)
                            self.log_signal.emit(f"Running command: {cmd_str}")
                            result = subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
                            self.log_signal.emit("dxcom compilation successful! INT8 Model ready for NPU.")

                        except subprocess.CalledProcessError as e:
                            self.log_signal.emit(f"Warning: dxcom compilation failed.\nError: {e.stderr}")
                        except FileNotFoundError:
                            self.log_signal.emit("Warning: 'dxcom' command not found. Please install the DX-Compiler environment.")

                if self.export_ir and OPENVINO_AVAILABLE:
                    ir_path = self.weight_path.replace('.pth', '.xml')
                    self.log_signal.emit(f"Translating model to OpenVINO Intermediate Representation (IR): {ir_path}...")
                    # OpenVINO can directly convert PyTorch objects from memory
                    ov_model = ov.convert_model(self.model, example_input=dummy_input)

                    # Strictly lock the input shape to (1, 1, 28, 28) to satisfy NPU/GPU compiler requirements
                    ov_model.reshape([1, 1, 28, 28])
                    ov.save_model(ov_model, ir_path)
                    self.log_signal.emit("OpenVINO IR translation successful.")

            self.finished_signal.emit(True)
        except Exception as e:
            self.log_signal.emit(f"Critical error during training sequence: {str(e)}")
            self.finished_signal.emit(False)


class GridLabel(QLabel):
    """
    Custom QLabel that dynamically scales a 28x28 image to fit the layout
    while maintaining aspect ratio and overlaying an accurate pixel grid.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setAlignment(Qt.AlignCenter)

        # Enforce a 1:1 aspect ratio layout policy to prevent rectangular distortion.
        # This ensures the widget remains a perfect square during fluid layout calculations.
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHeightForWidth(True)
        self.setSizePolicy(sizePolicy)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return width

    def paintEvent(self, event):
        if not self.pixmap():
            super().paintEvent(event)
            return

        painter = QPainter(self)
        # Dynamically scale the underlying 28x28 image to fit the current widget size
        scaled_pix = self.pixmap().scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation)

        # Calculate offsets to perfectly center the image inside the label
        x = (self.width() - scaled_pix.width()) // 2
        y = (self.height() - scaled_pix.height()) // 2
        painter.drawPixmap(x, y, scaled_pix)

        # Overlay the grid perfectly matched to the scaled image
        w = scaled_pix.width()
        h = scaled_pix.height()
        step_x = w / 28.0
        step_y = h / 28.0

        # Use a semi-transparent dark gray pen to avoid overwhelming the image features
        pen = QPen(QColor(100, 100, 100, 120), 1, Qt.SolidLine)
        painter.setPen(pen)

        # Draw the horizontal and vertical grid lines
        for i in range(1, 28):
            painter.drawLine(int(x + i * step_x), y, int(x + i * step_x), y + h)
            painter.drawLine(x, int(y + i * step_y), x + w, int(y + i * step_y))


class DrawingCanvas(QWidget):
    """
    Fluid responsive drawing canvas. Uses a hidden memory buffer to store ink,
    and dynamically extracts the active visible area for inference.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)

        # Enforce a 1:1 aspect ratio to perfectly align with the GridLabel dimension
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHeightForWidth(True)
        self.setSizePolicy(sizePolicy)

        # Allocate a reasonable 600x600 buffer.
        # Since the main window is locked to 800x480, the canvas width will not exceed 256px.
        # A 600px buffer safely accommodates up to 200%+ OS-level High DPI vector scaling
        # without wasting unnecessary RAM compared to the legacy 2000x2000 allocation.
        self.pixmap = QPixmap(600, 600)
        self.pixmap.fill(Qt.black)

        self.last_point = QPoint()
        self.current_bbox = None
        self.needs_clear = False

    def hasHeightForWidth(self):
        """
        Informs the PyQt layout manager that this widget's height is strictly
        dependent on its dynamically allocated width.
        """
        return True

    def heightForWidth(self, width):
        """
        Locks the height to exactly match the width, maintaining a 1:1 square ratio.
        """
        return width

    def clear_canvas(self):
        self.pixmap.fill(Qt.black)
        self.current_bbox = None
        self.needs_clear = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Auto-clear the canvas if a previous inference cycle just finished
            if self.needs_clear:
                self.clear_canvas()

            self.last_point = event.pos()
            self.current_bbox = None  # Hide the old bounding box when drawing starts
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            painter = QPainter(self.pixmap)

            # Dynamically adjust pen width relative to the actual rendered canvas size
            # This ensures the stroke thickness feels consistent regardless of High DPI scaling
            pen_width = max(10, int(self.width() * 0.05))
            pen = QPen(Qt.white, pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        # Only render the active viewport from the pre-allocated buffer
        painter.drawPixmap(0, 0, self.pixmap, 0, 0, self.width(), self.height())

        # Overlay a hollow red rectangle if bounding box coordinates exist
        if self.current_bbox:
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            left, upper, right, lower = self.current_bbox
            painter.drawRect(left, upper, right - left, lower - upper)

    def get_tensor(self, use_bbox=True):
        """
        Processes the canvas drawing into a normalized PyTorch tensor.
        Returns a tuple: (normalized_tensor, final_pil_image, bbox_coordinates)
        """
        bbox = None
        # Dynamically capture current viewport dimensions
        w, h = self.width(), self.height()

        if use_bbox:
            # --- [Optimized Approach: Bounding Box Extraction & Centering] ---
            # Extract the raw image buffer and convert it to a NumPy array
            qimage = self.pixmap.copy(0, 0, w, h).toImage().convertToFormat(QImage.Format_Grayscale8)
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            img_data = np.array(ptr).reshape((h, w))
            pil_img = Image.fromarray(img_data)

            bbox = pil_img.getbbox()
            if bbox:
                # Crop the exact digit, scale it down, and paste it into the center of a 28x28 canvas
                cropped = pil_img.crop(bbox)
                cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)
                final_img = Image.new('L', (28, 28), 0)
                offset_x = (28 - cropped.width) // 2
                offset_y = (28 - cropped.height) // 2
                final_img.paste(cropped, (offset_x, offset_y))
            else:
                final_img = Image.new('L', (28, 28), 0)

            # Normalize to [-1.0, 1.0] standard
            final_array = np.array(final_img, dtype=np.float32) / 255.0
            final_array = (final_array - 0.5) / 0.5
            tensor = torch.tensor(final_array).unsqueeze(0).unsqueeze(0)

        else:
            # --- [Baseline Approach: Naive Direct Scaling] ---
            # Directly squash the 280x280 canvas down to 28x28 without alignment
            scaled_pixmap = self.pixmap.copy(0, 0, w, h).scaled(28, 28, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            qimage = scaled_pixmap.toImage()

            img_data = np.zeros((28, 28), dtype=np.float32)
            for y in range(28):
                for x in range(28):
                    color = qimage.pixelColor(x, y)
                    img_data[y, x] = color.red() / 255.0

            final_img = Image.fromarray((img_data * 255).astype(np.uint8))
            img_data = (img_data - 0.5) / 0.5
            tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)

        return tensor, final_img, bbox


class MNISTGuiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyTorch - MNIST CNN")
        self.setFixedSize(800, 480)
        # self.setMinimumSize(800, 480)
        # self.resize(800, 480)

        # By default, keep the GUI and general operations on the CPU
        # The XPU is selectively targeted during the training loop for safety
        self.device = torch.device("cpu")
        self.model = SimpleCNN().to(self.device)

        # Initialize loaders and fetch dataset handles (test dataset is needed immediately for sampling)
        _, _, self.test_dataset = get_mnist_loaders(batch_size=64)
        self.train_loader = None  # Instantiated just-in-time during start_training()

        self.prob_bars = []

        # Initialize the timer for the Auto Test feature
        self.auto_timer = QTimer(self)
        self.auto_timer.timeout.connect(self.run_random_inference)

        # Verify and load pre-existing network parameters
        self.load_weights_if_exists()
        self.init_ui()

    def load_weights_if_exists(self):
        """Attempts to load the PyTorch state dictionary safely."""
        if os.path.exists(WEIGHT_PATH):
            try:
                self.model.load_state_dict(torch.load(WEIGHT_PATH, map_location=self.device, weights_only=True))
                self.weights_loaded = True
            except Exception:
                self.weights_loaded = False
        else:
            self.weights_loaded = False

    def reload_inference_engine(self):
        """
        Dynamically swaps the underlying inference backend (PyTorch, ONNX, OpenVINO)
        based on the user's combo box selection.
        """
        selection = self.engine_combo.currentText()
        self.active_engine = "pytorch"  # Default fallback

        if "ONNX" in selection:
            onnx_path = WEIGHT_PATH.replace('.pth', '.onnx')
            if os.path.exists(onnx_path):
                try:
                    self.ort_session = ort.InferenceSession(onnx_path)
                    self.active_engine = "onnx"
                    self.engine_status.setText("Status: ONNX Runtime Active")
                    self.weights_loaded = True
                except Exception as e:
                    self.engine_status.setText("Status: ONNX Error")
                    QMessageBox.warning(self, "ONNX Error", str(e))
            else:
                self.engine_status.setText("Status: Missing .onnx file")
                QMessageBox.warning(self, "Missing File", "Please export ONNX from the Training tab first.")
                self.engine_combo.setCurrentIndex(0)

        elif "OpenVINO" in selection:
            ir_path = WEIGHT_PATH.replace('.pth', '.xml')
            if os.path.exists(ir_path):
                try:
                    # Extract the specific hardware target (e.g., CPU, GPU, NPU) from the dropdown string
                    target_device = selection.split(" - ")[-1]

                    core = ov.Core()
                    compiled_model = core.compile_model(ir_path, device_name=target_device)

                    # Create standard OpenVINO structures for rapid inference execution
                    self.ov_infer_request = compiled_model.create_infer_request()
                    self.ov_input_layer = compiled_model.input(0)
                    self.ov_output_layer = compiled_model.output(0)

                    self.active_engine = "openvino"
                    self.engine_status.setText(f"Status: OpenVINO Active ({target_device})")
                    self.weights_loaded = True
                except Exception as e:
                    self.engine_status.setText("Status: OpenVINO Error")
                    QMessageBox.warning(self, "OpenVINO Error", str(e))
            else:
                self.engine_status.setText("Status: Missing .xml file")
                QMessageBox.warning(self, "Missing File", "Please export OpenVINO IR from the Training tab first.")
                self.engine_combo.setCurrentIndex(0)

        elif "DEEPX" in selection:
            dxnn_path = WEIGHT_PATH.replace('.pth', '.dxnn')
            if os.path.exists(dxnn_path):
                try:
                    self.dx_engine = dx_engine.InferenceEngine(dxnn_path)
                    self.active_engine = "deepx"
                    self.engine_status.setText("Status: DX-Runtime Active")
                    self.weights_loaded = True
                except Exception as e:
                    self.engine_status.setText("Status: DX-Runtime Error")
                    QMessageBox.warning(self, "DX-Runtime Error", str(e))

            else:
                self.engine_status.setText("Status: Missing .dxnn file")
                QMessageBox.warning(self, "Missing File", "Please compile DXNN from the Training tab first.")
                self.engine_combo.setCurrentIndex(0)

        else:
            # Fallback to standard PyTorch execution
            self.load_weights_if_exists()
            self.active_engine = "pytorch"
            if self.weights_loaded:
                self.engine_status.setText("Status: Native PyTorch Active")

    def predict_tensor(self, input_tensor):
        """
        Inference Router: Dispatches the prepared tensor to the currently active backend framework.
        """
        if self.active_engine == "openvino":
            results = self.ov_infer_request.infer({self.ov_input_layer: input_tensor.numpy()})
            return torch.tensor(results[self.ov_output_layer])

        elif self.active_engine == "onnx":
            ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor.numpy()}
            ort_outs = self.ort_session.run(None, ort_inputs)
            return torch.tensor(ort_outs[0])

        elif self.active_engine == "deepx":
            # Since the compiler has fused div(255) and normalize into the hardware weights at the lower level,
            # the NPU API expects to receive the raw, unprocessed pixel scale (0.0 ~ 255.0).
            raw_inputs = (input_tensor * 0.5 + 0.5) * 255.0

            # The hardware accelerator expects the underlying type to be UINT8 (not Float32) when processing fused image preprocessing.
            # If not cast to uint8, the 4 bytes occupied by a Float32 value will be misread as 4 independent pixels by the NPU, causing severe mispredictions.
            np_inputs = np.ascontiguousarray(raw_inputs.numpy().astype(np.uint8))

            # 3. Input shape remains consistent with ONNX at [1, 1, 28, 28]
            np_outs = self.dx_engine.run([np_inputs])
            return torch.tensor(np_outs[0])

        else:
            # Native PyTorch execution path
            self.model.eval()
            with torch.no_grad():
                return self.model(input_tensor.to(self.device)).cpu()

    def init_ui(self):
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        tabs.addTab(self.create_training_tab(), "Training")
        tabs.addTab(self.create_inference_tab(), "Inference")

    def create_training_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        config_box = QGroupBox("Hyperparameter")
        form_layout = QFormLayout()
        self.epochs_input = QLineEdit("5")
        self.lr_input = QLineEdit("0.001")
        form_layout.addRow("Epoch Count:", self.epochs_input)
        form_layout.addRow("Learning Rate:", self.lr_input)
        config_box.setLayout(form_layout)
        layout.addWidget(config_box)

        gpu_box = QGroupBox("Hardware Acceleration Target")
        gpu_layout = QHBoxLayout()

        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU")

        # Dynamically inspect the runtime environment for NVIDIA CUDA capabilities
        if torch.cuda.is_available():
            self.device_combo.addItem("NVIDIA CUDA (GPU)")

        # Dynamically inspect the runtime environment for Intel XPU capabilities
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device_combo.addItem("Intel XPU (GPU)")

        # Smart Auto-Selection: Automatically default to the most powerful detected accelerator
        # If any GPU backend is discovered, pre-select the last added advanced device
        if self.device_combo.count() > 1:
            self.device_combo.setCurrentIndex(self.device_combo.count() - 1)

        gpu_layout.addWidget(QLabel("Select Compute Device:"))
        gpu_layout.addWidget(self.device_combo)
        gpu_layout.addStretch()
        gpu_box.setLayout(gpu_layout)
        layout.addWidget(gpu_box)

        self.aug_checkbox = QCheckBox("Enable Data Augmentation (Random Affine)")
        self.aug_checkbox.setChecked(True)
        layout.addWidget(self.aug_checkbox)

        self.isolate_checkbox = QCheckBox("Isolate model during training (Apply weights only after completion)")
        self.isolate_checkbox.setChecked(True)
        layout.addWidget(self.isolate_checkbox)

        # Deployment & Export Configuration
        export_box = QGroupBox("Edge Deployment Options")
        export_layout = QVBoxLayout()

        self.export_onnx_checkbox = QCheckBox("Export to ONNX (.onnx)")
        self.export_onnx_checkbox.setChecked(False)
        export_layout.addWidget(self.export_onnx_checkbox)

        self.export_ir_checkbox = QCheckBox("Export to OpenVINO IR (.xml / .bin)")
        self.export_ir_checkbox.setChecked(False)
        if not OPENVINO_AVAILABLE:
            self.export_ir_checkbox.setEnabled(False)
            self.export_ir_checkbox.setText("Export to OpenVINO IR (Requires 'openvino' pip package)")
        export_layout.addWidget(self.export_ir_checkbox)

        export_box.setLayout(export_layout)
        layout.addWidget(export_box)

        self.start_train_btn = QPushButton("Start Training")
        self.start_train_btn.clicked.connect(self.start_training)
        layout.addWidget(self.start_train_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.console_log = QTextEdit()
        self.console_log.setReadOnly(True)
        if self.weights_loaded:
            self.console_log.append("Notice: Pre-trained weights loaded. Ready for inference.")
        else:
            self.console_log.append("Warning: No pre-trained weights found. Please train the model first.")
        layout.addWidget(self.console_log)

        tab.setLayout(layout)
        return tab

    def create_inference_tab(self):
        tab = QWidget()
        tab_layout = QVBoxLayout()

        # Engine Selection Top Bar
        engine_box = QGroupBox("Inference Engine Runtime Target")
        engine_layout = QHBoxLayout()

        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Native PyTorch (.pth)")

        if OPENVINO_AVAILABLE:
            try:
                core = ov.Core()
                for device_name in core.available_devices:
                    self.engine_combo.addItem(f"OpenVINO IR (.xml) - {device_name}")
            except Exception:
                self.engine_combo.addItem("OpenVINO IR (.xml) - AUTO")

        if ONNX_AVAILABLE:
            self.engine_combo.addItem("ONNX Runtime (.onnx) - CPU/GPU")

        if DEEPX_AVAILABLE:
            self.engine_combo.addItem("DEEPX NPU (.dxnn)")

        self.engine_combo.currentIndexChanged.connect(self.reload_inference_engine)

        self.engine_status = QLabel("Status: Awaiting Initialization")
        engine_layout.addWidget(QLabel("Active Backend:"))
        engine_layout.addWidget(self.engine_combo)
        engine_layout.addWidget(self.engine_status)
        engine_layout.addStretch()
        engine_box.setLayout(engine_layout)

        tab_layout.addWidget(engine_box)
        main_layout = QHBoxLayout()

        # ==========================================
        # Panel A: Dataset Random Sampling Component
        # ==========================================
        dataset_box = QGroupBox("Input Preview")
        dataset_box.setFixedWidth(256)
        dataset_layout = QVBoxLayout()

        self.dataset_img_label = GridLabel()
        self.dataset_img_label.setAlignment(Qt.AlignCenter)
        placeholder = QPixmap(280, 280)
        placeholder.fill(Qt.black)
        self.dataset_img_label.setPixmap(placeholder)
        dataset_layout.addWidget(self.dataset_img_label)

        self.random_btn = QPushButton("Sample Random Image")
        self.random_btn.clicked.connect(self.run_random_inference)
        dataset_layout.addWidget(self.random_btn)

        # Auto Test Controls
        sample_btn_layout = QHBoxLayout()
        self.start_auto_btn = QPushButton("Start Auto Test")
        self.start_auto_btn.clicked.connect(self.start_auto_test)

        self.stop_auto_btn = QPushButton("Stop Auto Test")
        self.stop_auto_btn.setEnabled(False)
        self.stop_auto_btn.clicked.connect(self.stop_auto_test)

        sample_btn_layout.addWidget(self.start_auto_btn)
        sample_btn_layout.addWidget(self.stop_auto_btn)
        dataset_layout.addLayout(sample_btn_layout)

        self.dataset_res_label = QLabel("Prediction: N/A\nGround Truth: N/A")
        self.dataset_res_label.setAlignment(Qt.AlignCenter)
        self.dataset_res_label.setWordWrap(True)
        dataset_layout.addWidget(self.dataset_res_label)

        dataset_layout.addStretch()
        dataset_box.setLayout(dataset_layout)
        main_layout.addWidget(dataset_box)

        # ==========================================
        # Panel B: Realtime Drawing Board Component
        # ==========================================
        canvas_box = QGroupBox("Drawing Canvas")
        canvas_box.setFixedWidth(256)
        canvas_layout = QVBoxLayout()

        self.canvas = DrawingCanvas()
        canvas_layout.addWidget(self.canvas)

        self.bbox_checkbox = QCheckBox("Enable Auto Bounding Box Alignment")
        self.bbox_checkbox.setChecked(True)
        canvas_layout.addWidget(self.bbox_checkbox)

        btn_layout = QHBoxLayout()
        self.predict_canvas_btn = QPushButton("Predict Canvas")
        self.predict_canvas_btn.clicked.connect(self.run_canvas_inference)
        self.clear_canvas_btn = QPushButton("Clear Canvas")
        self.clear_canvas_btn.clicked.connect(self.canvas.clear_canvas)
        btn_layout.addWidget(self.predict_canvas_btn)
        btn_layout.addWidget(self.clear_canvas_btn)
        canvas_layout.addLayout(btn_layout)

        self.canvas_res_label = QLabel("Canvas Prediction: N/A")
        self.canvas_res_label.setAlignment(Qt.AlignCenter)
        canvas_layout.addWidget(self.canvas_res_label)
        canvas_layout.addStretch()
        canvas_box.setLayout(canvas_layout)
        main_layout.addWidget(canvas_box)

        # ==========================================
        # Panel C: Probability Distribution Component
        # ==========================================
        prob_box = QGroupBox("Probability Distribution (Softmax)")
        prob_box.setFixedWidth(256)
        prob_layout = QVBoxLayout()

        for i in range(10):
            row_layout = QHBoxLayout()

            lbl = QLabel(f"{i}:")
            lbl.setFixedWidth(20)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setStyleSheet("QProgressBar { border: 1px solid grey; border-radius: 2px; text-align: center; } "
                              "QProgressBar::chunk { background-color: #4CAF50; width: 10px; }")

            row_layout.addWidget(lbl)
            row_layout.addWidget(bar)
            prob_layout.addLayout(row_layout)
            self.prob_bars.append(bar)

        prob_box.setLayout(prob_layout)
        main_layout.addWidget(prob_box)

        tab_layout.addLayout(main_layout)
        tab.setLayout(tab_layout)

        # Initialize the default PyTorch engine on UI load
        self.reload_inference_engine()
        return tab

    def start_training(self):
        use_aug = self.aug_checkbox.isChecked()
        # Regenerate DataLoader dynamically based on augmentation selection
        self.train_loader, _, self.test_dataset = get_mnist_loaders(batch_size=64, use_augmentation=use_aug)

        self.start_train_btn.setEnabled(False)
        self.console_log.clear()
        self.console_log.append(f"Starting training loop... (Augmentation: {use_aug})")
        self.progress_bar.setValue(0)

        epochs = int(self.epochs_input.text())
        lr = float(self.lr_input.text())

        selected_device_text = self.device_combo.currentText()

        if "XPU" in selected_device_text:
            train_device = torch.device("xpu")
            self.console_log.append("System: Intel XPU target selected for model training.")
        elif "CUDA" in selected_device_text:
            train_device = torch.device("cuda")
            self.console_log.append("System: NVIDIA CUDA target selected for model training.")
        else:
            train_device = torch.device("cpu")
            self.console_log.append("Notice: Computing via standard CPU pipeline.")

        if self.isolate_checkbox.isChecked():
            # Safe Mode: Protect inference engine from mutating weights
            train_model = copy.deepcopy(self.model)
            self.console_log.append("System: Isolated training mode activated.")
        else:
            # Live Mode: Direct reference manipulation
            train_model = self.model
            self.console_log.append("System: Live training mode activated.")

        train_model = train_model.to(train_device)

        export_onnx = self.export_onnx_checkbox.isChecked()
        export_ir = self.export_ir_checkbox.isChecked()

        # Dispatch the worker thread to prevent UI freezing
        self.thread = TrainingThread(train_model, train_device, self.train_loader, epochs, lr, WEIGHT_PATH, export_onnx, export_ir)
        self.thread.log_signal.connect(self.console_log.append)
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.finished_signal.connect(self.on_training_finished)
        self.thread.start()

    def on_training_finished(self, success):
        self.start_train_btn.setEnabled(True)
        if success:
            if self.isolate_checkbox.isChecked():
                self.load_weights_if_exists()
                QMessageBox.information(self, "Status Update", "Isolated training complete. New weights loaded.")
            else:
                self.weights_loaded = True
                QMessageBox.information(self, "Status Update", "Live training complete. Model weights are active.")
        else:
            QMessageBox.critical(self, "Execution Fault", "An anomaly halted the execution chain during training.")

    def run_random_inference(self):
        if not self.weights_loaded:
            QMessageBox.warning(self, "Halted", "Active network matrices unpopulated. Execute training loop first.")
            return

        # 1. Random Selection and Left Panel UI Update
        idx = random.randint(0, len(self.test_dataset) - 1)
        image_tensor, label = self.test_dataset[idx]

        # De-normalize image from PyTorch dataset back to standard RGB visualization space
        img_np = (image_tensor.squeeze().numpy() * 0.5 + 0.5) * 255
        img_np = img_np.astype(np.uint8)
        qimg = QImage(img_np.data, 28, 28, 28, QImage.Format_Grayscale8).copy()
        self.dataset_img_label.setPixmap(QPixmap.fromImage(qimg))

        # 2. Inference Execution via Router
        input_tensor = image_tensor.unsqueeze(0)  # Keep on CPU: required format for ONNX/OpenVINO
        output = self.predict_tensor(input_tensor)
        _, predicted = torch.max(output, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]

        self.dataset_res_label.setText(f"Prediction: {predicted.item()}\nGround Truth: {label}")

        # 3. Probability Distribution Panel Update
        for i in range(10):
            prob_percent = int(probabilities[i].item() * 100)
            self.prob_bars[i].setValue(prob_percent)

            # Highlight the dominant predicted class
            if i == predicted.item():
                self.prob_bars[i].setStyleSheet("QProgressBar { border: 1px solid grey; border-radius: 2px; text-align: center; } "
                                                "QProgressBar::chunk { background-color: #2196F3; }")
            else:
                self.prob_bars[i].setStyleSheet("QProgressBar { border: 1px solid grey; border-radius: 2px; text-align: center; } "
                                                "QProgressBar::chunk { background-color: #4CAF50; }")

        # 4. State Synchronization: Clear canvas to prevent confusion
        self.canvas.clear_canvas()
        self.canvas_res_label.setText("Canvas Prediction: N/A\nMode: Waiting for input")

    def run_canvas_inference(self):
        if not self.weights_loaded:
            QMessageBox.warning(self, "Halted", "Active network matrices unpopulated. Execute training loop first.")
            return

        use_bbox = self.bbox_checkbox.isChecked()

        # Capture user drawing and process into normalized tensor
        input_tensor, processed_img, bbox = self.canvas.get_tensor(use_bbox=use_bbox)

        # Render Bounding Box overlay on the drawing canvas
        if use_bbox and bbox:
            self.canvas.current_bbox = bbox
        else:
            self.canvas.current_bbox = None
        self.canvas.update()

        # Project the processed 28x28 internal representation to the left UI panel
        qimg = QImage(processed_img.tobytes(), 28, 28, QImage.Format_Grayscale8)
        self.dataset_img_label.setPixmap(QPixmap.fromImage(qimg))
        self.dataset_res_label.setText("Source: Hand-drawn (Processed Image)\nReady for Inference.")

        # Execute Inference
        output = self.predict_tensor(input_tensor)
        _, predicted = torch.max(output, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]

        for i in range(10):
            prob_percent = int(probabilities[i].item() * 100)
            self.prob_bars[i].setValue(prob_percent)

            if i == predicted.item():
                self.prob_bars[i].setStyleSheet("QProgressBar { border: 1px solid grey; border-radius: 2px; text-align: center; } "
                                                "QProgressBar::chunk { background-color: #2196F3; }")
            else:
                self.prob_bars[i].setStyleSheet("QProgressBar { border: 1px solid grey; border-radius: 2px; text-align: center; } "
                                                "QProgressBar::chunk { background-color: #4CAF50; }")

        mode_text = "Optimized (BBox Active)" if use_bbox else "Baseline (Naive)"
        self.canvas_res_label.setText(f"Canvas Prediction: {predicted.item()}\nMode: {mode_text}")

        # Flag the canvas to auto-clear upon the next user click
        self.canvas.needs_clear = True

    def start_auto_test(self):
        """Unlocks the QTimer constraints to execute extreme-speed automated inference."""
        if not self.weights_loaded:
            QMessageBox.warning(self, "Halted", "Active network matrices unpopulated. Please train the model first.")
            return

        self.start_auto_btn.setEnabled(False)
        self.stop_auto_btn.setEnabled(True)
        self.random_btn.setEnabled(False)
        self.engine_combo.setEnabled(False)
        self.bbox_checkbox.setEnabled(False)
        self.predict_canvas_btn.setEnabled(False)
        self.clear_canvas_btn.setEnabled(False)

        # Setting interval to 0 triggers timeout whenever the event loop is idle
        self.auto_timer.start(0)

    def stop_auto_test(self):
        """Halts the automated infinite inference stream."""
        self.auto_timer.stop()

        self.start_auto_btn.setEnabled(True)
        self.stop_auto_btn.setEnabled(False)
        self.random_btn.setEnabled(True)
        self.engine_combo.setEnabled(True)
        self.bbox_checkbox.setEnabled(True)
        self.predict_canvas_btn.setEnabled(True)
        self.clear_canvas_btn.setEnabled(True)


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    gui = MNISTGuiApp()
    gui.show()
    sys.exit(app.exec_())
