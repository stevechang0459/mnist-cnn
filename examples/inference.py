import torch
import matplotlib.pyplot as plt
import numpy as np

from model import SimpleCNN
from dataset import get_mnist_loaders

# --- Global Configuration ---
# Relative path to the serialized PyTorch state dictionary
WEIGHT_PATH = "../models/mnist_cnn_weights.pth"

def visualize_predictions(model, test_loader, device, num_images=6):
    """
    Extracts a random batch from the test dataset, executes inference,
    and plots the images alongside their predicted and ground-truth labels.

    Args:
        model (nn.Module): The trained PyTorch CNN model.
        test_loader (DataLoader): DataLoader providing the test dataset batches.
        device (torch.device): The hardware backend (CPU/GPU/XPU) to execute on.
        num_images (int): Number of sample images to display in the visualization grid.
    """
    print("Generating visualization...")

    # Lock the model into evaluation mode to freeze dynamic layers (e.g., Dropout, BatchNorm)
    model.eval()

    # Retrieve a single batch of image tensors and their corresponding labels
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Migrate the image tensor to the target hardware accelerator
    images_device = images.to(device)

    # Suspend the autograd engine to bypass gradient tracking, significantly saving memory and speeding up inference
    with torch.no_grad():
        outputs = model(images_device)
        # Extract the index of the highest probability prediction (the predicted class)
        _, predicted = torch.max(outputs, 1)

    # Transfer the prediction tensor back to system memory (CPU) for Matplotlib compatibility
    # Note: Matplotlib cannot render tensors residing in GPU/XPU memory
    predicted = predicted.cpu()

    # Initialize the Matplotlib visualization canvas
    fig = plt.figure(figsize=(10, 6))

    for idx in range(num_images):
        ax = fig.add_subplot(2, 3, idx + 1, xticks=[], yticks=[])

        # De-normalize the image tensor back to the [0.0, 1.0] range for visualization
        img = images[idx] * 0.5 + 0.5
        npimg = img.numpy()

        # Render the grayscale image (squeeze removes the singular channel dimension [1, 28, 28] -> [28, 28])
        plt.imshow(np.squeeze(npimg), cmap='gray')

        # Conditionally format the text color: Green for accurate predictions, Red for errors
        pred_label = predicted[idx].item()
        true_label = labels[idx].item()
        color = 'green' if pred_label == true_label else 'red'

        ax.set_title(f"Pred: {pred_label} (True: {true_label})", color=color, fontweight='bold')

    plt.tight_layout()
    plt.show()

def main():
    """
    The main inference pipeline:
    1. Hardware initialization
    2. Test data ingestion
    3. Model instantiation and weight loading
    4. Prediction execution and visualization
    """
    # 1. Hardware Target Initialization
    # Currently defaults to CPU. Uncomment the XPU line to enable Intel Arc GPU acceleration.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Running inference on device: {device}")

    # 2. Data Ingestion
    # We exclusively unpack the test_loader, explicitly discarding train/validation sets to save memory
    _, test_loader, _ = get_mnist_loaders(batch_size=64)

    # 3. Model Initialization & Weight Loading
    print(f"Loading model weights from {WEIGHT_PATH}...")

    # Instantiate the CNN architecture and deploy it to the selected hardware backend
    model = SimpleCNN().to(device)

    # Inject the pre-trained dictionary into our model architecture securely
    # weights_only=True restricts the unpickler to load only tensors, mitigating arbitrary code execution risks
    model.load_state_dict(torch.load(WEIGHT_PATH, weights_only=True))
    print("Model weights loaded successfully.")

    # 4. Execute Pipeline
    visualize_predictions(model, test_loader, device)

if __name__ == '__main__':
    main()
