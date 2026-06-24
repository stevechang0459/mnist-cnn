import os
import torch
import torch.nn as nn
import torch.optim as optim

from model import SimpleCNN
from dataset import get_mnist_loaders

# --- Global Configuration & Hyperparameters ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
# Relative path to store the serialized PyTorch state dictionary
WEIGHT_PATH = "../models/mnist_cnn_weights.pth"

def main():
    """
    Executes the standalone training pipeline.
    This script initializes the network, runs the optimization loop over the MNIST dataset,
    and serializes the learned weights to the disk for downstream inference or OpenVINO export.
    """
    # 1. Hardware Target Initialization
    # Currently defaults to CPU. Uncomment the XPU line to enable Intel Arc GPU acceleration.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Training on device: {device}")

    # 2. Data Ingestion
    # We only unpack the train_loader here, explicitly discarding validation/test sets to save memory
    train_loader, _, _ = get_mnist_loaders(batch_size=BATCH_SIZE)

    # 3. Model & Optimizer Initialization
    # Instantiate the CNN architecture and deploy it to the selected hardware backend
    model = SimpleCNN().to(device)

    # CrossEntropyLoss combines LogSoftmax and NLLLoss (perfect for mutually exclusive classes like digits)
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer adapts the learning rate for each parameter dynamically
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # Phase 1: The Optimization Loop
    # ==========================================
    print("Starting training process...")
    for epoch in range(EPOCHS):
        # Engage training mode (activates gradients and dynamic layers like Dropout/BatchNorm)
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            # Migrate the tensor batches to the target hardware (CPU/GPU/XPU)
            images = images.to(device)
            labels = labels.to(device)

            # Step A: Flush the accumulated gradients from the previous iteration
            optimizer.zero_grad()

            # Step B: Forward pass - compute the network's predictions and evaluate the loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Step C: Backward pass - compute gradients via backpropagation
            loss.backward()

            # Step D: Optimization - update the network weights
            optimizer.step()

            running_loss += loss.item()

        # Log the average loss for the current epoch
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

    # ==========================================
    # Phase 2: Model Serialization
    # ==========================================
    print(f"Training complete! Saving model weights to {WEIGHT_PATH}...")

    # Ensure the target directory exists before attempting to save to prevent FileNotFoundError
    os.makedirs(os.path.dirname(WEIGHT_PATH), exist_ok=True)

    # model.state_dict() extracts a dictionary mapping each layer to its parameter tensor
    torch.save(model.state_dict(), WEIGHT_PATH)
    print("Model saved successfully.")

if __name__ == '__main__':
    main()
