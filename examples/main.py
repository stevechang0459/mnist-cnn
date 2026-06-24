import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from model import SimpleCNN
from dataset import get_mnist_loaders

# --- Global Hyperparameters ---
# These parameters govern the training dynamics and batch sizing
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

def visualize_predictions(model, test_loader, device, num_images=6):
    """
    Extracts a random batch from the test dataset, executes inference,
    and visually plots the images alongside their predicted and ground-truth labels.

    Args:
        model (nn.Module): The trained PyTorch model.
        test_loader (DataLoader): The DataLoader containing the test dataset.
        device (torch.device): The hardware backend (CPU/GPU/XPU) to execute on.
        num_images (int): Number of sample images to display in the grid.
    """
    print("Generating visualization...")

    # Lock the model into evaluation mode to freeze Dropout and BatchNorm layers
    model.eval()

    # Retrieve a single batch of image tensors and their corresponding labels
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Migrate the image tensor to the target hardware accelerator
    images_device = images.to(device)

    # Execute inference within a no_grad context to bypass the autograd engine (saves memory)
    with torch.no_grad():
        outputs = model(images_device)
        # Extract the index of the highest probability prediction
        _, predicted = torch.max(outputs, 1)

    # Transfer the prediction tensor back to system memory (CPU) for Matplotlib compatibility
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

        # Conditionally format the text color based on prediction accuracy
        pred_label = predicted[idx].item()
        true_label = labels[idx].item()
        color = 'green' if pred_label == true_label else 'red'

        ax.set_title(f"Pred: {pred_label} (True: {true_label})", color=color, fontweight='bold')

    plt.tight_layout()
    plt.show()

def main():
    """
    The main execution pipeline:
    1. Hardware initialization
    2. Data ingestion
    3. Model training loop
    4. Accuracy evaluation
    5. Result visualization
    """
    # Define the target hardware backend
    # Note: Currently hardcoded to CPU. Uncomment the specific accelerator logic as needed.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Instantiate DataLoaders. We unpack the tuple and discard the raw test dataset here.
    train_loader, test_loader, _ = get_mnist_loaders(batch_size=BATCH_SIZE)

    # Initialize the CNN architecture and deploy it to the selected hardware
    model = SimpleCNN().to(device)

    # Define the objective function (CrossEntropy for multi-class classification)
    # and the optimization algorithm (Adam)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # Phase 1: Model Training Loop
    # ==========================================
    print("Starting training process...")
    for epoch in range(EPOCHS):
        # Enable training mode (activates gradient tracking and dynamic layers like Dropout)
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            # Transfer batch data to the target hardware
            images = images.to(device)
            labels = labels.to(device)

            # Forward Pass: Compute the model's predictions and calculate the loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward Pass: Compute gradients and update network weights
            optimizer.zero_grad()  # Flush accumulated gradients from the previous iteration
            loss.backward()        # Calculate new gradients via backpropagation
            optimizer.step()       # Adjust weights based on the calculated gradients

            running_loss += loss.item()

        # Report the average loss per epoch
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

    # ==========================================
    # Phase 2: Inference & Evaluation
    # ==========================================
    print("Starting inference and evaluation...")

    # Lock the model into evaluation mode for deterministic outputs
    model.eval()
    correct = 0
    total = 0

    # Suspend the autograd engine to prevent memory leaks during evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Determine the predicted class by finding the maximum logit value
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the 10000 test images: {accuracy:.2f}%")

    # ==========================================
    # Phase 3: Visualization
    # ==========================================
    visualize_predictions(model, test_loader, device)

if __name__ == '__main__':
    main()
