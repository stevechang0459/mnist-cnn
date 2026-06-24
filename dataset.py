import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=64, use_augmentation=False):
    """
    Downloads the MNIST dataset, applies dynamic transformations,
    and constructs PyTorch DataLoaders for training and inference.

    Args:
        batch_size (int): Number of samples per batch. Default is 64.
        use_augmentation (bool): If True, applies random affine transformations
                                 to increase model robustness. Default is False.

    Returns:
        tuple: (train_loader, test_loader, test_dataset)
               Note: test_dataset is returned natively (unbatched) alongside the loaders
               to allow the GUI to easily sample individual raw images for the preview panel.
    """
    # Base transformations required for both training and testing phases:
    # 1. ToTensor: Converts PIL Images [0, 255] to FloatTensors [0.0, 1.0]
    # 2. Normalize: Shifts the distribution to mean=0.5, std=0.5 (range [-1.0, 1.0])
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]

    if use_augmentation:
        # Inject morphological distortions to simulate real-world handwriting variations.
        # - degrees=15: Random rotation between -15 to +15 degrees.
        # - translate=(0.1, 0.1): Randomly shift image up to 10% horizontally/vertically.
        # - scale=(0.9, 1.1): Randomly zoom in/out by up to 10%.
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            *base_transforms
        ])
    else:
        # Baseline execution: Standard conversion without spatial distortion
        train_transform = transforms.Compose(base_transforms)

    # The validation/test set must NEVER be augmented. It must remain purely deterministic.
    test_transform = transforms.Compose(base_transforms)

    # Download and initialize the datasets from torchvision's official repository
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=test_transform, download=True)

    # Wrap datasets in DataLoaders to manage automated batching and shuffling
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test_dataset
