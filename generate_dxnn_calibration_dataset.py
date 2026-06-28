import os
import cv2
import numpy as np
from torchvision import datasets, transforms

def generate_dxnn_calibration_dataset(output_dir="models/calibration_dataset", num_images=100):
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    print(f"Extracting {num_images} images for INT8 calibration...")
    for i in range(num_images):
        img_tensor, label = test_dataset[i]
        img_uint8 = (img_tensor.squeeze().numpy() * 255.0).astype(np.uint8)
        img_path = os.path.join(output_dir, f"calib_{i:03d}.png")
        cv2.imwrite(img_path, img_uint8)

if __name__ == "__main__":
    generate_dxnn_calibration_dataset()
