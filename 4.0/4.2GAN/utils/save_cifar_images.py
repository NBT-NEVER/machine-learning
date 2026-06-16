import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

def save_cifar_images(root='./data', output_dir='./cifar_images', train=True):
    """Save CIFAR dataset as individual images"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Skip saving CIFAR images for now to avoid torch._six error
    # print(f"Skipping saving CIFAR images to avoid torch._six error")
    # return
    
    # Load CIFAR dataset
    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=None)
    
    # Save each image
    for i, (image, label) in enumerate(dataset):
        # Create subdirectory for each class if it doesn't exist
        class_dir = os.path.join(output_dir, str(label))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        # Save the image
        image_path = os.path.join(class_dir, f'image_{i}.png')
        image.save(image_path)
    
    print(f"Saved {len(dataset)} images to {output_dir}")

if __name__ == '__main__':
    # Save both train and test datasets
    save_cifar_images(train=True)
    save_cifar_images(train=False)