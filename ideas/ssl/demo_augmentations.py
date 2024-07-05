import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from augmentations import augment_image_sequence


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_train = datasets.MNIST(root='.', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='.', train=False, download=True, transform=transform)

def visualize_sequences(original_sequence, augmented_sequence, seq_len=5):
    fig, axs = plt.subplots(2, seq_len, figsize=(15, 3))
    
    for i in range(seq_len):
        axs[0, i].imshow(original_sequence[i].squeeze(), cmap='gray')
        axs[0, i].axis('off')
        
        axs[1, i].imshow(augmented_sequence[i].squeeze(), cmap='gray')
        axs[1, i].axis('off')
    
    axs[0, 0].set_title('Original Sequence')
    axs[1, 0].set_title('Augmented Sequence')
    plt.show()

def create_image_sequence(dataset, seq_len=5):
    indices = torch.randint(0, len(dataset), (seq_len,))
    images = torch.stack([dataset[i][0] for i in indices])
    return images

# Demo the function
seq_len = 5
original_sequence = create_image_sequence(mnist_test, seq_len)
augmented_sequence = augment_image_sequence(original_sequence)

visualize_sequences(original_sequence, augmented_sequence, seq_len)
