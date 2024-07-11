from augmentations import augment_image_sequence
from torchvision import datasets, transforms
from utils import create_image_sequence, visualize_sequences

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

mnist_train = datasets.MNIST(root=".", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root=".", train=False, download=True, transform=transform)

# Config (comment probas out for default values)
p_swap = 1.0
p_revert = 0.0
p_drop = 0.0

seq_len = 5

# Demo the function
original_sequence = create_image_sequence(mnist_test, seq_len)
augmented_sequence = augment_image_sequence(
    original_sequence, p_swap=p_swap, p_revert=p_revert, p_drop=p_drop
)

visualize_sequences(original_sequence, augmented_sequence, seq_len)
