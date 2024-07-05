import torch
import random


def augment_image_sequence(tensor: torch.Tensor, p_swap=0.2, p_revert=0.2, p_drop=0.2) -> torch.Tensor:
    """This is inital work / draft of the function that will be used to augment image sequences for self-supervised learning. The function will be used to augment image sequences in a way that will help the model to learn temporal relationships between images. The function will swap neighboring images with p-swap probability, revert the sequence with p-revert probability, and drop an image from the sequence with p-drop probability. The function will be used in the self-supervised learning pipeline to augment image sequences.
    
    Args:
        tensor (torch.Tensor): Input tensor representing images sequence.
        p_swap (float, optional): Proba of swapping neighbouring images. Defaults to 0.2.
        p_revert (float, optional): Proba of reverting the whole sequence. Defaults to 0.2.
        p_drop (float, optional): Proba of dropping frame. Defaults to 0.2.

    Returns:
        torch.Tensor: Output tensor representing images sequence
    """    
    # TODO: performance - multiple for's are likely bad - use tensor ops instead
    # TODO: more savvy way of swapping (like exp-decay and not only neighboring)
    # TODO: combinations of operations (like swap and then revert) instead of separate probabilities
    # TODO: integrate with lightly transforms (multiview transforms)

    seq_len, C, H, W = tensor.shape
    sequence = [tensor[i] for i in range(seq_len)]
    
    # Swap neighboring images with p-swap probability
    for i in range(seq_len - 1):
        if random.random() < p_swap:
            sequence[i], sequence[i + 1] = sequence[i + 1], sequence[i]
    
    # Revert sequence with p-revert probability
    if random.random() < p_revert:
        sequence.reverse()
    
    # Drop image from sequence with p-drop probability and fill remaining place
    for i in range(seq_len):
        if random.random() < p_drop:
            sequence[i] = sequence[i - 1] if i > 0 else sequence[0]
    
    augmented_tensor = torch.stack(sequence)
    
    return augmented_tensor
