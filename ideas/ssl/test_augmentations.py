import unittest
import torch

from augmentations import augment_image_sequence


class TestAugmentImageSequence(unittest.TestCase):
    def setUp(self):
        self.seq_len = 5
        self.C, self.H, self.W = 1, 28, 28
        self.tensor = torch.arange(self.seq_len * self.C * self.H * self.W, dtype=torch.float32).view(self.seq_len, self.C, self.H, self.W)

    def test_revert_sequence(self):
        augmented = augment_image_sequence(self.tensor, p_swap=0.0, p_revert=1.0, p_drop=0.0)
        expected = torch.flip(self.tensor, dims=[0])
        self.assertTrue(torch.equal(augmented, expected), "The sequence was not correctly reverted.")

    # FIXME: this is broken, as we don't know yet what 'swapping' with 1.0 proba will actually mean
    # def test_swap_sequence(self):
    #     augmented = augment_image_sequence(self.tensor, p_swap=1.0, p_revert=0.0, p_drop=0.0)
    #     for i in range(0, self.seq_len - 1, 2):
    #         self.assertTrue(torch.equal(augmented[i], self.tensor[i + 1]), "Neighboring images were not swapped correctly.")
    #         self.assertTrue(torch.equal(augmented[i + 1], self.tensor[i]), "Neighboring images were not swapped correctly.")

    def test_drop_image(self):
        augmented = augment_image_sequence(self.tensor, p_swap=0.0, p_revert=0.0, p_drop=1.0)
        for i in range(1, self.seq_len):
            self.assertTrue(torch.equal(augmented[i], augmented[i - 1]), "Dropped image was not replaced correctly.")

    def test_no_augmentation(self):
        augmented = augment_image_sequence(self.tensor, p_swap=0.0, p_revert=0.0, p_drop=0.0)
        self.assertTrue(torch.equal(augmented, self.tensor), "The sequence should not have been augmented.")

if __name__ == '__main__':
    unittest.main()
