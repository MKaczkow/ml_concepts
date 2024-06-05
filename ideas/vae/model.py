import torch
from torch import nn


class VAE(nn.Module):
    """Basic Variational Autoencoder.

    input img -> hidden dim -> mean, std -> sample z -> decoder -> output img
    """

    def __init__(self, input_dim, h_dim=200, z_dim=200, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.relu = nn.ReLU(inplace=True)

        # Encoder
        self.img_2_hidden = nn.Linear(input_dim, h_dim)
        # KL divergence
        self.hidden_2_mean = nn.Linear(h_dim, z_dim)
        self.hidden_2_std = nn.Linear(h_dim, z_dim)

        # Decoder
        self.z_2_hidden = nn.Linear(z_dim, h_dim)
        self.hidden_2_img = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        # q_pthi(z|x)
        h = self.img_2_hidden(x)
        h = self.relu(h)
        mean = self.hidden_2_mean(h)
        std = self.hidden_2_std(h)
        return mean, std

    def decode(self, z):
        # p_theta(x|z)
        h = self.z_2_hidden(z)
        img = self.hidden_2_img(h)
        img = torch.sigmoid(img)
        return img

    def forward(self, x):
        mean, std = self.encode(x)
        eps = torch.randn_like(std)
        # TODO: why reparametrize?
        z_reparametrized = mean + std * eps
        z_reconstructed = self.decode(z_reparametrized)
        return z_reconstructed, mean, std


if __name__ == "__main__":
    vae = VAE()
    print(vae)
