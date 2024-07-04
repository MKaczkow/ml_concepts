import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead


class SimCLRModelVariableEmbeddingSize(pl.LightningModule):
    def __init__(
        self,
        latent_space_size: int = 512,
        projection_head_hidden_dim: int = 512,
        max_epochs: int = 50,
    ):
        super().__init__()

        self.max_epochs = max_epochs
        self.latent_space_size = latent_space_size
        self.projection_head_hidden_dim = projection_head_hidden_dim

        self.training_step_outputs = []

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.connector = nn.Linear(resnet.fc.in_features, self.latent_space_size)
        self.projection_head = SimCLRProjectionHead(
            self.latent_space_size, self.projection_head_hidden_dim, 128
        )

        # warning:
        # self.latent_space_size and self.projection_head_hidden_dim
        # need to be kept in sync with the actual sizes of the layers in the projection head
        # (just for sanity check, so callback can access this data)

        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        h_changed_size = self.connector(h)
        # Latent space has size self.latent_space_size (self.connector output)
        # this is added to check if the size of the latent space contributes to the performance
        # (will loss change?). In original paper images from CIFAR-10 are used, which are 'harder'
        # than MNIST, so latent space may be smaller for our case.
        z = self.projection_head(h_changed_size)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.training_step_outputs.append(loss)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]


class SimCLRModel(pl.LightningModule):
    def __init__(self, max_epochs: int = 50):
        super().__init__()

        self.max_epochs = max_epochs
        self.training_step_outputs = []

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.training_step_outputs.append(loss)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]
