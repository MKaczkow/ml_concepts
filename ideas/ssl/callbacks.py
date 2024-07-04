import pytorch_lightning as pl
import torch


class LossLoggingCallback(pl.Callback):
    """This callback is used to 'manually' log the mean loss for the epoch. This is needed because the loss is not returned from the training_step method, but is stored in the training_step_outputs list."""

    def __init__(self):
        self.train_epoch_losses = []

    def on_train_epoch_end(self, trainer, pl_module):

        # Log the mean loss for the epoch (training step outputs are actually the losses)
        epoch_mean_loss = torch.stack(pl_module.training_step_outputs).mean()
        pl_module.log("training_epoch_mean", epoch_mean_loss)
        pl_module.training_step_outputs.clear()

        # This is weird, but needed to keep mlflow 'logging' outside of pytorch-lightning
        # because, mlflow is used in 'wider' context (also logging visualizations, etc.)
        self.train_epoch_losses.append(epoch_mean_loss.item())


class HiddenDimensionsCheckingCallback(pl.Callback):
    """This callback is used to keep track of the hidden dimensions of the model, namely the latent space size and the projection head hidden dimension."""

    def __init__(self):
        self.latent_space_size = 0
        self.projection_head_hidden_dim = 0

    def on_train_start(self, trainer, pl_module):
        # This is weird, but needed to keep mlflow 'logging' outside of pytorch-lightning
        # because, mlflow is used in 'wider' context (also logging visualizations, etc.)
        self.latent_space_size = pl_module.latent_space_size
        self.projection_head_hidden_dim = pl_module.projection_head_hidden_dim
