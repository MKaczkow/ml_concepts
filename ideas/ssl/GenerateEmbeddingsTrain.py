from sklearn.manifold import TSNE
import torch
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt


class GenerateEmbeddingsTrain(Callback):
    def __init__(self, model, dataloader, save_dir, device="cuda"):
        self.model = model
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.device = device

    def on_train_epoch_end(self, trainer, pl_module):
        embeddings = []
        for batch in self.dataloader:
            (x0, _), _, _ = batch
            h = pl_module.backbone(x0.to(self.device)).flatten(start_dim=1)
            embeddings.append(h)

        # save simple embeddings
        embeddings = torch.cat(embeddings)
        torch.save(
            embeddings, f"{self.save_dir}/embeddings_train_{trainer.current_epoch}.pth"
        )

        # save tsne embeddings
        tsne = TSNE(n_components=2)
        h_embedded = tsne.fit_transform(embeddings.cpu().detach().numpy())
        torch.save(
            h_embedded,
            f"{self.save_dir}/embeddings_train_tsne_{trainer.current_epoch}.pth",
        )

        # plot tsne embeddings
        plt.scatter(h_embedded[:, 0], h_embedded[:, 1])
        plt.savefig(
            f"{self.save_dir}/embeddings_train_tsne_plot_{trainer.current_epoch}.png"
        )
        # plt.show()
