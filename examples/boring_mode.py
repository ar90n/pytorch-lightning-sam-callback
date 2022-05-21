import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning_sam_callback import SAM


class RandomDataset(Dataset[torch.Tensor]):
    def __init__(self, size: int, num_samples: int):
        self.data = torch.randn(num_samples, size)

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        return self(batch).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


model = BoringModel()
trainer = Trainer(max_epochs=3, callbacks=[SAM()])
trainer.fit(model, train_dataloaders=DataLoader(RandomDataset(32, 64), batch_size=2))
