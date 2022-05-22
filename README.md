# pytorch-lightning-sam-callback
[![Build][build-shiled]][build-url]
[![Version][version-shield]][version-url]
[![Downloads][download-shield]][download-url]
[![Contributors][contributors-shield]][contributors-url]
[![Issues][issues-shield]][issues-url]
[![Codecov][codecov-shield]][codecov-url]
[![MIT][license-shield]][license-url]

pytorch-lightning-sam-callback is an implementation of [SAM](https://arxiv.org/abs/2010.01412) using pytorch-lightning's Callback API.
This project is motivated to integrate SAM with LightningModels without any modifications.

## Features
* [SAM](https://arxiv.org/abs/2010.01412) implementation
* Provided as pytorch-lightning's Callback API

## Features
* Mixed Precision Training is not supported

## Installation
```
$ pip install pytorch-lightning-sam-callback
```

## Example
```python
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer

from pytorch_lightning_sam_callback import SAM


class RandomDataset(Dataset):
    def __init__(self, size, num_samples):
        self.data = torch.randn(num_samples, size)

    def __getitem__(self, index):
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
```

## Training Result
![training_loss](https://github.com/ar90n/pytorch-lightning-sam-callback/blob/assets/images/training_loss.png?raw=true)
![validation_loss](https://github.com/ar90n/pytorch-lightning-sam-callback/blob/assets/images/validation_loss.png)

## See Also
* [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412)
* [davda54/sam](https://github.com/davda54/sam)

## License
[MIT](https://github.com/ar90n/pytorch-lightning-sam-callback/blob/main/LICENSE)

[download-shield]: https://img.shields.io/pypi/dm/pytorch-lightning-sam-callback?style=flat
[download-url]: https://pypi.org/project/pytorch-lightning-sam-callback/
[version-shield]: https://img.shields.io/pypi/v/pytorch-lightning-sam-callback?style=flat
[version-url]: https://pypi.org/project/pytorch-lightning-sam-callback/
[build-shiled]: https://img.shields.io/github/workflow/status/ar90n/pytorch-lightning-sam-callback/CI%20testing/main
[build-url]: https://github.com/ar90n/pytorch-lightning-sam-callback/actions/workflows/ci-testing.yml
[contributors-shield]: https://img.shields.io/github/contributors/ar90n/pytorch-lightning-sam-callback.svg?style=flat
[contributors-url]: https://github.com/ar90n/pytorch-lightning-sam-callback/graphs/contributors
[issues-shield]: https://img.shields.io/github/issues/ar90n/pytorch-lightning-sam-callback.svg?style=flat
[issues-url]: https://github.com/ar90n/pytorch-lightning-sam-callback/issues
[license-shield]: https://img.shields.io/github/license/ar90n/pytorch-lightning-sam-callback.svg?style=flat
[license-url]: https://github.com/ar90n/pytorch-lightning-sam-callback/blob/main/LICENSE
[codecov-shield]: https://codecov.io/gh/ar90n/pytorch-lightning-sam-callback/branch/main/graph/badge.svg?token=8GKU96ODLY
[codecov-url]: https://codecov.io/gh/ar90n/pytorch-lightning-sam-callback
