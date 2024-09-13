# PyTorch Lightning

## General
* separating research code, engineering code and add-ons

## Architecture
* `Trainer` -> engineering code
* `LightningModule` -> research code
* `Callback` -> add-ons

## Callbacks
* `callback` is really a general term from CS, which means pluging-in some code to be executed at arbitrary point in the code, which means arbitrary time and state of code execution
* in `lightning` some functionalities, like `early stopping`, `model checkpointing`, `logging`, `learning rate scheduling`, etc. are implemented as `callbacks`

## References
* [part of yt tutorial on callbacks](https://www.youtube.com/watch?v=Wcze6oGch1g)
* [docs, obviously](https://pytorch-lightning.readthedocs.io/en/1.0.8/callbacks.html)
