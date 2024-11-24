from pytorch_lightning import Trainer
from model import Lightning_model
from data import TestDataModule
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import ArgumentParser
import pytorch_lightning as pl
import wandb
import re
import os

def find_best_checkpoint(directory):
    """Find the checkpoint with the lowest validation loss in the given directory."""
    files = os.listdir(directory)
    min_loss = float('inf')
    best_checkpoint = None

    for file in files:
        match = re.search(r'val_loss=([0-9]+\.[0-9]+)', file)
        if match:
            val_loss = float(match.group(1))
            if val_loss < min_loss:
                min_loss = val_loss
                best_checkpoint = file

    return os.path.join(directory, best_checkpoint) if best_checkpoint else None

def test(hparams):
    """Run the test process with the given hyperparameters."""
    data_dir = './data'
    pl.seed_everything(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    batch_size = 64

    project = f"{hparams.data}_test_{hparams.date}_{hparams.ratio}"


    models = ['MFNet']
    for model_name in models:
        hparams.archi = model_name
        wandb_logger = WandbLogger(name=model_name, project=project)
        resize = (224, 224)
        data_module = TestDataModule(data_dir=data_dir, fold=4, batch_size=batch_size, data=hparams.data, resize=resize, ratio=hparams.ratio)

        checkpoint_dir = f'../ckpt'
        best_checkpoint_path = find_best_checkpoint(checkpoint_dir)

        if hparams.archi in ['unet', 'attunet', 'swinunetr', 'swinunet']:
            model = Lightning_model.load_from_checkpoint(checkpoint_path=best_checkpoint_path, archi=hparams.archi, lr=hparams.lr, sync=hparams.sync)
        else:
            resize = False
            data_module = TestDataModule(data_dir=data_dir, fold=4, batch_size=batch_size, data=hparams.data, resize=resize, ratio=hparams.ratio)
            model = Lightning_model.load_from_checkpoint(checkpoint_path=best_checkpoint_path, archi=hparams.archi, multi=True, lr=hparams.lr, sync=hparams.sync)

        trainer = Trainer(deterministic=False, accelerator=hparams.accelerator, devices=hparams.devices, max_epochs=hparams.epochs, logger=wandb_logger)
        trainer.test(model, datamodule=data_module)
        wandb.finish()


def parse_device_list(device_string):
    """Parse a comma-separated list of devices into a list of integers."""
    return [int(device) for device in device_string.split(',')]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default='0', type=parse_device_list)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--epochs", default=150)
    parser.add_argument("--archi", default='MFNet')
    parser.add_argument("--sync", default=False, type=bool)
    parser.add_argument("--data", default='camelyon', type=str)
    parser.add_argument("--test", default=1, type=int)
    parser.add_argument("--ds", default=True, type=bool)
    parser.add_argument("--multi", default=True, type=bool)
    parser.add_argument("--ratio", default=0, type=int)
    parser.add_argument("--date", default='0813', type=str)
    parser.add_argument("--token", default='9', type=int)

    args = parser.parse_args()
    test(args)