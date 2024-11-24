from pytorch_lightning import Trainer
from model import *
from data import TestDataModule
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from argparse import ArgumentParser
import pytorch_lightning as pl
import wandb 

def train(hparams):
        
        data_dir = './data'
        
        pl.seed_everything(42)
        torch.backends.cudnn.benchmark=True
        torch.backends.cudnn.deterministic=False
        batch_size= 64
        project = hparams.project


        checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',  # 모니터링할 메트릭
                dirpath=f'{data_dir}/my_model/{hparams.data}/{project}/{hparams.archi}',  # 체크포인트 저장 경로
                filename=f'{hparams.archi}-model-{{epoch:03d}}-{{val_loss:.6f}}',  # 파일명 포맷 변경
                save_top_k=3,  # 가장 좋은 k개의 모델만 저장
                save_last=True,
                mode='min',  # 'min': 메트릭 최소화, 'max': 메트릭 최대화
            )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        wandb_logger = WandbLogger(name=f'{hparams.archi}_{hparams.num_tokens}', project=project)
        

        if hparams.archi in ['unet','attunet','swinunetr','swinunet']:
            resize = (224,224)
            data_module = TestDataModule(data_dir=data_dir, fold=4, batch_size=batch_size,resize=resize,data=hparams.data,ratio=50)
            model = Lightning_model(archi=hparams.archi, lr=hparams.lr, sync=hparams.sync)

            trainer = Trainer(deterministic=False,
                accelerator=hparams.accelerator, 
                devices=hparams.devices,
                max_epochs=hparams.epochs,
                logger=wandb_logger,
                callbacks=[checkpoint_callback, lr_monitor]
            )
            
            # Run testing
            trainer.fit(model, datamodule=data_module)
            wandb.finish()
        elif hparams.archi in ['DMRUnet','MFNet']:
            if hparams.data == 'camelyon':
                resize = False
            else:
                resize = (224,224)
            data_module = TestDataModule(data_dir=data_dir, fold=4, batch_size=batch_size,resize=resize,data=hparams.data,ratio=50)
            model = Lightning_model(archi=hparams.archi,multi=True, lr=hparams.lr, sync=hparams.sync,ds=True,num_tokens=hparams.num_tokens)

            trainer = Trainer(deterministic=False,
                accelerator=hparams.accelerator, 
                devices=hparams.devices,
                max_epochs=hparams.epochs,
                logger=wandb_logger,
                callbacks=[checkpoint_callback, lr_monitor]
            )

            # Run testing
            trainer.fit(model, datamodule=data_module)
            wandb.finish()
        else:

            if hparams.archi =='Transfuse':
                resize = (192,256)
            elif hparams.archi == 'DSUnet':
                resize = (256,256)
            else:
                resize = (224,224)

            data_module = TestDataModule(data_dir=data_dir, fold=4, batch_size=batch_size,resize=resize,data=hparams.data,ratio=50)

            model = Lightning_model(archi=hparams.archi, lr=hparams.lr, sync=hparams.sync,ds=True)

            trainer = Trainer(deterministic=False,
                accelerator=hparams.accelerator, 
                devices=hparams.devices,
                max_epochs=hparams.epochs,
                logger=wandb_logger,
                callbacks=[checkpoint_callback, lr_monitor]
            )

            # Run testing
            trainer.fit(model, datamodule=data_module)
            wandb.finish()



def parse_device_list(device_string):
    # Split the string by commas and convert each substring to an integer
    return [int(device) for device in device_string.split(',')]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default='0', type=parse_device_list)
    parser.add_argument("--lr", default=0.00001, type= float)
    parser.add_argument("--epochs", default=50, type= int)
    parser.add_argument("--archi", default='MFNet')
    parser.add_argument("--sync", default=False, type=bool)
    parser.add_argument("--data", default='camelyon')
    parser.add_argument("--ds", default=True, type=bool)
    parser.add_argument("--multi", default=True, type=bool)
    parser.add_argument("--project", default='wsi', type=str)
    parser.add_argument("--num_tokens", default=10 , type=int)
    args = parser.parse_args()

    train(args)