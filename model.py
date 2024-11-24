
import pytorch_lightning as pl
import torch
from wsi_utils import bring_model
from utils import *
import torchmetrics
from torch.optim.lr_scheduler import StepLR

def dice_loss(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    
    return 1 - dice

class Lightning_model(pl.LightningModule):
    def __init__(self, archi, lr=3e-4, sync=False,multi=True,ds=True,num_tokens=8):
        super().__init__()
        self.archi = archi
        self.model = bring_model(archi=archi,ds=ds,num_tokens=num_tokens)
        self.criterion= nn.BCEWithLogitsLoss()
        self.lr = lr
        self.save_hyperparameters()
        self.pre_update_params = {}
        self.sync = sync
        self.multi=multi
        self.train_accuracy = torchmetrics.Accuracy(task="binary",threshold=0.5)
        self.train_precision = torchmetrics.Precision(task="binary",threshold=0.5)
        self.train_recall = torchmetrics.Recall(task="binary",threshold=0.5)
        self.train_f1_score = torchmetrics.F1Score(task="binary",threshold=0.5)
        self.train_specificity = torchmetrics.Specificity(task="binary",threshold=0.5)
        self.train_dice = torchmetrics.Dice(multiclass=False)  
        self.train_jaccard = torchmetrics.JaccardIndex(task="binary", num_classes=2)

        self.accuracy = torchmetrics.Accuracy(task="binary",threshold=0.5)
        self.precision = torchmetrics.Precision(task="binary",threshold=0.5)
        self.recall = torchmetrics.Recall(task="binary",threshold=0.5)
        self.f1_score = torchmetrics.F1Score(task="binary",threshold=0.5)
        self.specificity = torchmetrics.Specificity(task="binary",threshold=0.5)
        self.dice = torchmetrics.Dice(multiclass=False)  
        self.jaccard = torchmetrics.JaccardIndex(task="binary", num_classes=2)

        self.test_accuracy = torchmetrics.Accuracy(task="binary",threshold=0.5)
        self.test_precision = torchmetrics.Precision(task="binary",threshold=0.5)
        self.test_recall = torchmetrics.Recall(task="binary",threshold=0.5)
        self.test_f1_score = torchmetrics.F1Score(task="binary",threshold=0.5)
        self.test_specificity = torchmetrics.Specificity(task="binary",threshold=0.5)
        self.test_dice = torchmetrics.Dice(multiclass=False)  
        self.test_jaccard = torchmetrics.JaccardIndex(task="binary", num_classes=2)

        if self.archi in  ['Transfuse','DMRUnet','DSUnet','DHUnet','MFNet']:
            self.ds = True
        else:
            self.ds = False

        self.train_step_outputs=[]
        self.validation_step_outputs=[]
        self.test_step_outputs = []

    def forward(self,level2, x=None):
        # 모델 forward 정의
        if  self.archi == 'MFNet':
            if self.multi == True:
                return self.model(x, level2)
            else:
                return self.model(level2, level2)
        else:
            return self.model(level2)

    def training_step(self, batch, batch_idx):
        level2 = batch['image'].to(self.device)
        
        y = batch['mask'].to(self.device)
        if y.max() > 1:
            y = y.float() / 255 
        if  self.archi == 'MFNet':
            x = batch['level0'].to(self.device)
            if self.ds:
                y_classification = (y.view(y.size(0), -1).max(dim=1)[0] > 0).float().unsqueeze(1)
                preds1,preds2,preds3,token = self(level2,x)
                loss1 = self.criterion(preds1, y) + dice_loss(preds1, y)
                loss2 = self.criterion(preds2, y) + dice_loss(preds2, y)
                loss3 = self.criterion(preds3, y) + dice_loss(preds3, y)
                loss4 = self.criterion(token,y_classification)
                loss = 0.6*loss1+0.2*loss2+0.2*loss3+0.2*loss4
            else:
                preds1 = self(level2, x)
                loss = self.criterion(preds1,y)
        else:
            if self.ds:
                preds1,preds2,preds3 = self(level2)
                loss1 = self.criterion(preds1, y) + dice_loss(preds1, y)
                loss2 = self.criterion(preds2, y) + dice_loss(preds2, y)
                loss3 = self.criterion(preds3, y) + dice_loss(preds3, y)
                loss = 0.6*loss1+0.2*loss2+0.2*loss3
            else:
                preds1 = self(level2)
                loss = self.criterion(preds1,y)
                           
        preds_prob = torch.sigmoid(preds1)

        preds_labels = (preds_prob > 0.5).bool()
        preds_labels = preds_labels.view(preds_labels.size(0), -1)
        y = y.view(y.size(0), -1)
        # 메트릭 업데이트
        self.train_accuracy.update(preds_labels, y.bool())
        self.train_precision.update(preds_labels, y.bool())
        self.train_recall.update(preds_labels, y.bool())
        self.train_f1_score.update(preds_labels, y.bool())
        self.train_specificity.update(preds_labels, y.bool())
        self.train_dice.update(preds_labels, y.bool())
        self.train_jaccard.update(preds_labels, y.bool())

        self.train_step_outputs.append({"loss": loss})
        return {"loss": loss}

    def on_train_epoch_end(self):
        # 에폭 끝에서 메트릭 계산 및 로깅
        accuracy = self.train_accuracy.compute()
        precision = self.train_precision.compute()
        recall = self.train_recall.compute()
        f1_score = self.train_f1_score.compute()
        specificity = self.train_specificity.compute()
        dice_score = self.train_dice.compute()
        jaccard_score = self.train_jaccard.compute()

        self.log('train_loss', torch.stack([x['loss'] for x in self.train_step_outputs]).mean() ,prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('train_accuracy', accuracy,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('train_precision', precision,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('train_recall', recall,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('train_f1', f1_score,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('train_specificity', specificity,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('train_dice', dice_score,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('train_jaccard', jaccard_score,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync)

        # 메트릭 리셋
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1_score.reset()
        self.train_specificity.reset()
        self.train_dice.reset()
        self.train_jaccard.reset()

        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        level2 = batch['image']
        
        y = batch['mask']
        if y.max() > 1:
            y = y.float() / 255  # Normalizing the mask


        if  self.archi == 'MFNet':
            x = batch['level0']
            if self.ds:
                y_classification = (y.view(y.size(0), -1).max(dim=1)[0] > 0).float().unsqueeze(1)
                preds1,preds2,preds3,token = self(level2,x)
                loss1 = self.criterion(preds1, y) + dice_loss(preds1, y)
                loss2 = self.criterion(preds2, y) + dice_loss(preds2, y)
                loss3 = self.criterion(preds3, y) + dice_loss(preds3, y)
                loss4 = self.criterion(token,y_classification)
                loss = 0.6*loss1+0.2*loss2+0.2*loss3+0.2*loss4
            else:
                preds1 = self(level2, x)
                loss = self.criterion(preds1,y)
        else:
            if self.ds:
                preds1,preds2,preds3 = self(level2)
                loss1 = self.criterion(preds1, y) + dice_loss(preds1, y)
                loss2 = self.criterion(preds2, y) + dice_loss(preds2, y)
                loss3 = self.criterion(preds3, y) + dice_loss(preds3, y)
                loss = 0.6*loss1+0.2*loss2+0.2*loss3
            else:
                preds1 = self(level2)
                loss = self.criterion(preds1,y)     

        preds_prob = torch.sigmoid(preds1)
        preds_labels = (preds_prob > 0.5).bool()
        preds_labels = preds_labels.view(preds_labels.size(0), -1)
        y = y.view(y.size(0), -1)
        # 메트릭 업데이트
        self.accuracy.update(preds_labels, y.bool())
        self.precision.update(preds_labels, y.bool())
        self.recall.update(preds_labels, y.bool())
        self.f1_score.update(preds_labels, y.bool())
        self.specificity.update(preds_labels, y.bool())
        self.dice.update(preds_labels, y.bool())
        self.jaccard.update(preds_labels, y.bool())
        self.validation_step_outputs.append({"loss": loss})
        return {"loss": loss}

    def on_validation_epoch_end(self):
        # 에폭 끝에서 메트릭 계산 및 로깅

        self.log('val_loss', torch.stack([x['loss'] for x in self.validation_step_outputs]).mean(), prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('val_accuracy', self.accuracy.compute(), prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('val_precision', self.precision.compute(), prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('val_recall', self.recall.compute(), prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('val_f1', self.f1_score.compute(), prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('val_specificity', self.specificity.compute(), prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('val_dice', self.dice.compute(), prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('val_jaccard', self.jaccard.compute(), prog_bar=True, logger=True, sync_dist=self.sync)

        # 메트릭 리셋
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_score.reset()
        self.specificity.reset()
        self.dice.reset()
        self.jaccard.reset()
        # 메모리 정리
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        level2 = batch['image']
        
        y = batch['mask']
        ids = batch['id']
        if y.max() > 1:
            y = y.float() / 255  # Normalizing the mask

        if  self.archi == 'MFNet':
            x = batch['level0']
            if self.ds:
                y_classification = (y.view(y.size(0), -1).max(dim=1)[0] > 0).float().unsqueeze(1)
                preds1,preds2,preds3,token = self(level2,x)
                loss1 = self.criterion(preds1, y) + dice_loss(preds1, y)
                loss2 = self.criterion(preds2, y) + dice_loss(preds2, y)
                loss3 = self.criterion(preds3, y) + dice_loss(preds3, y)
                loss4 = self.criterion(token,y_classification)
                loss = 0.6*loss1+0.2*loss2+0.2*loss3+0.2*loss4
            else:
                preds1 = self(level2, x)
                loss = self.criterion(preds1,y)
        else:
            if self.ds:
                preds1,preds2,preds3 = self(level2)
                loss1 = self.criterion(preds1, y) + dice_loss(preds1, y)
                loss2 = self.criterion(preds2, y) + dice_loss(preds2, y)
                loss3 = self.criterion(preds3, y) + dice_loss(preds3, y)
                loss = 0.6*loss1+0.2*loss2+0.2*loss3
            else:
                preds1 = self(level2)
                loss = self.criterion(preds1,y)     

        preds_prob = torch.sigmoid(preds1)
        preds_labels = (preds_prob > 0.5).bool()
            
        preds_labels = preds_labels.view(preds_labels.size(0), -1)
        y = y.view(y.size(0), -1)
    
        # 메트릭 업데이트
        self.test_accuracy.update(preds_labels, y.bool())
        self.test_precision.update(preds_labels, y.bool())
        self.test_recall.update(preds_labels, y.bool())
        self.test_f1_score.update(preds_labels, y.bool())
        self.test_specificity.update(preds_labels, y.bool())
        self.test_dice.update(preds_labels, y.bool())
        self.test_jaccard.update(preds_labels, y.bool())
        self.test_step_outputs.append({"loss": loss})
        
        return {"loss": loss}

    def on_test_epoch_end(self):
        # 테스트 에폭 끝에서 메트릭 계산 및 로깅
        self.log('test_loss', torch.stack([x['loss'] for x in self.test_step_outputs]).mean(), prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('test_accuracy', self.test_accuracy.compute(), prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('test_precision', self.test_precision.compute(), prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('test_recall', self.test_recall.compute(), prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('test_f1', self.test_f1_score.compute(), prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('test_specificity', self.test_specificity.compute(), prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('test_dice', self.test_dice.compute(), prog_bar=True, logger=True, sync_dist=self.sync)
        self.log('test_jaccard', self.test_jaccard.compute(), prog_bar=True, logger=True, sync_dist=self.sync)

        # Reset metrics
        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1_score.reset()
        self.test_specificity.reset()
        self.test_dice.reset()
        self.test_jaccard.reset()
        # Clear memory
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        # 옵티마이저 (및 필요시 학습률 스케줄러) 설정
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.7)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    