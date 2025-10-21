from .model_falcon import get_model
import sys
sys.path.append("..")
from config import cfg
import torch 
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import wandb

def neg_log(x):
    return -torch.log(x + 1e-5)

class MaskedActionModeling(LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.llm, *_ = get_model(cfg.train.llm_model, cfg.train.num_actions, cfg.train.llm_hidden_dim)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = kwargs.get('batch_size', 1)
        self.num_workers = kwargs.get('num_workers', 16)
        self.lr = kwargs.get('lr', 1e-5)
        self.criterion_action = nn.BCEWithLogitsLoss()
        self.criterion_state = nn.MSELoss()

    def forward(self, embeds, action_seq, patch_seq, gt_action):

        state, state_pred, state_gt = self.llm(
                        inputs_embeds=embeds,
                        actions=action_seq,
                        patch_sequence=patch_seq[:, 1:],
                        patch_size=cfg.data.patch_size,
                        pretrain=True)

        # =================geoexplorer state-action modeling==================
        loss_action = self.criterion_action(state, gt_action.float())
        loss_state = self.criterion_state(state_pred.float(), state_gt.float())

        loss = loss_action + loss_state
        #loss = loss_action

        return loss, loss_action, loss_state

    def shared_step(self, batch, batch_idx):
        embeds, action_seq, patch_seq, gt_action = batch
        loss, loss_action, loss_state  = self(embeds, np.array(action_seq).T.tolist(), patch_seq, gt_action)
        return loss, loss_action, loss_state

    def training_step(self, batch, batch_idx):
        loss, loss_action, loss_state = self.shared_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        wandb.log({"train_loss": loss, "train_loss_action": loss_action, "train_loss_state": loss_state})
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, loss_action, loss_state = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        #wandb.log({"val_loss": loss, "val_loss_action": loss_action, "val_loss_state": loss_state})
        
        return {"loss": loss, "num_correct": loss_action, "num_total": loss_state}

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                        shuffle=True,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        persistent_workers=True,
                        pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                        shuffle=False,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        persistent_workers=True,
                        pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.pretrain.hparams.lr, weight_decay=cfg.pretrain.hparams.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, cfg.pretrain.hparams.warmup)
        return [optimizer], [scheduler]