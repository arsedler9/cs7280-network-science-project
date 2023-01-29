from os import path
import torch
from torch.utils.data import random_split
import torch_geometric as pg
from torch_geometric.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data import BrainConnectivity
from models import BrainGAE, BrainVGAE

# Load the dataset and create the DataLoaders
dataset = BrainConnectivity('~/tmp/brain_graphs_weighted_LCC')
train_data, val_data = dataset[:850], dataset[850:]
train_loader = DataLoader(
    train_data, num_workers=32, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, num_workers=32, batch_size=32)

MODEL_DIR = path.expanduser('~/netsci/models/vgae_wt_lcc')
# checkpoint_callback = ModelCheckpoint(
#     monitor='val_loss',
#     dirpath=MODEL_DIR,
#     filename='vgae-{epoch:02d}-{val_loss:.2f}',
#     save_top_k=3,
#     mode='min')

# Need to figure out how to incorporate (exponential likelihood of?) 
# edge weights into the autoencoder
model = BrainVGAE(weighted=True)
trainer = pl.Trainer(
    gpus=1, 
    default_root_dir=MODEL_DIR,
    # callbacks=[checkpoint_callback],
    # auto_lr_find=True
)
# trainer.tune(model, train_loader, val_loader)
trainer.fit(model, train_loader, val_loader)
