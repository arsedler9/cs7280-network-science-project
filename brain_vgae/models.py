import torch 
import torch_geometric as pg
import pytorch_lightning as pl


class BrainGAE(pl.LightningModule):
    def __init__(self, input_dim=1015, hidden_dim=50, lr=2.5e-2):
        super().__init__()
        encoder = pg.nn.GCNConv(input_dim, hidden_dim, improved=True)
        self.gae = pg.nn.GAE(encoder)
        self.lr = lr
    
    def forward(self, batch):
        z = self.gae.encode(batch.x, batch.edge_index)
        return z
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        z = self.gae.encode(train_batch.x, train_batch.edge_index)
        loss = self.gae.recon_loss(z, train_batch.edge_index)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        z = self.gae.encode(val_batch.x, val_batch.edge_index)
        loss = self.gae.recon_loss(z, val_batch.edge_index)
        self.log('val_loss', loss)
        return loss


class VariationalEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.shared_gcn = pg.nn.GCNConv(input_dim, hidden_dim, improved=True)
        self.relu = torch.nn.ReLU()
        self.mean_gcn = pg.nn.GCNConv(hidden_dim, latent_dim, improved=True)
        self.logstd_gcn = pg.nn.GCNConv(hidden_dim, latent_dim, improved=True)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.shared_gcn(x, edge_index, edge_weight)
        h = self.relu(h)
        z_mean = self.mean_gcn(h, edge_index, edge_weight)
        z_logstd = self.logstd_gcn(h, edge_index, edge_weight)
        return z_mean, z_logstd


class BrainVGAE(pl.LightningModule):
    def __init__(self, 
                 input_dim=1015, 
                 hidden_dim=64,
                 latent_dim=32,
                 lr=2.5e-2, 
                 lam=1e-4,
                 weighted=False):
        super().__init__()
        encoder = VariationalEncoder(
            input_dim, hidden_dim, latent_dim)
        self.gae = pg.nn.VGAE(encoder)
        self.lam = lam
        self.lr = lr
        self.weighted = weighted
    
    def forward(self, batch):
        z = self.gae.encode(batch.x.float(), batch.edge_index)
        return z
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        edge_weight = train_batch.edge_attr[:,0] if self.weighted else None
        z = self.gae.encode(train_batch.x.float(), train_batch.edge_index, edge_weight)
        if self.weighted:
            recon_loss = self.mse_loss(
                z, train_batch.edge_index, edge_weight, train_batch.batch)
        else:
            recon_loss = self.gae.recon_loss(z, train_batch.edge_index)
        self.log('recon_loss', recon_loss)
        kl_loss = self.gae.kl_loss()
        self.log('kl_loss', kl_loss)
        loss = recon_loss + self.lam * kl_loss
        self.log('loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        edge_weight = val_batch.edge_attr[:,0] if self.weighted else None
        z = self.gae.encode(val_batch.x.float(), val_batch.edge_index, edge_weight)
        if self.weighted:
            recon_loss = self.mse_loss(
                z, val_batch.edge_index, edge_weight, val_batch.batch)
        else:
            recon_loss = self.gae.recon_loss(z, val_batch.edge_index)
        self.log('val_recon_loss', recon_loss)
        kl_loss = self.gae.kl_loss()
        self.log('val_kl_loss', kl_loss)
        loss = recon_loss + self.lam * kl_loss
        self.log('val_loss', loss)
        return loss

    def mse_loss(self, z, edge_index, edge_weight, batch):
        loss_fn = torch.nn.MSELoss()
        pos_pred_weights = self.gae.decoder(z, edge_index, sigmoid=False)
        pos_loss = loss_fn(pos_pred_weights, edge_weight)
        neg_edge_index = pg.utils.batched_negative_sampling(
            edge_index, batch, force_undirected=True)
        neg_pred_weights = self.gae.decoder(z, neg_edge_index, sigmoid=False)
        neg_loss = loss_fn(neg_pred_weights, torch.zeros(neg_pred_weights.shape[0]).cuda())
        return pos_loss + neg_loss
