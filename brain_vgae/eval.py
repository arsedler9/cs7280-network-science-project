import torch_geometric as pg

def evaluate(model, dataset):
    all_auc, all_ap = [], []
    for data in dataset:
        neg_edge_index = pg.utils.negative_sampling(
            data.edge_index, num_nodes=1015, force_undirected=True)
        z = model.forward(data)
        auc, ap = model.gae.test(z, data.edge_index, neg_edge_index)
        all_auc.append(auc)
        all_ap.append(ap)
    return all_auc, all_ap

def get_graph_embeds(model, dataset):
    all_embeds = []
    for data in dataset:
        z = model.forward(data).detach().numpy()
        zg = np.concatenate([
            np.max(z, axis=0),
            np.sum(z, axis=0),
            np.mean(z, axis=0),
        ])
        # zg = np.sum(z, axis=0)
        all_embeds.append(zg)
    return np.stack(all_embeds)

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from scipy.spatial.distance import cdist

    from data import BrainConnectivity
    from models import BrainGAE, BrainVGAE

    # Load and split the dataset
    dataset = BrainConnectivity('~/tmp/brain_graphs')
    train_data, val_data = dataset[:850], dataset[850:]

    # # Restore an existing model
    # MODEL_PATH = '/snel/home/asedler/netsci/models/gae/gae-epoch=01-val_loss=0.83.ckpt'
    # model = BrainGAE.load_from_checkpoint(MODEL_PATH)

    # # Too high KL penalty
    # MODEL_PATH = '/snel/home/asedler/netsci/models/vgae/lightning_logs/version_7/checkpoints/epoch=6.ckpt'
    # model = BrainVGAE.load_from_checkpoint(MODEL_PATH)

    # # Reasonable KL penalty
    # MODEL_PATH = '/snel/home/asedler/netsci/models/vgae/lightning_logs/version_8/checkpoints/epoch=12.ckpt'
    # model = BrainVGAE.load_from_checkpoint(MODEL_PATH)

    # # "Improved" GCN layers
    # MODEL_PATH = '/snel/home/asedler/netsci/models/vgae/lightning_logs/version_9/checkpoints/epoch=13.ckpt'
    # model = BrainVGAE().load_from_checkpoint(MODEL_PATH)

    # Add weights and MSE loss
    MODEL_PATH = '/snel/home/asedler/netsci/models/vgae_wt_lcc/lightning_logs/version_0/checkpoints/epoch=15.ckpt'
    model = BrainVGAE.load_from_checkpoint(MODEL_PATH)

    # # Well-trained
    # MODEL_PATH = '/snel/home/asedler/netsci/models/vgae_wt_lcc/lightning_logs/version_3/checkpoints/epoch=554.ckpt'
    # model = BrainVGAE.load_from_checkpoint(MODEL_PATH)

    # # Plot example node and graph embeddings in 2D
    # train_z = model.forward(train_data[0]).detach().numpy()
    # train_zg = np.mean(train_z, axis=0, keepdims=True)
    # valid_z = model.forward(val_data[0]).detach().numpy()
    # valid_zg = np.mean(valid_z, axis=0, keepdims=True)
    # reducer = PCA(2).fit(train_z)
    # lowd_train_z = reducer.transform(train_z)
    # lowd_valid_z = reducer.transform(valid_z)
    # lowd_train_zg = reducer.transform(train_zg)
    # lowd_valid_zg = reducer.transform(valid_zg)


    

    # fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 6))
    # ax1.scatter(*lowd_train_z.T, alpha=0.3, s=5, label='Node embeddings')
    # ax1.scatter(*lowd_train_zg.T, label='Graph embedding')
    # ax1.set_title('Training graph')
    # ax2.scatter(*lowd_valid_z.T, alpha=0.3, s=5, label='Node embeddings')
    # ax2.scatter(*lowd_train_zg.T, label='Graph embedding')
    # ax2.set_title('Validation graph')
    # ax2.legend()

    # # Plot all graph embeddings in 2D space
    # train_zgs = get_graph_embeds(model, train_data)
    # valid_zgs = get_graph_embeds(model, val_data)
    # # Compute distances to find the most similar and most different graphs
    # dists = cdist(train_zgs, train_zgs)
    # np.fill_diagonal(dists, np.inf)
    # near_inds = np.unravel_index(np.argmin(dists), dists.shape)
    # np.fill_diagonal(dists, -np.inf)
    # far_inds = np.unravel_index(np.argmax(dists), dists.shape)
    # print('Most similar graphs: {}, {}'.format(*near_inds))
    # print('Most different graphs: {}, {}'.format(*far_inds))
    # lowd_zgs = TSNE(2, random_state=0).fit_transform(train_zgs)
    # # lowd_train_zgs = lowd_zgs[:len(train_zgs)]
    # # lowd_valid_zgs = lowd_zgs[len(train_zgs):]
    # # fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 6))
    # # ax1.scatter(*lowd_train_zgs.T, alpha=0.3, s=5)
    # # ax1.set_title('Training whole-graph embeddings')
    # # ax2.scatter(*lowd_valid_zgs.T, alpha=0.3, s=5)
    # # ax2.set_title('Validation whole-graph embeddings')
    # # plt.show()

    # fig, ax = plt.subplots(figsize=(4,4))
    # plt.scatter(*lowd_zgs.T, alpha=0.3, s=5)
    # plt.axis('off')
    # import pdb; pdb.set_trace()

    # Run evaluation on training and validation subsets
    train_auc, train_ap = evaluate(model, train_data)
    valid_auc, valid_ap = evaluate(model, val_data)
    print(f"Train AUC: {np.mean(train_auc):.4} +/- {np.std(train_auc):.4}")
    print(f"Valid AUC: {np.mean(valid_auc):.4} +/- {np.std(valid_auc):.4}")
    print(f"Train AP: {np.mean(train_ap):.4} +/- {np.std(train_ap):.4}")
    print(f"Valid AP: {np.mean(valid_ap):.4} +/- {np.std(valid_ap):.4}")
