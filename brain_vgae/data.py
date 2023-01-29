import shutil
import numpy as np
import networkx as nx
import os.path as osp
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from sklearn import preprocessing

import torch
from torch_geometric.data import InMemoryDataset, Data

DATA_HOME = osp.expanduser('~/netsci/dataset')

def process_graph(G):
    # Get the identities of non-isolated nodes
    G = nx.convert_node_labels_to_integers(G)
    G.remove_nodes_from(list(nx.isolates(G)))
    node_ids = torch.tensor(list(G.nodes))
    # Reassign the node labels to create appropriate edge_index
    G = nx.convert_node_labels_to_integers(G)
    # Use directed graph for PyTorch compatibility
    G = G.to_directed() if not nx.is_directed(G) else G
    # Create edge index tensor
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).T.contiguous()
    # Create features based on node identities
    node_feats = torch.nn.functional.one_hot(node_ids, 1015).float()

    def get_node_attr(name):
        attr_dict = nx.get_node_attributes(G, name)
        return [attr_dict[n] for n in G.nodes]
    def get_edge_attr(name):
        attr_dict = nx.get_edge_attributes(G, name)
        return [attr_dict[e] for e in G.edges]

    # NODE ATTRIBUTES
    # # position
    # x = get_node_attr('dn_position_x')
    # y = get_node_attr('dn_position_y')
    # z = get_node_attr('dn_position_z')
    # pos = torch.tensor([x, y, z]).T
    # # categorical variables
    # name = get_node_attr('dn_name')
    # fsname = get_node_attr('dn_fsname')
    # group = [n.split('_')[0] for n in fsname]
    # hemisphere = get_node_attr('dn_hemisphere')
    # region = get_node_attr('dn_region')
    # categorical_feats = np.stack(
    #     [name, fsname, group, hemisphere, region], -1)
    # ohe = preprocessing.OneHotEncoder(sparse=False)
    # data = ohe.fit_transform(categorical_feats)
    # node_feats = torch.tensor(data)
    # # EDGE ATTRIBUTES
    # node2ix = {n: i for i, n in enumerate(G.nodes)}
    # edge_ix = [[node2ix[n1], node2ix[n2]] for (n1, n2) in G.edges]
    # edge_index = torch.tensor(edge_ix, dtype=torch.long).T
    # flm = get_edge_attr('fiber_length_mean')
    # fm = get_edge_attr('FA_mean')
    nof = get_edge_attr('number_of_fibers')
    edge_feats = torch.tensor([nof]).T

    # # Use node identities as the features - they are the same across graphs
    # node_feats = torch.eye(len(G.nodes))
    data = Data(
        x=node_feats,
        edge_index=edge_index,
        edge_attr=edge_feats,
        # pos=pos,
    )
    return data

class BrainConnectivity(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pattern = osp.join(DATA_HOME, '*.graphml')
        filepaths = sorted(glob(pattern))
        filenames = [osp.basename(p) for p in filepaths]
        return filenames

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        # Copy the files from their home into the root directory
        for file in glob(osp.join(DATA_HOME, '*.graphml')):
            shutil.copy(file, self.raw_dir)
    
    def process(self):
        # Read the brain graphs into memory
        pattern = osp.join(self.raw_dir, '*.graphml')
        graph_filenames = sorted(glob(pattern))
        with Pool() as pool:
            print('Reading .graphml files...')
            graphs = pool.map(nx.read_graphml, graph_filenames)
            # Would be faster but doesn't work
            # data_list = pool.map(process_graph, graphs)
        print('Processing graphs into PyTorch Data...')
        data_list = [process_graph(g) for g in tqdm(graphs)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        
if __name__ == '__main__':
    # "Downloads" and preprocesses the data
    dataset = BrainConnectivity('~/tmp/brain_graphs_weighted_LCC')
