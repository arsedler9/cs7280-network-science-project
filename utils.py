from functools import lru_cache
from os import path
from glob import glob
from multiprocessing import Pool
import networkx as nx

DATA_HOME = path.expanduser('~/netsci/dataset')

@lru_cache(maxsize=5)
def get_brain_graphs(n_graphs):
    pattern = path.join(DATA_HOME, '*.graphml')
    graph_filenames = sorted(glob(pattern))[:n_graphs]
    with Pool() as pool:
        graphs = pool.map(nx.read_graphml, graph_filenames)
    return graphs

