import torch
from utils.functional import normalize
from utils.tools import get_npz_data, get_homophily
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS, WikipediaNetwork, WebKB, Actor, \
    AttributedGraphDataset, TUDataset, CitationFull, HeterophilousGraphDataset
from torch_geometric.utils import degree
import os
from utils.datasplit import get_split
import numpy as np
import pickle
import urllib.request

class Dataset:
    '''
    Dataset Class.
    This class loads, preprocesses and splits various datasets.

    Parameters
    ----------
    data : str
        The name of dataset.
    feat_norm : bool
        Whether to normalize the features.
    verbose : bool
        Whether to print statistics.
    n_splits : int
        Number of data splits.
    path : str
        Path to save dataset files.
    '''

    def __init__(self, data, feat_norm=True, adj_norm=True, verbose=True, path=None,
                 train_size=None, val_size=None, test_size=None,
                 train_percent=None, val_percent=None, test_percent=None,
                 train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None,
                 add_self_loop=True, split_type='default', from_npz=False, device='cuda:0'):
        self.name = data
        self.path = os.path.join(os.getcwd(), path)
        self.device = torch.device(device)
        self.single_graph = True
        self.self_loop = add_self_loop
        self.split_type = split_type

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.train_percent = train_percent
        self.val_percent = val_percent
        self.test_percent = test_percent
        self.train_examples_per_class = train_examples_per_class
        self.val_examples_per_class = val_examples_per_class
        self.test_examples_per_class = test_examples_per_class

        self.prepare_data(data, feat_norm, from_npz)
        self.feats = self.feats.to(torch.float)
        self.edge_index = self.adj.indices()
        if self.single_graph:
            self.split_data(verbose)
        else:
            self.split_graphs(verbose)
        # self.homophily = get_homophily(self.labels, self.adj.to_dense(), type='edge', fill=None)
        if add_self_loop:
            self.adj_with_loop = self.adj + torch.eye(self.adj.shape[0], device=self.adj.device).to_sparse()
            self.edge_index_loop = self.adj_with_loop.coalesce().indices()
        if adj_norm:
            self.normalized_adj = normalize(self.adj_with_loop, add_loop=False)
            self.normalized_adj = self.normalized_adj.coalesce()
            self.edge_index_normliazed = self.normalized_adj.coalesce().indices()


    def prepare_data(self, ds_name, feat_norm, from_npz):
        '''
        Function to Load various datasets.
        Homophilous datasets are loaded via pyg, while heterophilous datasets are loaded with `hetero_load`.
        The results are saved as `self.feats, self.adj, self.labels, self.train_masks, self.val_masks, self.test_masks`.
        Noth that `self.adj` is undirected and has no self loops.

        Parameters
        ----------
        ds_name : str
            The name of dataset.
        feat_norm : bool
            Whether to normalize the features.
        from_npz : bool
            Whether to load data from an existing npz file.

        '''

        if from_npz:
            adj, features, labels = get_npz_data(self.path + ds_name + '.npz', self_loop=self.self_loop)
            self.adj = adj.to(self.device).coalesce()
            self.feats = features.to(self.device)
            self.labels = torch.tensor(labels, dtype=torch.int64).to(self.device)
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.n_edges = self.adj.indices().shape[1] / 2
            self.n_classes = labels.max() + 1
            self.clean_label = self.labels.clone()
            if feat_norm:
                self.feats = normalize(self.feats, style='row')

        elif ds_name in ['cora', 'pubmed', 'citeseer', 'amazoncom', 'amazonpho', 'coauthorcs', 'coauthorph',
                         'blogcatalog', 'flickr', 'wikics', 'cornell', 'texas', 'wisconsin', 'dblp', 'amazon-ratings',
                         'roman-empire', 'actor', 'chameleon', 'squirrel']:
            self.data_raw = pyg_load_dataset(ds_name, path=self.path)
            self.g = self.data_raw[0]
            self.feats = self.g.x  # unnormalized
            if ds_name == 'flickr':
                self.feats = self.feats.to_dense()
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.labels = self.g.y
            self.adj = torch.sparse_coo_tensor(self.g.edge_index, torch.ones(self.g.edge_index.shape[1]),
                                               [self.n_nodes, self.n_nodes])
            self.n_edges = self.g.num_edges / 2
            self.n_classes = self.data_raw.num_classes

            self.feats = self.feats.to(self.device)
            self.labels = self.labels.to(self.device)


            self.adj = self.adj.to(self.device)
            # normalize features
            if feat_norm:
                self.feats = normalize(self.feats, style='row')

        self.adj = self.adj.coalesce()
        row = self.adj.indices()[0]
        d = degree(row, self.n_nodes)
        self.ave_degree = float(torch.mean(d))
        self.clean_label = self.labels.clone()

        print("""----Data statistics------'
                Name: %s
                #Nodes %d
                #Edges %d
                #Classes %d
                #Ave_degree %.2f""" %
              (self.name, self.n_nodes, self.n_edges, self.n_classes, self.ave_degree))

    def split_data(self, verbose=True):
        """
        数据集划分，分为三种方式：
        1.default（默认方式）：使用数据集自己提供的mask
        2.percent：按照指定的比例划分
        3.samples_per_class：指定每个类别的数量

        :param verbose:
        :return:
        """

        self.train_masks = None
        self.val_masks = None
        self.test_masks = None

        if self.split_type == 'default':
            train_type = 'default'
            val_type = 'default'
            test_type = 'default'
            # 1.规定数据集范围
            assert self.name in ['cora', 'citeseer', 'pubmed', 'blogcatalog', 'flickr', 'roman-empire',
                                 'amazon-ratings', 'amazoncom', 'chameleon', 'squirrel',
                                 'minesweeper', 'tolokers', 'questions', 'wikics', 'airport', 'actor', 'texas',
                                 'cornell', 'wisconsin',
                                 'brazil', 'europe', 'chameleon', 'squirrel',
                                 'facehook'], 'This dataset has no public splits.'
            # 2.引文网数据集（自带mask）
            if self.name in ['cora', 'citeseer', 'pubmed']:
                train_indices = torch.nonzero(self.g.train_mask, as_tuple=False).squeeze().numpy()
                val_indices = torch.nonzero(self.g.val_mask, as_tuple=False).squeeze().numpy()
                test_indices = torch.nonzero(self.g.test_mask, as_tuple=False).squeeze().numpy()
            # 3.blogcatalog', 'flickr', 'airport数据集
            elif self.name in ['blogcatalog', 'flickr', 'airport']:
                def load_obj(file_name):
                    with open(file_name, 'rb') as f:
                        return pickle.load(f)

                def download(name):
                    url = 'https://github.com/zhao-tong/GAug/raw/master/data/graphs/'
                    try:
                        print('Downloading', url + name)
                        urllib.request.urlretrieve(url + name, os.path.join(self.path, self.name, name))
                        print('Done!')
                    except:
                        raise Exception(
                            '''Download failed! Make sure you have stable Internet connection and enter the right name''')

                split_file = self.name + '_tvt_nids.pkl'
                if not os.path.exists(os.path.join(self.path, self.name, split_file)):
                    download(split_file)
                train_indices, val_indices, test_indices = load_obj(os.path.join(self.path, self.name, split_file))

            elif self.name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions', 'wikics', 'actor', 'texas', 'cornell', 'wisconsin', 'chameleon', 'squirrel']:
                train_indices = [torch.nonzero(x, as_tuple=False).squeeze().numpy() for x in self.g.train_mask.T][0]
                val_indices = [torch.nonzero(x, as_tuple=False).squeeze().numpy() for x in self.g.val_mask.T][0]
                test_indices = [torch.nonzero(x, as_tuple=False).squeeze().numpy() for x in self.g.test_mask.T][0]

        elif self.split_type == 'percent':
            if self.train_size is not None:
                train_size = self.train_size
                train_type = 'specified'
            elif self.train_percent is not None:
                train_size = int(self.n_nodes * self.train_percent)
                train_type = str(self.train_percent * 100) + ' % of nodes'
            else:
                print('Split error: split type = percent. Train size and train percent were not configured')
                exit(0)

            if self.val_size is not None:
                val_size = self.val_size
                val_type = 'specified'
            elif self.val_percent is not None:
                val_size = int(self.n_nodes * self.val_percent)
                val_type = str(self.val_percent * 100) + ' % of nodes'
            else:
                print('Split error: split type = percent. Val size and Val percent were not configured')
                exit(0)

            if self.test_size is not None:
                test_size = self.test_size
                test_type = 'specified'
            elif self.test_percent is not None:
                test_size = int(self.n_nodes * self.test_percent)
                test_type = str(self.test_percent * 100) + ' % of nodes'
            else:
                test_size = None
                test_type = 'remaining'
            train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(),
                                                                 train_size=train_size,
                                                                 val_size=val_size,
                                                                 test_size=test_size, )
        elif self.split_type == 'samples_per_class':
            train_size = None
            val_size = None
            test_size = None
            if self.train_examples_per_class is not None:
                train_examples_per_class = self.train_examples_per_class
                train_type = str(self.train_examples_per_class) + ' nodes per class'
            elif self.train_size is not None:
                train_examples_per_class = None
                train_size = self.train_size
                train_type = 'specified'
            else:
                print('Split error: split type = samples_per_class. Train size and train percent were not configured')
                exit(0)

            if self.val_examples_per_class is not None:
                val_examples_per_class = self.val_examples_per_class
                val_type = str(self.val_examples_per_class) + ' nodes per class'
            elif self.val_size is not None:
                val_examples_per_class = None
                val_size = self.val_size
                val_type = 'specified'
            else:
                print('Split error: split type = samples_per_class. Val size and val percent were not configured')
                exit(0)

            if self.test_examples_per_class is not None:
                test_examples_per_class = self.test_examples_per_class
                test_type = str(self.test_examples_per_class) + ' nodes per class'
            elif self.test_size is not None:
                test_examples_per_class = None
                test_size = self.test_size
                test_type = 'specified'
            else:
                test_examples_per_class = None
                test_size = None
                test_type = 'remaining'
            train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(),
                                                                 train_examples_per_class=train_examples_per_class,
                                                                 val_examples_per_class=val_examples_per_class,
                                                                 test_examples_per_class=test_examples_per_class,
                                                                 train_size=train_size,
                                                                 val_size=val_size,
                                                                 test_size=test_size)
        else:
            print('Split error: split type ' + self.split_type + ' not implemented')
            exit(0)



        # masks是ndarray型
        self.train_masks = train_indices
        self.val_masks = val_indices
        self.test_masks = test_indices

        all_idx = np.arange(self.n_nodes)
        used_idx = np.unique(np.concatenate([self.train_masks, self.val_masks, self.test_masks]))
        unlabel_idx = np.setdiff1d(all_idx, used_idx, assume_unique=True)

        # 保存 numpy 版与 tensor 版（放到 device）
        self.unlabel_masks = unlabel_idx  # numpy
        self.idx_unlabel = torch.from_numpy(unlabel_idx).long().to(self.device)  # torch

        # 可选：把 train/val/test 也同步成 tensor 版，后续更好用
        self.idx_train = torch.from_numpy(self.train_masks).long().to(self.device)
        self.idx_val = torch.from_numpy(self.val_masks).long().to(self.device)
        self.idx_test = torch.from_numpy(self.test_masks).long().to(self.device)

        # bool向量
        num_nodes = self.n_nodes  # 或者你已有的节点总数
        self.mask_train = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        self.mask_val = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        self.mask_test = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        self.mask_unlabel = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)

        self.mask_train[self.idx_train] = True
        self.mask_val[self.idx_val] = True
        self.mask_test[self.idx_test] = True
        self.mask_unlabel[self.idx_unlabel] = True

        if verbose:
            print("""----Split statistics------'
                #Train samples %d (%s)
                #Val samples %d (%s)
                #Test samples %d (%s)""" %
                  (len(self.train_masks), train_type,
                   len(self.val_masks), val_type,
                   len(self.test_masks), test_type))


def pyg_load_dataset(name, path='./data/'):
    dic = {'cora': 'Cora',
           'citeseer': 'CiteSeer',
           'pubmed': 'PubMed',
           'amazoncom': 'Computers',
           'amazonpho': 'Photo',
           'coauthorcs': 'CS',
           'coauthorph': 'Physics',
           'wikics': 'WikiCS',
           'chameleon': 'Chameleon',
           'squirrel': 'Squirrel',
           'cornell': 'Cornell',
           'texas': 'Texas',
           'wisconsin': 'Wisconsin',
           'actor': 'Actor',
           'blogcatalog': 'blogcatalog',
           'flickr': 'flickr',
           'amazon-ratings': 'Amazon-ratings',
           'roman-empire': 'Roman-empire'}
    if name in dic.keys():
        name = dic[name]
    else:
        name = name

    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=path, name=name)
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(root=path, name=name)
    elif name in ["CS", "Physics"]:
        dataset = Coauthor(root=path, name=name)
    elif name in ['WikiCS']:
        dataset = WikiCS(root=os.path.join(path, name))
    elif name in ['Chameleon', 'Squirrel', 'Crocodile']:
        dataset = WikipediaNetwork(root=path, name=name)
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root=path, name=name)
    elif name == 'Actor':
        dataset = Actor(root=os.path.join(path, name))
    elif name in ['blogcatalog', 'flickr']:
        dataset = AttributedGraphDataset(root=path, name=name)
    elif name in ['Amazon-ratings', 'Roman-empire']:
        dataset = HeterophilousGraphDataset(root=path, name=name)
    elif name in ['dblp']:
        dataset = CitationFull(root=path, name=name)
    else:
        dataset = TUDataset(root=path, name=name)
    return dataset


# """
# 下载数据集
# """
# dataset = Dataset(
#     data='squirrel',
#     feat_norm=True,
#     split_type='default',
#     train_percent=0.1,
#     val_percent=0.1,
#     test_percent=0.8,
#     path='data',
#     device='cuda:0'
# )
# print(dataset)
# print("ddas")
