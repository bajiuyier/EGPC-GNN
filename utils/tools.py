import argparse
import os
import ruamel.yaml as yaml
import warnings
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

def classification_metrics_from_logits(logits, labels, num_classes):
    """
    Compute node classification metrics from logits and ground-truth labels.

    Returns:
    1. accuracy: standard accuracy;
    2. balanced_accuracy: balanced accuracy;
    3. macro_f1: macro-averaged F1;
    4. roc_auc: multi-class ROC-AUC.

    Note:
    If a split does not contain all classes, ROC-AUC may not be computable.
    In this case, return nan instead of interrupting the program.
    """

    if labels.numel() == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "roc_auc": float("nan")
        }

    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    acc = accuracy_score(y_true, preds)

    bacc = balanced_accuracy_score(y_true, preds)

    macro_f1 = f1_score(
        y_true,
        preds,
        labels=np.arange(num_classes),
        average="macro",
        zero_division=0
    )

    try:
        if num_classes == 2:
            roc_auc = roc_auc_score(y_true, probs[:, 1])
        else:
            roc_auc = roc_auc_score(
                y_true,
                probs,
                multi_class="ovr",
                labels=np.arange(num_classes)
            )
    except Exception:
        roc_auc = float("nan")

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc)
    }

def load_conf(path: str = None, method: str = None, dataset: str = None):
    '''
    Function to load config file.

    Parameters
    ----------
    path : str
        Path to load config file. Load default configuration if set to `None`.
    method : str
        Name of the used mathod. Necessary if ``path`` is set to `None`.
    dataset : str
        Name of the corresponding dataset. Necessary if ``path`` is set to `None`.

    Returns
    -------
    conf : argparse.Namespace
        The config file converted to Namespace.

    '''
    if path == None and method == None:
        raise KeyError
    if path == None and dataset == None:
        raise KeyError
    if path == None:
        dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        path = os.path.join(dir, method, method + '_' + dataset + ".yaml")
        if os.path.exists(path) == False:
            raise KeyError("The method configuration file is not provided.")

    conf = open(path, "r", encoding="utf-8").read()
    conf = yaml.load(conf)
    conf = argparse.Namespace(**conf)
    return conf


def save_conf(path: str = None, method: str = None, dataset: str = None, conf: any = None):
    '''
    Function to load config file.

    Parameters
    ----------
    path : str
        Path to load config file. Load default configuration if set to `None`.
    method : str
        Name of the used mathod. Necessary if ``path`` is set to `None`.
    dataset : str
        Name of the corresponding dataset. Necessary if ``path`` is set to `None`.
    conf : argparse.Namespace
        The config file to save.

    Returns
    -------
        None
    '''

    if path == None and method == None:
        raise KeyError
    if path == None and dataset == None:
        raise KeyError
    if path == None:
        dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        path = os.path.join(dir, method, method + '_' + dataset + ".yaml")

    with open(path, 'w') as f:
        yaml.safe_dump(conf, f, default_flow_style=False)
    print('config file ' + path + ' updated')
    return None

def get_npz_data(file_name, self_loop):
    adj, features, labels = load_npz(file_name)
    adj = adj + adj.T
    adj = adj.tolil()
    adj[adj > 1] = 1
    lcc = largest_connected_components(adj)
    adj = adj[lcc][:, lcc]
    if not self_loop:
        adj.setdiag(0)
    else:
        adj.setdiag(1)
    features = features[lcc]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    labels = labels[lcc]
    return adj, features, labels


def load_npz(file_name, is_sparse=True):
    with np.load(file_name) as loader:
        if is_sparse:
            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                 loader['adj_indptr']), shape=loader['adj_shape'])
            if 'attr_data' in loader:
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                          loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                features = None
            labels = loader.get('labels')
        else:
            adj = loader['adj_data']
            if 'attr_data' in loader:
                features = loader['attr_data']
            else:
                features = None
            labels = loader.get('labels')
    if features is None:
        features = np.eye(adj.shape[0])
    features = sp.csr_matrix(features, dtype=np.float32)
    return adj, features, labels


def largest_connected_components(adj, n_components=1):
    """Select k largest connected components.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        input adjacency matrix
    n_components : int
        n largest connected components we want to select
    """

    _, component_indices = sp.csgraph.connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def scipy_sparse_to_sparse_tensor(sparse_mx):
    '''
    Convert a scipy sparse matrix to a torch sparse tensor.

    Parameters
    ----------
    sparse_mx : scipy.sparse_matrix
        Sparse matrix to convert.

    Returns
    -------
    sparse_tensor: torch.Tensor in sparse form
        A tensor stored in sparse form.
    '''
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_node_homophily(label, adj):
    '''
    Calculate the node homophily of a graph.

    Parameters
    ----------
    label : torch.tensor
        The ground truth labels.
    adj : torch.tensor
        The adjacency matrix in dense form.

    Returns
    -------
    homophily : torch.float
        The node homophily of the graph.

    '''
    label = label.cpu().numpy()
    adj = adj.cpu().numpy()
    num_node = len(label)
    label = label.repeat(num_node).reshape(num_node, -1)
    n = (np.multiply((label == label.T), adj)).sum(axis=1)
    d = adj.sum(axis=1)
    homos = []
    for i in range(num_node):
        if d[i] > 0:
            homos.append(n[i] * 1. / d[i])
    return np.mean(homos)


def get_edge_homophily(label, adj):
    '''
    Calculate the node homophily of a graph.

    Parameters
    ----------
    label : torch.tensor
        The ground truth labels.
    adj : torch.tensor
        The adjacency matrix in dense form.

    Returns
    -------
    homophily : torch.float
        The edge homophily of the graph.

    '''
    num_edge = adj.sum()
    cnt = 0
    for i, j in adj.nonzero():
        if label[i] == label[j]:
            cnt += adj[i, j]
    return cnt/num_edge


def get_homophily(label, adj, type='node', fill=None):
    '''
    Calculate node or edge homophily of a graph.

    Parameters
    ----------
    label : torch.tensor
        The ground truth labels.
    adj : torch.tensor
        The adjacency matrix in dense form.
    type : str
        This decides whether to calculate node homo or edge homo.
    fill : str
        The value to fill in the diagonal of `adj`. If set to `None`, the operation won't be done.

    Returns
    -------
    homophily : np.float
        The node or edge homophily of a graph.

    '''
    if fill:
        np.fill_diagonal(adj, fill)
    return eval('get_'+type+'_homophily(label, adj)')



def setup_seed(seed):

    import os
    import random
    import numpy as np
    import torch

    # Python hash randomness
    os.environ["PYTHONHASHSEED"] = str(seed)

    # CUDA deterministic configuration
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enforce PyTorch determinism
    torch.use_deterministic_algorithms(True)


def get_neighbors(adj, mask):
    edge_index = adj.indices()
    row, col = edge_index[0, :].cpu().numpy(), edge_index[1, :].cpu().numpy()
    col_mask = np.in1d(row, mask)
    masked_col = col[col_mask]
    neighbors = np.unique(masked_col)
    return neighbors


def heatmap(matrix, title, n_classes):

    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    '''
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, round(matrix[i, j], 2), ha="center", va="center", color="w", fontsize=5)
    '''

    ax.set_title(title, fontdict={'weight': 'normal', 'size': 25})
    fig.tight_layout()
    plt.show()
    fig.savefig('./data_eval/' + title + '.png', format='png')

# ============================================================
# General utility functions used by EGPC-GNN
# ============================================================

def get_conf_value(conf_part, key, default=None):
    """
    Read a parameter from a configuration object or dictionary.

    Supports two formats:
    1. conf.loss['lambda_cls']
    2. conf.loss.lambda_cls
    """
    if conf_part is None:
        return default

    if isinstance(conf_part, dict):
        return conf_part[key] if key in conf_part else default

    if hasattr(conf_part, key):
        return getattr(conf_part, key)

    try:
        return conf_part[key]
    except Exception:
        return default


def to_bool_mask(mask_or_index, num_nodes, device):
    """
    Convert indices or a mask into a boolean mask.

    Supported inputs:
    1. Boolean tensor with shape [N];
    2. 0/1 tensor with shape [N];
    3. numpy index；
    4. torch index；
    5. list index。
    """
    if torch.is_tensor(mask_or_index):
        mask = mask_or_index.to(device)
    else:
        mask = torch.as_tensor(mask_or_index, device=device)

    if mask.dtype == torch.bool and mask.numel() == num_nodes:
        return mask

    if mask.numel() == num_nodes and mask.dtype != torch.bool:
        unique_values = torch.unique(mask)
        if torch.all((unique_values == 0) | (unique_values == 1)):
            return mask.bool()

    bool_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    bool_mask[mask.long()] = True

    return bool_mask


def mask_to_index(mask_or_index, num_nodes, device):
    """
    Convert a boolean mask or indices into LongTensor indices.
    """
    mask = to_bool_mask(mask_or_index, num_nodes, device)
    return mask.nonzero(as_tuple=False).view(-1)


def l2_normalize(x, eps=1e-12):
    """
    Apply L2 normalization to feature vectors.
    """
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)


def class_balanced_ce_loss(logits, labels, train_mask, num_classes):
    """
    Class-balanced cross-entropy loss.

    First compute the average loss within each class, then average across classes,
    reducing the dominance of majority classes in the training objective.
    """
    device = logits.device
    num_nodes = logits.size(0)

    labels = labels.to(device).long()
    train_idx = mask_to_index(train_mask, num_nodes, device)

    if train_idx.numel() == 0:
        return logits.new_tensor(0.0)

    per_node_loss = F.cross_entropy(
        logits[train_idx],
        labels[train_idx],
        reduction='none'
    )

    train_y = labels[train_idx]
    losses = []

    for cls in range(num_classes):
        cls_mask = train_y == cls

        if cls_mask.any():
            losses.append(per_node_loss[cls_mask].mean())

    if len(losses) == 0:
        return logits.new_tensor(0.0)

    return torch.stack(losses).mean()


def compute_class_prototypes(h, labels, train_mask, num_classes, normalize=True):
    """
    Compute class prototypes from training nodes.

    If a class has no nodes in the training set, use the global mean of training nodes as a fallback.
    """
    device = h.device
    num_nodes = h.size(0)
    feat_dim = h.size(1)

    labels = labels.to(device).long()
    train_idx = mask_to_index(train_mask, num_nodes, device)

    if train_idx.numel() > 0:
        global_proto = h[train_idx].mean(dim=0)
    else:
        global_proto = h.mean(dim=0)

    prototypes = []

    for cls in range(num_classes):
        cls_idx = train_idx[labels[train_idx] == cls]

        if cls_idx.numel() == 0:
            proto = global_proto
        else:
            proto = h[cls_idx].mean(dim=0)

        prototypes.append(proto.view(1, feat_dim))

    prototypes = torch.cat(prototypes, dim=0)

    if normalize:
        prototypes = l2_normalize(prototypes)

    return prototypes


def prototype_logits(h, prototypes, normalize=True):
    """
    Compute logits from node representations and class prototypes.

    By default, use the normalized dot product, which is equivalent to cosine similarity.
    """
    if normalize:
        h = l2_normalize(h)
        prototypes = l2_normalize(prototypes)

    return torch.mm(h, prototypes.t())


def entropy_reliability(q, eps=1e-12):
    """
    Compute node self-evidence reliability from the entropy of the class-evidence distribution.

    The more concentrated the distribution, the lower the entropy and the higher the reliability.
    """
    num_classes = q.size(1)
    log_c = float(np.log(max(num_classes, 2)))

    entropy = -(q * q.clamp(min=eps).log()).sum(dim=1)
    reliability = 1.0 - entropy / log_c

    return reliability.clamp(min=0.0, max=1.0)


def js_divergence(p, q, eps=1e-12):
    """
    Compute the JS divergence between two class-evidence distributions.
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)

    m = 0.5 * (p + q)

    kl_pm = (p * (p.log() - m.clamp(min=eps).log())).sum(dim=1)
    kl_qm = (q * (q.log() - m.clamp(min=eps).log())).sum(dim=1)

    js = 0.5 * kl_pm + 0.5 * kl_qm

    return js / float(np.log(2.0))


def build_reference_indices(q_self, labels, train_mask):
    """
    Construct the reference class for each node.

    Training nodes use ground-truth labels, while other nodes use classes predicted from self-evidence.
    """
    device = q_self.device
    num_nodes = q_self.size(0)

    labels = labels.to(device).long()
    train_mask = to_bool_mask(train_mask, num_nodes, device)

    ref_class = q_self.argmax(dim=1)
    ref_class[train_mask] = labels[train_mask]

    return ref_class.long()


def margin_drop_score(logits_self, logits_prop, ref_class):
    """
    Compute the degradation of class-evidence margins before and after propagation.
    """
    num_nodes, num_classes = logits_self.size()
    device = logits_self.device

    gather_index = ref_class.view(-1, 1)

    self_ref = logits_self.gather(1, gather_index)
    prop_ref = logits_prop.gather(1, gather_index)

    margin_self = self_ref - logits_self
    margin_prop = prop_ref - logits_prop

    drop = (margin_self - margin_prop).clamp(min=0.0)

    mask = torch.ones((num_nodes, num_classes), dtype=torch.bool, device=device)
    mask.scatter_(1, gather_index, False)

    drop = drop.masked_fill(~mask, 0.0)

    return drop.sum(dim=1) / max(num_classes - 1, 1)


def classwise_standardize_scores(score, ref_class, num_classes, eps=1e-12):
    """
    Perform class-wise standardization on node scores.

    Avoid inconsistent score scales across classes caused by differences in sample size or distribution.
    """
    out = torch.zeros_like(score)

    for cls in range(num_classes):
        idx = ref_class == cls

        if idx.any():
            s = score[idx]
            mean = s.mean()
            std = s.std(unbiased=False).clamp(min=eps)
            out[idx] = (s - mean) / std

    return out


def compute_completion_weight(
    q_self,
    q_prop,
    logits_self,
    logits_prop,
    labels,
    train_mask,
    num_classes,
    gamma=2.0,
    bias=0.0,
    max_w=1.0,
    stop_gradient=True,
    eps=1e-12,
):
    """
    Compute node-level evidence completion weights.

    A larger weight indicates a higher likelihood of propagation-induced class-evidence degradation.
    """
    ref_class = build_reference_indices(
        q_self=q_self,
        labels=labels,
        train_mask=train_mask
    )

    rel_self = entropy_reliability(q_self, eps=eps)
    js_score = js_divergence(q_self, q_prop, eps=eps)
    mar_score = margin_drop_score(logits_self, logits_prop, ref_class)

    raw_score = rel_self * (js_score + mar_score)

    norm_score = classwise_standardize_scores(
        score=raw_score,
        ref_class=ref_class,
        num_classes=num_classes,
        eps=eps
    )

    w = torch.sigmoid(gamma * norm_score + bias)

    if max_w < 1.0:
        w = w * max_w

    if stop_gradient:
        w = w.detach()

    aux = {
        'ref_class': ref_class,
        'rel_self': rel_self,
        'js_score': js_score,
        'mar_score': mar_score,
        'raw_score': raw_score,
        'norm_score': norm_score
    }

    return w, aux


def build_reference_prototype(q_self, labels, train_mask, proto_self, use_soft_for_unlabeled=True):
    """
    Construct a reference class prototype for each node.

    Training nodes use the prototype of their ground-truth class;
    Non-training nodes use a soft reference prototype weighted by q_self by default.
    """
    device = q_self.device
    num_nodes = q_self.size(0)

    labels = labels.to(device).long()
    train_mask = to_bool_mask(train_mask, num_nodes, device)

    if use_soft_for_unlabeled:
        ref_proto = torch.mm(q_self, proto_self)
    else:
        pred = q_self.argmax(dim=1)
        ref_proto = proto_self[pred]

    ref_proto[train_mask] = proto_self[labels[train_mask]]

    return ref_proto


def margin_completion_loss(
    h_prop,
    anchor,
    proto_prop,
    labels,
    train_mask,
    w,
    num_classes,
    delta=0.0,
    eps=1e-12,
):
    """
    Margin-based completion consistency loss.

    Constrain the class margin of the completion anchor to be no weaker than that of the original propagation representation.
    """
    device = h_prop.device
    num_nodes = h_prop.size(0)

    labels = labels.to(device).long()
    train_idx = mask_to_index(train_mask, num_nodes, device)

    if train_idx.numel() == 0:
        return h_prop.new_tensor(0.0)

    logits_prop = prototype_logits(h_prop, proto_prop, normalize=True)
    logits_anchor = prototype_logits(anchor, proto_prop, normalize=True)

    y = labels[train_idx]
    gather_index = y.view(-1, 1)

    prop_ref = logits_prop[train_idx].gather(1, gather_index)
    anchor_ref = logits_anchor[train_idx].gather(1, gather_index)

    margin_prop = prop_ref - logits_prop[train_idx]
    margin_anchor = anchor_ref - logits_anchor[train_idx]

    violation = (margin_prop + delta - margin_anchor).clamp(min=0.0)

    mask = torch.ones_like(violation, dtype=torch.bool)
    mask.scatter_(1, gather_index, False)

    violation = violation.masked_fill(~mask, 0.0)

    per_node = violation.sum(dim=1) / max(num_classes - 1, 1)

    weight = w[train_idx].detach()
    loss = (per_node * weight).sum() / (weight.sum() + eps)

    return loss
