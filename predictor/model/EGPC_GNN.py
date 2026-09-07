import torch
import torch.nn as nn
import torch.nn.functional as F

from predictor.model.Base_GNN import *

from utils.tools import (
    get_conf_value,
    to_bool_mask,
    compute_class_prototypes,
    prototype_logits,
    class_balanced_ce_loss,
)


class SelfEvidenceEncoder(nn.Module):
    """
    Node self-evidence encoder.

    This branch only uses node attributes without graph structure
    to obtain h_self.
    """

    def __init__(self, conf):
        super().__init__()

        in_dim = int(conf.model['n_feat'])
        hidden_dim = int(conf.model['n_hidden'])
        dropout = float(conf.model['dropout'])

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize MLP parameters.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.encoder(x.float())


class PropagationEvidenceEncoder(nn.Module):
    """
    Graph propagation-evidence encoder.

    This module reuses the GCN defined in Base_GNN.py and sets
    its output dimension to the hidden dimension to obtain h_prop.
    """

    def __init__(self, conf):
        super().__init__()

        self.gcn = GCN(
            nfeat=conf.model['n_feat'],
            nhid=conf.model['n_hidden'],
            nclass=conf.model['n_hidden'],
            n_layers=conf.model['n_layer'],
            dropout=conf.model['dropout'],
            act=conf.model['act'],
            input_layer=conf.model['input_layer'],
            output_layer=False
        )

    def forward(self, x, adj):
        return self.gcn(x.float(), adj)


class AnchorMapper(nn.Module):
    """
    Completion anchor mapper.

    The inputs are the node self-evidence representation h_self
    and the reference prototype r_i.
    The output is the completion anchor u_i in the propagation space.
    """

    def __init__(self, conf):
        super().__init__()

        hidden_dim = int(conf.model['n_hidden'])
        dropout = float(conf.model['dropout'])

        self.mapper = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize mapper parameters.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, h_self, ref_proto_self):
        return self.mapper(torch.cat([h_self, ref_proto_self], dim=-1))


def _get_loss_conf(conf, key, default=None):
    """
    Read a parameter from conf.loss. If the parameter does not exist,
    use the default value.

    Functions:
    - Uniformly read hyperparameters from the loss section, such as
      lambda_e, lambda_c, egpc_max_w, and warmup.
    - If a field is missing from the configuration file, return default
      to avoid errors caused by missing fields.

    Parameter correspondence:
    - lambda_cls corresponds to lambda_cls in the total loss.
    - lambda_e corresponds to lambda_e in the total loss.
    - lambda_c corresponds to lambda_c in the total loss.
    - egpc_max_w corresponds to the maximum completion strength eta
      in the completion-weight formula.
    - warmup corresponds to the number of epochs T_w during which
      completion is disabled at the beginning of training.
    """
    return get_conf_value(getattr(conf, 'loss', None), key, default)


def mask_to_index(mask_or_index, num_nodes, device):
    """
    Convert a boolean mask or indices into LongTensor indices.
    """
    mask = to_bool_mask(mask_or_index, num_nodes, device)
    return mask.nonzero(as_tuple=False).view(-1)


def supervised_ce_loss(logits, labels, train_mask):
    """
    Standard supervised cross-entropy loss corresponding to
    -y log q or -y log p.

    Formula correspondence:
    - When logits come from the final classification head,
      this corresponds to L_cls = -sum y_ic log p_i(c).
    - When logits come from the evidence branches,
      this corresponds to -sum y_ic log q_i(c) in L_evi.
    """
    device = logits.device
    num_nodes = logits.size(0)

    labels = labels.to(device).long()
    train_idx = mask_to_index(train_mask, num_nodes, device)

    if train_idx.numel() == 0:
        return logits.new_tensor(0.0)

    return F.cross_entropy(logits[train_idx], labels[train_idx])


def supervised_loss(logits, labels, train_mask, num_classes, balanced=True):
    """
    Supervised loss.

    When balanced=True, class-balanced cross-entropy is used:
    first compute the average loss within each class, and then
    average across classes.
    This form is recommended for class-imbalanced node classification.
    """
    if balanced:
        return class_balanced_ce_loss(
            logits=logits,
            labels=labels,
            train_mask=train_mask,
            num_classes=num_classes
        )

    return supervised_ce_loss(
        logits=logits,
        labels=labels,
        train_mask=train_mask
    )


def entropy_reliability(q, eps=1e-12):
    """
    Compute node self-evidence reliability r_self according to
    the entropy of the self-evidence distribution.

    Formula correspondence:
    - Input q corresponds to q_i^{self}.
    - entropy corresponds to
      H(q_i^{self}) = -sum_c q_i^{self}(c) log q_i^{self}(c).
    - reliability corresponds to
      r_i^{self}=1-H(q_i^{self})/log C.

    Interpretation:
    - The more concentrated q_i^{self} is, the smaller the entropy
      and the larger r_i^{self}, indicating clearer evidence from
      the node's own attributes.
    - The more uniform q_i^{self} is, the larger the entropy
      and the smaller r_i^{self}, indicating more ambiguous evidence
      from the node's own attributes.

    Role of eps:
    - Avoid numerical issues caused by log(0).
    """
    num_classes = q.size(1)
    log_c = torch.log(q.new_tensor(float(max(num_classes, 2))))

    entropy = -(q * q.clamp(min=eps).log()).sum(dim=1)
    reliability = 1.0 - entropy / log_c

    return reliability.clamp(min=0.0, max=1.0)


def build_reference_class(q_self, labels, train_mask):
    """
    Construct the reference class a_i.

    Formula correspondence:
    - Labeled nodes: a_i = y_i.
    - Unlabeled nodes: a_i = argmax_c q_i^{self}(c).

    Why q_self is used instead of q_prop:
    - This work focuses on whether graph propagation weakens
      the node's original evidence.
    - Therefore, the reference class should come from the node's
      self-evidence rather than evidence already affected by
      neighborhood propagation.
    """
    device = q_self.device
    num_nodes = q_self.size(0)

    labels = labels.to(device).long()
    train_mask = to_bool_mask(train_mask, num_nodes, device)

    ref_class = q_self.argmax(dim=1).long()
    ref_class[train_mask] = labels[train_mask]

    return ref_class


def margin_drop_score(logits_self, logits_prop, ref_class):
    """
    Compute the propagation-induced margin degradation score d_mar.

    Formula correspondence:
    - logits_self[:, c] corresponds to z_i^{self}(c).
    - logits_prop[:, c] corresponds to z_i^{prop}(c).
    - ref_class corresponds to the reference class a_i.
    - margin_self corresponds to
      m_i^{self}(c)=z_i^{self}(a_i)-z_i^{self}(c).
    - margin_prop corresponds to
      m_i^{prop}(c)=z_i^{prop}(a_i)-z_i^{prop}(c).
    - drop corresponds to
      [m_i^{self}(c)-m_i^{prop}(c)]_+.
    - The return value corresponds to
      d_i^{mar}=1/(C-1) sum_{c!=a_i}
      [m_i^{self}(c)-m_i^{prop}(c)]_+.

    Interpretation:
    - If the propagation margin is smaller than the self-evidence margin,
      propagation has weakened the node's relative support for the
      reference class.
    - Only the positive part [.]_+ is retained, meaning that only
      margin degradation is counted, while margin improvement after
      propagation is not penalized.
    """
    num_nodes, num_classes = logits_self.size()
    device = logits_self.device

    ref_class = ref_class.to(device).long()
    gather_index = ref_class.view(-1, 1)

    self_ref = logits_self.gather(1, gather_index)
    prop_ref = logits_prop.gather(1, gather_index)

    margin_self = self_ref - logits_self
    margin_prop = prop_ref - logits_prop

    drop = (margin_self - margin_prop).clamp(min=0.0)

    # The reference class itself is excluded from the average over competing classes.
    mask = torch.ones((num_nodes, num_classes), dtype=torch.bool, device=device)
    mask.scatter_(1, gather_index, False)
    drop = drop.masked_fill(~mask, 0.0)

    return drop.sum(dim=1) / max(num_classes - 1, 1)


def classwise_standardize_stable(
    score,
    ref_class,
    labels,
    train_mask,
    num_classes,
    standardize_with_unlabeled=True,
    min_labeled_for_class_stat=3,
    min_std=1e-3,
    clip_s_hat=10.0,
):
    """
    Stable class-wise standardization.

    1. Prefer labeled nodes to compute the within-class mean and standard deviation;
    2. When a class contains too few labeled nodes, optionally fall back to all
       nodes whose reference class is that class;
    3. Use min_std as the lower bound of the standard deviation instead of eps
       to prevent s_hat from exploding when the tail-class standard deviation is zero;
    4. Finally clip s_hat.
    """
    device = score.device
    num_nodes = score.size(0)

    labels = labels.to(device).long()
    train_mask = to_bool_mask(train_mask, num_nodes, device)
    ref_class = ref_class.to(device).long()

    out = torch.zeros_like(score)

    for cls in range(num_classes):
        target_idx = ref_class == cls
        stat_idx = train_mask & (labels == cls)

        if not target_idx.any():
            continue

        # Normal case: use training nodes when enough labeled nodes are available.
        if stat_idx.sum().item() >= min_labeled_for_class_stat:
            stat_score = score[stat_idx]

        # Extreme long-tail case: when too few training nodes are available,
        # allow all nodes with the same reference class to assist the statistics.
        elif standardize_with_unlabeled and target_idx.sum().item() >= min_labeled_for_class_stat:
            stat_score = score[target_idx]

        # Final fallback: use the available training nodes or target nodes,
        # together with min_std to prevent numerical explosion.
        elif stat_idx.any():
            stat_score = score[stat_idx]
        else:
            stat_score = score[target_idx]

        mean = stat_score.mean()
        std = stat_score.std(unbiased=False).clamp(min=min_std)

        out[target_idx] = (score[target_idx] - mean) / std

    if clip_s_hat is not None and clip_s_hat > 0:
        out = out.clamp(min=-clip_s_hat, max=clip_s_hat)

    return out


def compute_margin_completion_weight(
    q_self,
    logits_self,
    logits_prop,
    labels,
    train_mask,
    num_classes,
    max_w=0.05,
    stop_gradient=True,
    standardize_with_unlabeled=True,
    min_labeled_for_class_stat=3,
    min_std=1e-3,
    clip_s_hat=10.0,
    eps=1e-12,
):
    """
    Compute the completion weight w_i based on propagation-induced margin degradation.

    s_raw = r_self * d_mar
    s_hat = stable_classwise_standardize(s_raw)
    w_base = [s_hat]_+ / (1 + [s_hat]_+)
    w = max_w * w_base

    Here, max_w limits the maximum completion strength to prevent
    the anchor from excessively replacing h_prop.
    """
    ref_class = build_reference_class(
        q_self=q_self,
        labels=labels,
        train_mask=train_mask
    )

    r_self = entropy_reliability(q_self, eps=eps)

    d_mar = margin_drop_score(
        logits_self=logits_self,
        logits_prop=logits_prop,
        ref_class=ref_class
    )

    raw_score = r_self * d_mar

    norm_score = classwise_standardize_stable(
        score=raw_score,
        ref_class=ref_class,
        labels=labels,
        train_mask=train_mask,
        num_classes=num_classes,
        standardize_with_unlabeled=standardize_with_unlabeled,
        min_labeled_for_class_stat=min_labeled_for_class_stat,
        min_std=min_std,
        clip_s_hat=clip_s_hat,
    )

    positive_score = norm_score.clamp(min=0.0)
    w_base = positive_score / (1.0 + positive_score)

    w = w_base.clamp(min=0.0, max=1.0) * max_w

    if stop_gradient:
        w = w.detach()

    aux = {
        'ref_class': ref_class,
        'r_self': r_self,
        'd_mar': d_mar,
        's_raw': raw_score,
        's_hat': norm_score,
        'w_base': w_base,
        'w': w,
    }

    return w, aux


def build_reference_prototype(q_self, labels, train_mask, proto_self):
    """
    Construct the reference prototype r_i.

    Labeled nodes use the self-evidence prototype of their ground-truth class,
    while unlabeled nodes use the soft reference prototype weighted by q_self.
    """
    device = q_self.device
    num_nodes = q_self.size(0)

    labels = labels.to(device).long()
    train_mask = to_bool_mask(train_mask, num_nodes, device)

    ref_proto = torch.mm(q_self, proto_self)
    ref_proto[train_mask] = proto_self[labels[train_mask]]

    return ref_proto


def margin_completion_loss(h_prop, anchor, proto_prop, labels, train_mask, w, num_classes, eps=1e-12):
    """
    Margin-based completion consistency loss L_corr.

    This loss encourages the completion anchor u_i to provide a stronger
    class-consistent margin than the propagation representation h_prop.
    """
    device = h_prop.device
    num_nodes = h_prop.size(0)

    labels = labels.to(device).long()
    train_idx = mask_to_index(train_mask, num_nodes, device)

    if train_idx.numel() == 0:
        return h_prop.new_tensor(0.0)

    logits_prop = prototype_logits(
        h=h_prop,
        prototypes=proto_prop,
        normalize=True
    )

    logits_anchor = prototype_logits(
        h=anchor,
        prototypes=proto_prop,
        normalize=True
    )

    y = labels[train_idx]
    gather_index = y.view(-1, 1)

    prop_ref = logits_prop[train_idx].gather(1, gather_index)
    anchor_ref = logits_anchor[train_idx].gather(1, gather_index)

    margin_prop = prop_ref - logits_prop[train_idx]
    margin_anchor = anchor_ref - logits_anchor[train_idx]

    violation = (margin_prop - margin_anchor).clamp(min=0.0)

    # The ground-truth class itself is excluded from the average over competing classes.
    mask = torch.ones_like(violation, dtype=torch.bool)
    mask.scatter_(1, gather_index, False)
    violation = violation.masked_fill(~mask, 0.0)

    per_node = violation.sum(dim=1) / max(num_classes - 1, 1)

    weight = w[train_idx].detach()

    if weight.sum().item() <= 0:
        return h_prop.new_tensor(0.0)

    loss = (per_node * weight).sum() / (weight.sum() + eps)

    return loss


class EGPC_GNN(nn.Module):
    """
    EGPC-GNN1 model.

    The current version corresponds to the method section:
    1. Dual-evidence modeling;
    2. Propagation-induced margin degradation estimation;
    3. Evidence-guided propagation representation completion;
    4. Three loss terms: L_cls, L_evi, and L_corr.

    Compared with the purely theoretical implementation, this version
    introduces several necessary engineering safeguards:
    1. Warmup;
    2. Maximum completion weight egpc_max_w;
    3. Stable class-wise standardization;
    4. Class-balanced supervised loss.
    """

    def __init__(self, conf, data, device='cuda:0'):
        super().__init__()

        self.conf = conf
        self.data = data
        self.device = torch.device(device)

        self.num_classes = int(conf.model['n_class'])
        self.hidden_dim = int(conf.model['n_hidden'])

        self.eps = float(_get_loss_conf(conf, 'eps', 1e-12))

        self.self_encoder = SelfEvidenceEncoder(conf)
        self.prop_encoder = PropagationEvidenceEncoder(conf)
        self.anchor_mapper = AnchorMapper(conf)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

        # Loss weights.
        self.lambda_cls = float(_get_loss_conf(conf, 'lambda_cls', 1.0))
        self.lambda_e = float(_get_loss_conf(conf, 'lambda_e', 1.0))
        self.lambda_c = float(_get_loss_conf(conf, 'lambda_c', 1.0))

        # Supervised loss settings.
        self.use_class_balanced_loss = bool(_get_loss_conf(conf, 'use_class_balanced_loss', True))

        # Completion weight settings.
        self.egpc_max_w = float(_get_loss_conf(conf, 'egpc_max_w', 0.05))
        self.egpc_stop_weight_grad = bool(_get_loss_conf(conf, 'egpc_stop_weight_grad', True))

        # Warmup settings.
        self.use_warmup = bool(_get_loss_conf(conf, 'use_warmup', False))
        self.warmup = int(_get_loss_conf(conf, 'warmup', 0))

        # Stable class-wise standardization settings.
        self.standardize_with_unlabeled = bool(_get_loss_conf(conf, 'standardize_with_unlabeled', True))
        self.min_labeled_for_class_stat = int(_get_loss_conf(conf, 'min_labeled_for_class_stat', 3))
        self.min_std = float(_get_loss_conf(conf, 'min_std', 1e-3))
        self.clip_s_hat = float(_get_loss_conf(conf, 'clip_s_hat', 10.0))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the final classification head.
        """
        nn.init.xavier_uniform_(self.classifier.weight)

        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.0)

    def encode(self, feats, adj):
        """
        Compute the self-evidence representation and propagation-evidence representation.
        """
        h_self = self.self_encoder(feats)
        h_prop = self.prop_encoder(feats, adj)

        return h_self, h_prop

    def complete(self, h_prop, anchor, w):
        """
        Perform evidence completion on the propagation representation according to w_i.
        """
        w = w.view(-1, 1)
        return (1.0 - w) * h_prop + w * anchor

    def _is_warmup(self, epoch):
        """
        Determine whether the current epoch is in the warmup stage.
        """
        return self.use_warmup and epoch is not None and epoch < self.warmup

    def forward(self, feats, adj, train_mask, labels, epoch=None):
        """
        Perform forward propagation and return logits, total loss, and auxiliary information.
        """
        device = feats.device
        num_nodes = feats.size(0)

        labels = labels.to(device).long()
        train_mask = to_bool_mask(train_mask, num_nodes, device)

        # 1. Dual-evidence encoding.
        h_self, h_prop = self.encode(feats, adj)

        # 2. Construct self-evidence prototypes and propagation-evidence prototypes
        #    using labeled nodes.
        proto_self = compute_class_prototypes(
            h=h_self,
            labels=labels,
            train_mask=train_mask,
            num_classes=self.num_classes,
            normalize=True
        )

        proto_prop = compute_class_prototypes(
            h=h_prop,
            labels=labels,
            train_mask=train_mask,
            num_classes=self.num_classes,
            normalize=True
        )

        # 3. Compute prototype-based evidence logits and evidence distributions.
        logits_self = prototype_logits(
            h=h_self,
            prototypes=proto_self,
            normalize=True
        )

        logits_prop = prototype_logits(
            h=h_prop,
            prototypes=proto_prop,
            normalize=True
        )

        q_self = F.softmax(logits_self, dim=1)
        q_prop = F.softmax(logits_prop, dim=1)

        # 4. Compute completion weights based on propagation-induced margin degradation.
        w, weight_aux = compute_margin_completion_weight(
            q_self=q_self,
            logits_self=logits_self,
            logits_prop=logits_prop,
            labels=labels,
            train_mask=train_mask,
            num_classes=self.num_classes,
            max_w=self.egpc_max_w,
            stop_gradient=self.egpc_stop_weight_grad,
            standardize_with_unlabeled=self.standardize_with_unlabeled,
            min_labeled_for_class_stat=self.min_labeled_for_class_stat,
            min_std=self.min_std,
            clip_s_hat=self.clip_s_hat,
            eps=self.eps
        )

        is_warmup = self._is_warmup(epoch)

        # Disable completion during the warmup stage.
        if is_warmup:
            w = torch.zeros_like(w)

        # 5. Construct the reference prototype r_i and completion anchor u_i.
        ref_proto_self = build_reference_prototype(
            q_self=q_self,
            labels=labels,
            train_mask=train_mask,
            proto_self=proto_self
        )

        anchor = self.anchor_mapper(h_self, ref_proto_self)

        # 6. Representation completion and final classification.
        h_tilde = self.complete(h_prop, anchor, w)
        logits = self.classifier(h_tilde)

        # 7. Classification loss L_cls.
        loss_cls = supervised_loss(
            logits=logits,
            labels=labels,
            train_mask=train_mask,
            num_classes=self.num_classes,
            balanced=self.use_class_balanced_loss
        )

        # 8. Evidence supervision loss L_evi.
        loss_evi = (
            supervised_loss(
                logits=logits_self,
                labels=labels,
                train_mask=train_mask,
                num_classes=self.num_classes,
                balanced=self.use_class_balanced_loss
            )
            + supervised_loss(
                logits=logits_prop,
                labels=labels,
                train_mask=train_mask,
                num_classes=self.num_classes,
                balanced=self.use_class_balanced_loss
            )
        )

        # 9. Margin-based completion consistency loss L_corr.
        if is_warmup:
            loss_corr = h_prop.new_tensor(0.0)
        else:
            loss_corr = margin_completion_loss(
                h_prop=h_prop,
                anchor=anchor,
                proto_prop=proto_prop,
                labels=labels,
                train_mask=train_mask,
                w=w,
                num_classes=self.num_classes,
                eps=self.eps
            )

        # 10. Total objective: L = lambda_cls L_cls + lambda_e L_evi + lambda_c L_corr.
        loss = (
            self.lambda_cls * loss_cls
            + self.lambda_e * loss_evi
            + self.lambda_c * loss_corr
        )

        # Synchronize w in aux to ensure that the debug output reflects
        # the actual w used after warmup processing.
        weight_aux['w'] = w

        aux = {
            'loss_cls': loss_cls.detach(),
            'loss_evi': loss_evi.detach(),
            'loss_corr': loss_corr.detach(),
            'w': w.detach(),
            'h_self': h_self.detach(),
            'h_prop': h_prop.detach(),
            'h_tilde': h_tilde.detach(),
            'anchor': anchor.detach(),
            'logits_self': logits_self.detach(),
            'logits_prop': logits_prop.detach(),
            'q_self': q_self.detach(),
            'q_prop': q_prop.detach(),
            'proto_self': proto_self.detach(),
            'proto_prop': proto_prop.detach(),
            'is_warmup': is_warmup,
        }

        for key, value in weight_aux.items():
            aux[key] = value.detach() if torch.is_tensor(value) else value

        return logits, loss, aux

