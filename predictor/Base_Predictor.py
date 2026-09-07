import time
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils.logger import SingleExpRecorder
from copy import deepcopy
import nni


class Predictor:
    def __init__(self, conf, data, device='cuda:0'):
        super(Predictor, self).__init__()

        self.conf = conf
        self.data = data
        self.device = torch.device(device)

        self.general_init(conf, data)
        self.method_init(conf, data)

    def general_init(self, conf, data):
        """
        General initialization function.

        The current version is designed for class-imbalanced node classification:
        1. Only data.labels is used;
        2. Clean labels and perturbed labels are no longer distinguished;
        3. Training, validation, and testing all use the same label tensor;
        4. Data splits are provided by data.train_masks / data.val_masks / data.test_masks
           and the corresponding boolean masks.
        """

        self.loss_fn = F.binary_cross_entropy_with_logits if data.n_classes == 1 else F.cross_entropy
        self.metric = roc_auc_score if data.n_classes == 1 else self._accuracy_from_numpy

        # Graph structure
        self.edge_index = data.adj.indices()
        self.adj = data.adj if self.conf.dataset['sparse'] else data.adj.to_dense()

        if conf.dataset['add_self_loop']:
            self.edge_index_loop = (
                data.edge_index_loop
                if self.conf.dataset['sparse']
                else data.edge_index_loop.to_dense()
            )
            self.adj_with_loop = (
                data.adj_with_loop
                if self.conf.dataset['sparse']
                else data.adj_with_loop.to_dense()
            )

        if conf.dataset['adj_norm']:
            self.edge_index_normliazed = (
                data.normalized_adj.coalesce().indices()
                if self.conf.dataset['sparse']
                else data.normalized_adj.coalesce().indices().to_dense()
            )
            self.normalized_adj = (
                data.normalized_adj
                if self.conf.dataset['sparse']
                else data.normalized_adj.to_dense()
            )
        else:
            self.edge_index_normliazed = self.edge_index
            self.normalized_adj = self.adj

        # Experiment recorder
        self.recoder = SingleExpRecorder(
            self.conf.training['patience'],
            self.conf.training['criterion']
        )

        # Basic data information
        self.feats = data.feats
        self.n_nodes = data.n_nodes
        self.n_class = data.n_classes
        self.n_feat = data.feats.shape[1]

        # Labels
        self.labels = data.labels

        # ndarray-format indices
        self.train_mask = data.train_masks
        self.val_mask = data.val_masks
        self.test_mask = data.test_masks
        self.unlabel_mask = data.unlabel_masks

        # Tensor-format indices
        self.idx_train = data.idx_train
        self.idx_val = data.idx_val
        self.idx_test = data.idx_test
        self.idx_unlabel = data.idx_unlabel

        # Boolean masks
        self.mask_train = data.mask_train
        self.mask_val = data.mask_val
        self.mask_test = data.mask_test
        self.mask_unlabel = data.mask_unlabel

        # Result records
        self.result = {
            'train': -1,
            'valid': -1,
            'test': -1
        }

        self.weights = None
        self.start_time = time.time()
        self.total_time = -1

        self.best_pred = None
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_acc_pred_val = 0

        self.data = data
        self.conf = conf

    def method_init(self, conf, data):
        """
        Method-specific initialization function.

        Subclasses should override this function to create the model and optimizer.
        """
        self.model = None
        self.optim = None
        return None

    def _to_bool_mask(self, mask_or_index):
        """
        Convert ndarray, index tensor, or boolean tensor into a unified boolean mask.
        """
        if torch.is_tensor(mask_or_index):
            mask = mask_or_index.to(self.device)
        else:
            mask = torch.as_tensor(mask_or_index, device=self.device)

        if mask.dtype == torch.bool and mask.numel() == self.n_nodes:
            return mask

        if mask.numel() == self.n_nodes and mask.dtype != torch.bool:
            unique_values = torch.unique(mask)
            if torch.all((unique_values == 0) | (unique_values == 1)):
                return mask.bool()

        bool_mask = torch.zeros(self.n_nodes, dtype=torch.bool, device=self.device)
        bool_mask[mask.long()] = True

        return bool_mask

    def _accuracy_from_numpy(self, labels_np, logits_np):
        """
        Compute multi-class classification accuracy.

        Compatible with the metric(label, output) calling convention
        used in the original framework.
        """
        labels = torch.as_tensor(labels_np).long()
        logits = torch.as_tensor(logits_np)

        if labels.numel() == 0:
            return 0.0

        pred = logits.argmax(dim=1)

        return float((pred == labels).float().mean().item())

    def input_distributer(self):
        """
        Construct the basic model input.
        """
        return {
            'x': self.feats,
            'adj': self.normalized_adj
        }

    def get_prediction(self, label=None, mask=None):
        """
        Perform model forward propagation and compute the loss and metric
        on the specified split.
        """
        output = self.model(**self.input_distributer())

        loss, acc = None, None

        if (label is not None) and (mask is not None):
            mask = self._to_bool_mask(mask)
            label = label.to(output.device).long()

            loss = self.loss_fn(output[mask], label[mask])

            acc = self.metric(
                label[mask].detach().cpu().numpy(),
                output[mask].detach().cpu().numpy()
            )

        return output, loss, acc

    def train(self):
        """
        General training procedure.

        Special methods may override this function.
        """
        for epoch in range(self.conf.training['n_epochs']):

            improve = ''
            t0 = time.time()

            self.model.train()
            self.optim.zero_grad()

            output, loss_train, acc_train = self.get_prediction(
                self.labels,
                self.mask_train
            )

            loss_train.backward()
            self.optim.step()

            loss_val, acc_val = self.evaluate(
                self.labels,
                self.mask_val
            )

            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())

            elif flag_earlystop:
                break

            if self.conf.training['debug']:
                nni.report_intermediate_result(acc_val)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | "
                    "Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1,
                        time.time() - t0,
                        loss_train.item(),
                        acc_train,
                        loss_val.item() if torch.is_tensor(loss_val) else float(loss_val),
                        acc_val,
                        improve
                    )
                )

        loss_test, acc_test = self.test(self.mask_test)
        self.result['test'] = acc_test

        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print(
                "Loss(test) {:.4f} | Acc(test) {:.4f}".format(
                    loss_test.item() if torch.is_tensor(loss_test) else float(loss_test),
                    acc_test
                )
            )

        return self.result

    def evaluate(self, label, mask):
        """
        Evaluate the model on the specified split.
        """
        self.model.eval()

        with torch.no_grad():
            _, loss, acc = self.get_prediction(label, mask)

        return loss, acc

    def test(self, mask):
        """
        Evaluate the model on the test set.
        """
        if self.weights is not None:
            self.model.load_state_dict(self.weights)

        return self.evaluate(self.labels, mask)

