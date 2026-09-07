import time
from copy import deepcopy

import torch
import torch.nn.functional as F

from predictor.Base_Predictor import Predictor
from predictor.model.EGPC_GNN import EGPC_GNN
from utils.tools import classification_metrics_from_logits, to_bool_mask


class egpcgnn_Predictor(Predictor):
    """
    EGPC-GNN predictor.

    Outputs four metrics:
    1. Accuracy;
    2. Balanced Accuracy;
    3. Macro F1;
    4. ROC-AUC.
    """

    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        """
        Initialize the model and optimizer.
        """
        self.main_model = EGPC_GNN(
            conf=conf,
            data=data,
            device=self.device
        ).to(self.device)

        self.optim = torch.optim.Adam(
            self.main_model.parameters(),
            lr=self.conf.training['lr'],
            weight_decay=self.conf.training['weight_decay']
        )

    def _select_score(self, metrics):
        """
        Select the early-stopping metric according to the configuration.

        Supported options:
        acc / accuracy
        bacc / balanced_accuracy
        macro_f1 / f1
        roc_auc / auc
        """
        select_metric = self.conf.training.get("select_metric", "accuracy")

        if select_metric in ["acc", "accuracy"]:
            score = metrics["accuracy"]
        elif select_metric in ["bacc", "balanced_accuracy"]:
            score = metrics["balanced_accuracy"]
        elif select_metric in ["macro_f1", "f1"]:
            score = metrics["macro_f1"]
        elif select_metric in ["roc_auc", "auc"]:
            score = metrics["roc_auc"]
        else:
            score = metrics["accuracy"]

        # Avoid NaN values affecting the early-stopping recorder
        # when ROC-AUC cannot be computed.
        if score != score:
            return 0.0

        return score

    def _bool_mask(self, mask, num_nodes, device):
        """
        Convert a split mask or indices into a unified boolean mask.
        """
        return to_bool_mask(mask, num_nodes, device)

    def train(self):
        """
        Train EGPC-GNN1.
        """
        best_output = None
        best_pred = None
        best_epoch = -1

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()

            self.main_model.train()
            self.optim.zero_grad()

            output, loss_train, aux = self.main_model(
                feats=self.feats,
                adj=self.normalized_adj,
                train_mask=self.mask_train,
                labels=self.labels,
                epoch=epoch
            )

            train_mask = self._bool_mask(self.mask_train, output.size(0), output.device)

            train_metrics = classification_metrics_from_logits(
                logits=output[train_mask],
                labels=self.labels.to(output.device)[train_mask],
                num_classes=self.n_class
            )

            loss_train.backward()
            self.optim.step()

            loss_val, val_metrics = self.evaluate(
                x=self.feats,
                adj=self.normalized_adj,
                mask=self.mask_val,
                label=self.labels,
                epoch=epoch
            )

            val_score = self._select_score(val_metrics)
            loss_val_value = loss_val.item() if torch.is_tensor(loss_val) else float(loss_val)

            flag, flag_earlystop = self.recoder.add(loss_val, val_score)

            if flag:
                improve = '*'
                best_epoch = epoch
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val_value

                self.result['valid'] = val_metrics
                self.result['train'] = train_metrics

                self.weights = deepcopy(self.main_model.state_dict())
                best_output = output.detach().clone()
                best_pred = output.argmax(dim=1).detach().clone()

            elif flag_earlystop:
                break

            if self.conf.training['debug']:
                loss_cls = aux['loss_cls'].item() if 'loss_cls' in aux else 0.0
                loss_evi = aux['loss_evi'].item() if 'loss_evi' in aux else 0.0
                loss_corr = aux['loss_corr'].item() if 'loss_corr' in aux else 0.0

                w = aux['w'] if 'w' in aux and torch.is_tensor(aux['w']) else None
                s_hat = aux['s_hat'] if 's_hat' in aux and torch.is_tensor(aux['s_hat']) else None

                w_mean = w.mean().item() if w is not None else 0.0
                w_max = w.max().item() if w is not None and w.numel() > 0 else 0.0

                d_mar_mean = aux['d_mar'].mean().item() if 'd_mar' in aux else 0.0
                r_self_mean = aux['r_self'].mean().item() if 'r_self' in aux else 0.0
                s_hat_mean = s_hat.mean().item() if s_hat is not None else 0.0
                s_hat_abs_max = s_hat.abs().max().item() if s_hat is not None and s_hat.numel() > 0 else 0.0

                is_warmup = aux.get('is_warmup', False)

                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | "
                    "Train Acc {:.4f} | Train BAcc {:.4f} | Train Macro-F1 {:.4f} | Train ROC-AUC {:.4f} | "
                    "Loss(val) {:.4f} | Val Acc {:.4f} | Val BAcc {:.4f} | Val Macro-F1 {:.4f} | Val ROC-AUC {:.4f} | "
                    "L_cls {:.4f} | L_evi {:.4f} | L_corr {:.4f} | "
                    "w_mean {:.4f} | w_max {:.4f} | d_mar {:.4f} | r_self {:.4f} | "
                    "s_hat_mean {:.4f} | s_hat_abs_max {:.4f} | warmup {} | {}".format(
                        epoch + 1,
                        time.time() - t0,
                        loss_train.item(),

                        train_metrics["accuracy"],
                        train_metrics["balanced_accuracy"],
                        train_metrics["macro_f1"],
                        train_metrics["roc_auc"],

                        loss_val_value,
                        val_metrics["accuracy"],
                        val_metrics["balanced_accuracy"],
                        val_metrics["macro_f1"],
                        val_metrics["roc_auc"],

                        loss_cls,
                        loss_evi,
                        loss_corr,
                        w_mean,
                        w_max,
                        d_mar_mean,
                        r_self_mean,
                        s_hat_mean,
                        s_hat_abs_max,
                        is_warmup,
                        improve
                    )
                )

        if self.weights is None:
            self.weights = deepcopy(self.main_model.state_dict())

            self.main_model.eval()
            with torch.no_grad():
                best_output, _, _ = self.main_model(
                    feats=self.feats,
                    adj=self.normalized_adj,
                    train_mask=self.mask_train,
                    labels=self.labels,
                    epoch=self.conf.training['n_epochs'] - 1
                )
                best_pred = best_output.argmax(dim=1)

        self.main_model.load_state_dict(self.weights)

        # Using best_epoch is more consistent with the early-stopping logic.
        # If it was not recorded, fall back to final_epoch.
        eval_epoch = best_epoch if best_epoch >= 0 else self.conf.training['n_epochs'] - 1

        loss_test, test_metrics = self.evaluate(
            x=self.feats,
            adj=self.normalized_adj,
            mask=self.mask_test,
            label=self.labels,
            epoch=eval_epoch
        )

        self.result['test'] = test_metrics

        self.main_model.eval()
        with torch.no_grad():
            best_output, _, _ = self.main_model(
                feats=self.feats,
                adj=self.normalized_adj,
                train_mask=self.mask_train,
                labels=self.labels,
                epoch=eval_epoch
            )
            best_pred = best_output.argmax(dim=1)

        self.result['logits'] = best_output.detach().cpu()
        self.result['pred'] = best_pred.detach().cpu()

        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Best Epoch: {}'.format(eval_epoch + 1))
            print('Time(s): {:.4f}'.format(self.total_time))
            print(
                "Loss(test) {:.4f} | "
                "Test Acc {:.4f} | Test BAcc {:.4f} | Test Macro-F1 {:.4f} | Test ROC-AUC {:.4f}".format(
                    loss_test.item() if torch.is_tensor(loss_test) else float(loss_test),
                    test_metrics["accuracy"],
                    test_metrics["balanced_accuracy"],
                    test_metrics["macro_f1"],
                    test_metrics["roc_auc"]
                )
            )

        return self.result

    def evaluate(self, x, adj, mask, label, epoch):
        """
        Evaluate the model on the specified split.
        """
        self.main_model.eval()

        with torch.no_grad():
            output, _loss_total, _aux = self.main_model(
                feats=x,
                adj=adj,
                train_mask=self.mask_train,
                labels=self.labels,
                epoch=epoch
            )

            label = label.to(output.device).long()
            mask = self._bool_mask(mask, output.size(0), output.device)

            if mask.sum().item() == 0:
                loss_eval = output.new_tensor(0.0)
            else:
                loss_eval = F.cross_entropy(
                    output[mask],
                    label[mask]
                )

            metrics = classification_metrics_from_logits(
                logits=output[mask],
                labels=label[mask],
                num_classes=self.n_class
            )

        return loss_eval, metrics



