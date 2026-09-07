import os
import argparse
import warnings

import numpy as np

from utils.dataloader import Dataset
from utils.tools import load_conf, setup_seed

# Import the class-imbalance split function
from utils.datasplit import get_classnum_imb_split

# Import all predictor classes
# For example, rgldgnn_Predictor will be dynamically called through eval(args.method + "_Predictor")
from predictor import *


warnings.filterwarnings("ignore")

# Some CUDA operations require this environment variable when using deterministic PyTorch algorithms
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_args():
    """
    Parse command-line arguments.

    The current version is used for graph class-imbalance experiments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="amazoncom",
        choices=["cora", "citeseer", "amazoncom", "amazonpho", 'chameleon', 'squirrel', 'actor'],
        help="Dataset name."
    )

    parser.add_argument(
        "--method",
        type=str,
        default="egpcgnn",
        help="Method name."
    )

    parser.add_argument(
        "--imb_ratio",
        type=float,
        default=100,
        help="Class imbalance ratio. For example, 10 denotes approximately 10:1, 20 denotes approximately 20:1, and 100 denotes approximately 100:1."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Training device, e.g., cuda:0 or cpu."
    )

    return parser.parse_args()


def get_seed_from_conf(conf):
    """
    Read the random seed from the configuration file.

    Recommended configuration format:

    training:
      seed: 31
    """

    if "seed" in conf.training:
        return conf.training["seed"]

    return 2024


def print_label_status(data, title="Current label split status"):
    """
    Print the label distribution of the current dataset.

    The printed information includes:
    1. Number of nodes per class in the full dataset;
    2. Number of nodes per class in the training set;
    3. Number of nodes per class in the validation set;
    4. Number of nodes per class in the test set;
    5. Number of elements in idx_train / idx_val / idx_test;
    6. Number of elements in mask_train / mask_val / mask_test;
    7. Actual class imbalance ratio of the training set.
    """

    labels = data.labels.detach().cpu().long()
    n_classes = int(labels.max().item()) + 1

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    # Number of nodes per class in the full dataset
    all_counts = torch.bincount(
        labels,
        minlength=n_classes
    ).tolist()

    print(f"Number of nodes per class in the full dataset: {all_counts}")
    print(f"Total number of nodes in the full dataset: {sum(all_counts)}")

    # Number of nodes per class in the training set
    train_labels = data.labels[data.mask_train].detach().cpu().long()
    train_counts = torch.bincount(
        train_labels,
        minlength=n_classes
    ).tolist()

    print(f"Number of nodes per class in the training set: {train_counts}")
    print(f"Total number of nodes in the training set: {sum(train_counts)}")

    # Number of nodes per class in the validation set
    val_labels = data.labels[data.mask_val].detach().cpu().long()
    val_counts = torch.bincount(
        val_labels,
        minlength=n_classes
    ).tolist()

    print(f"Number of nodes per class in the validation set: {val_counts}")
    print(f"Total number of nodes in the validation set: {sum(val_counts)}")

    # Number of nodes per class in the test set
    test_labels = data.labels[data.mask_test].detach().cpu().long()
    test_counts = torch.bincount(
        test_labels,
        minlength=n_classes
    ).tolist()

    print(f"Number of nodes per class in the test set: {test_counts}")
    print(f"Total number of nodes in the test set: {sum(test_counts)}")

    # Index count check
    print("\nIndex count check:")
    print(f"Number of idx_train elements: {data.idx_train.numel()}")
    print(f"Number of idx_val elements: {data.idx_val.numel()}")
    print(f"Number of idx_test elements: {data.idx_test.numel()}")

    # Mask count check
    print("\nMask count check:")
    print(f"Number of mask_train elements: {int(data.mask_train.sum().item())}")
    print(f"Number of mask_val elements: {int(data.mask_val.sum().item())}")
    print(f"Number of mask_test elements: {int(data.mask_test.sum().item())}")

    # Check the class imbalance ratio of the training set
    train_nonzero = [x for x in train_counts if x > 0]

    if len(train_nonzero) > 0:
        max_train = max(train_nonzero)
        min_train = min(train_nonzero)
        real_ratio = max_train / min_train

        print("\nTraining-set class imbalance check:")
        print(f"Largest class size in the training set: {max_train}")
        print(f"Smallest nonzero class size in the training set: {min_train}")
        print(f"Actual max/min ratio in the training set: {real_ratio:.4f}")

    if 0 in train_counts:
        print("\nWarning: The training set contains classes with zero samples, which may cause NaN values or sampling errors in some methods.")

    print("=" * 100 + "\n")


if __name__ == "__main__":

    # Parse command-line arguments
    args = parse_args()

    # Dataset root directory
    data_path = "dataset"

    # Load the configuration file
    # For example: ./config/rgldgnn_cora.yaml
    conf = load_conf(f"./config/{args.method}/{args.method}_{args.dataset}.yaml")

    # Save the class imbalance ratio to the configuration for later modules or logging
    conf.imb_ratio = args.imb_ratio

    # Read the random seed
    args.seed = get_seed_from_conf(conf)

    print("=" * 100)
    print("Experiment parameters:")
    print(args)
    print("=" * 100)

    # Fix the random seed
    setup_seed(args.seed)

    # Load the dataset
    data = Dataset(
        args.dataset,
        path=data_path,

        # Whether to normalize node features
        feat_norm=conf.dataset["feat_norm"],

        # Whether to normalize the adjacency matrix
        adj_norm=conf.dataset["adj_norm"],

        # Fixed-size split parameters
        train_size=conf.dataset["train_size"],
        val_size=conf.dataset["val_size"],
        test_size=conf.dataset["test_size"],

        # Percentage-based split parameters
        train_percent=conf.dataset["train_percent"],
        val_percent=conf.dataset["val_percent"],
        test_percent=conf.dataset["test_percent"],

        # Fixed number of samples per class for splitting
        train_examples_per_class=conf.dataset["train_examples_per_class"],
        val_examples_per_class=conf.dataset["val_examples_per_class"],
        test_examples_per_class=conf.dataset["test_examples_per_class"],

        # Whether to add self-loops
        add_self_loop=conf.dataset["add_self_loop"],

        # Whether to load the largest connected component from NPZ data
        from_npz=conf.dataset["from_npz_largest_component"],

        # Device on which the data is stored
        device=args.device,

        # Original split strategy.
        # Note: The original split will later be overwritten by get_classnum_imb_split().
        split_type=conf.dataset["split_type"]
    )

    print(f"Current device of node features: {data.feats.device}")

    # Print the original split status
    print_label_status(
        data,
        title=f"{args.dataset} original split label status"
    )

    # Use the class-imbalance split to overwrite the original train / val / test split
    data = get_classnum_imb_split(
        target_data=data,
        imb_level=args.imb_ratio,
        shuffle_seed=args.seed,
        debug=True
    )

    # Print the final data split used for model training
    print_label_status(
        data,
        title=f"{args.dataset} final label status after class-imbalance split | imb_ratio={args.imb_ratio}"
    )

    # Update the model input dimension and number of classes
    conf.model["n_feat"] = data.dim_feats
    conf.model["n_class"] = data.n_classes

    # Enable training debug information
    conf.training["debug"] = True

    # Store test results
    test_results = []

    # Currently run only once; change to range(num_runs) if multiple runs are needed later
    for run_id in range(1):
        print("=" * 100)
        print(f"Starting experiment run {run_id + 1}")
        print("=" * 100)

        # Reset the random seed before each run
        setup_seed(args.seed)

        # Dynamically construct the predictor
        predictor = eval(args.method + "_Predictor")(conf, data, args.device)

        # Train the model and return the results
        result = predictor.train()

        # Store the current test results
        test_results.append(result["test"])

    # Compute the mean and standard deviation of all test metrics
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc"
    ]

    print("=" * 100)
    print("Final test results")
    print("=" * 100)

    for metric_name in metric_names:
        values = np.array(
            [item[metric_name] for item in test_results],
            dtype=np.float64
        )

        mean_value = np.nanmean(values)
        std_value = np.nanstd(values)

        print(f"Test {metric_name} Mean: {mean_value:.4f}")
        print(f"Test {metric_name} Std:  {std_value:.4f}")

    print("=" * 100)

