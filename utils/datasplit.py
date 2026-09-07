from sklearn.model_selection import StratifiedKFold
import random
import numpy as np
import torch
import math

def resolve_imb_ratio(imb_level):
    """
    Convert the class imbalance level to a specific imbalance ratio.

    Examples:
    imb_level = 1   indicates class balance;
    imb_level = 10  indicates an approximate 10:1 ratio between the head and tail classes;
    imb_level = 20  indicates an approximate 20:1 ratio between the head and tail classes;
    imb_level = 100 indicates an approximate 100:1 ratio between the head and tail classes.
    """
    if isinstance(imb_level, str):
        return float(imb_level)

    return float(imb_level)


def sort(data):
    """
    Sort classes in descending order according to the number of nodes in each class.

    Parameters
    ----
    data:
        Dataset object containing data.labels.

    Returns
    ----
    class_num_list:
        Total number of nodes in each class in the original class order.

    indices:
        Sorted class indices, with classes containing more samples placed first.

    inv_indices:
        Indices used to restore the sorted long-tailed counts to the original class order.
    """
    labels = data.labels.detach().cpu().long()
    n_classes = int(labels.max().item()) + 1

    class_num_list = torch.bincount(labels, minlength=n_classes).long()

    _, indices = torch.sort(class_num_list, descending=True)

    inv_indices = torch.empty_like(indices)
    inv_indices[indices] = torch.arange(n_classes)

    return class_num_list, indices, inv_indices

#
# def split_lt(class_num_list, indices, inv_indices, imb_ratio, n_cls, n, keep=0):
#     """
#     Calculate the number of training nodes assigned to each class according to a long-tailed distribution.
#
#     Parameters
#     ----
#     class_num_list:
#         Original number of nodes in each class.
#
#     indices:
#         Class indices sorted in descending order by class size.
#
#     inv_indices:
#         Indices used to restore the original class order.
#
#     imb_ratio:
#         Class imbalance ratio, e.g., 10, 20, or 100.
#
#     n_cls:
#         Number of classes.
#
#     n:
#         Total number of training nodes, usually set to 10% of the total number of nodes.
#
#     keep:
#         Number of reserved nodes per class, default is 0.
#
#     Returns
#     ----
#     class_num_list_train:
#         Number of training nodes assigned to each class in the original class order.
#     """
#     class_num_list = class_num_list[indices]
#
#     imb_ratio = float(imb_ratio)
#
#     if imb_ratio == 1:
#         n_max = n / n_cls
#         mu = 1.0
#         inv_mu = 1.0
#     else:
#         mu = np.power(imb_ratio, 1.0 / (n_cls - 1))
#         inv_mu = 1.0 / mu
#         n_max = n / (imb_ratio * mu - 1) * (mu - 1) * imb_ratio
#
#     class_num_list_lt = []
#
#     for i in range(n_cls):
#         target_num = n_max * np.power(inv_mu, i)
#
#         # In principle, keep at least one training node per class to avoid training errors caused by empty classes
#         target_num = max(target_num, 1)
#
#         # The current class cannot use more nodes than are available
#         available_num = int(class_num_list[i].item() - keep)
#
#         if available_num <= 0:
#             final_num = 0
#         else:
#             final_num = round(min(target_num, available_num))
#             final_num = max(final_num, 1)
#
#         class_num_list_lt.append(final_num)
#
#     class_num_list_lt = torch.tensor(class_num_list_lt, dtype=torch.long)
#
#     # Restore the original class order
#     return class_num_list_lt[inv_indices]

def split_lt(
    class_num_list,
    indices,
    inv_indices,
    imb_ratio,
    n_cls,
    n,
    keep=0,
    debug=False
):
    """
    Construct integer-valued long-tailed training class counts.

    Strictly guarantee:
    1. The total number of training nodes is exactly equal to n;
    2. The maximum class size / minimum class size is exactly equal to imb_ratio;
    3. Class counts are monotonically non-increasing from the majority class to the minority class;
    4. The count does not exceed the number of actually available nodes in each class.

    Guarantee as much as possible:
    5. The final integer class counts are close to an exponential long-tailed distribution.

    Note:
    Because class counts must be integers, it is generally impossible to strictly satisfy
    exactly the same exponential ratio between every pair of adjacent classes at the same time.
    """

    # ---------------------------------------------------------
    # 1. Parameter validation
    # ---------------------------------------------------------
    rho_float = float(imb_ratio)
    rho = int(round(rho_float))

    if not np.isclose(
        rho_float,
        rho,
        rtol=0.0,
        atol=1e-12
    ):
        raise ValueError(
            f"The current implementation requires imb_ratio to be an integer, "
            f"but the current value is {imb_ratio}"
        )

    if rho < 1:
        raise ValueError(
            f"imb_ratio must be greater than or equal to 1, but the current value is {rho}"
        )

    if n_cls != len(class_num_list):
        raise ValueError(
            f"n_cls={n_cls}，"
            f"but the length of class_num_list is "
            f"{len(class_num_list)}"
        )

    if n <= 0:
        raise ValueError(
            f"The total training-set size n must be greater than 0, but the current value is {n}"
        )

    indices_cpu = (
        indices
        .detach()
        .cpu()
        .long()
    )

    inv_indices_cpu = (
        inv_indices
        .detach()
        .cpu()
        .long()
    )

    # ---------------------------------------------------------
    # 2. Sort classes in descending order by their original sizes
    # ---------------------------------------------------------
    available_sorted = (
        class_num_list
        .detach()
        .cpu()
        .long()[indices_cpu]
        - int(keep)
    ).clamp_min(0)

    available_np = (
        available_sorted
        .numpy()
        .astype(np.int64)
    )

    if np.any(available_np <= 0):
        raise ValueError(
            "At least one class has no available nodes: "
            f"{available_np.tolist()}"
        )

    # ---------------------------------------------------------
    # 3. Balanced case: rho = 1
    # ---------------------------------------------------------
    if rho == 1:
        # Use the largest strictly balanced split that does not exceed the target training-set size n
        per_class = n // n_cls

        if per_class < 1:
            raise ValueError(
                f"The total number of training nodes is too small to ensure at least one sample per class: "
                f"n={n}, n_cls={n_cls}"
            )

        if np.any(available_np < per_class):
            insufficient_classes = np.where(
                available_np < per_class
            )[0].tolist()

            raise ValueError(
                f"Some classes have fewer than {per_class} available nodes, "
                f"class positions={insufficient_classes}, "
                f"available counts={available_np.tolist()}"
            )

        counts_sorted = np.full(
            n_cls,
            per_class,
            dtype=np.int64
        )

        actual_total = int(
            counts_sorted.sum()
        )

        if debug:
            print("\n" + "=" * 100)
            print("Strictly balanced split result")
            print("=" * 100)
            print(f"Target number of training nodes: {n}")
            print(f"Number of classes: {n_cls}")
            print(f"Training nodes per class: {per_class}")
            print(f"Actual total number of training nodes: {actual_total}")
            print(f"Number of nodes not assigned to the training set: {n - actual_total}")
            print(f"Training class counts: {counts_sorted.tolist()}")
            print("Actual imbalance ratio: 1.0")
            print("=" * 100 + "\n")

        counts_tensor = torch.tensor(
            counts_sorted,
            dtype=torch.long
        )

        return counts_tensor[inv_indices_cpu]

    # ---------------------------------------------------------
    # 4. Theoretical exponential decay factor
    #
    # From the majority class to the minority class:
    # n_i = n_max * r^i
    #
    # r^(n_cls-1) = 1 / rho
    # ---------------------------------------------------------
    decay_ratio = np.power(
        rho,
        -1.0 / (n_cls - 1)
    )

    # Feasible upper bound for the minority-class count
    max_minority_count = min(
        int(available_np[-1]),
        int(available_np[0] // rho)
    )

    if max_minority_count < 1:
        raise ValueError(
            f"Unable to construct a strict {rho}:1 split. "
            f"Available majority-class count={available_np[0]}, "
            f"available minority-class count={available_np[-1]}"
        )

    best_solution = None
    best_score = None

    # ---------------------------------------------------------
    # 5. Enumerate integer minority-class counts
    # ---------------------------------------------------------
    for minority_count in range(
        1,
        max_minority_count + 1
    ):
        majority_count = (
            rho * minority_count
        )

        if majority_count > available_np[0]:
            continue

        # Every class must be able to accommodate at least minority_count nodes
        if np.any(
            available_np < minority_count
        ):
            continue

        # After fixing the head and tail classes, the middle classes must account for this total
        middle_total = (
            n
            - majority_count
            - minority_count
        )

        if n_cls == 2:
            if middle_total != 0:
                continue

            counts_sorted = np.array(
                [
                    majority_count,
                    minority_count
                ],
                dtype=np.int64
            )

            score = 0.0

        else:
            if middle_total < 0:
                continue

            # -------------------------------------------------
            # 5.1 Original continuous exponential sequence
            # -------------------------------------------------
            ideal_counts = np.array(
                [
                    majority_count
                    * np.power(
                        decay_ratio,
                        i
                    )
                    for i in range(n_cls)
                ],
                dtype=np.float64
            )

            ideal_middle_sum = float(
                ideal_counts[1:-1].sum()
            )

            if ideal_middle_sum <= 0:
                continue

            # -------------------------------------------------
            # 5.2 Uniformly scale the middle classes
            #
            # Preserve the exponential relationship among the middle classes,
            # while ensuring that the total continuous target count is exactly equal to n.
            # -------------------------------------------------
            middle_scale = (
                middle_total
                / ideal_middle_sum
            )

            if middle_scale <= 0:
                continue

            target_counts = (
                ideal_counts.copy()
            )

            target_counts[0] = (
                majority_count
            )

            target_counts[-1] = (
                minority_count
            )

            target_counts[1:-1] *= (
                middle_scale
            )

            # -------------------------------------------------
            # 5.3 The continuous target counts must decrease monotonically
            # -------------------------------------------------
            if np.any(
                target_counts[:-1]
                < target_counts[1:]
                - 1e-12
            ):
                continue

            # Must not be lower than 1
            if np.any(
                target_counts < 1
            ):
                continue

            # Must not exceed the number of available nodes in each class
            if np.any(
                target_counts
                > available_np
                + 1e-12
            ):
                continue

            # -------------------------------------------------
            # 5.4 First take the floor
            # -------------------------------------------------
            counts_sorted = np.floor(
                target_counts + 1e-12
            ).astype(np.int64)

            # Re-fix the head and tail classes
            counts_sorted[0] = (
                majority_count
            )

            counts_sorted[-1] = (
                minority_count
            )

            remaining = int(
                n - counts_sorted.sum()
            )

            if remaining < 0:
                continue

            # -------------------------------------------------
            # 5.5 Deterministic completion using the largest remainder method
            #
            # Each time 1 is added,
            # select the class with the smallest increase in squared error relative to the continuous target.
            # -------------------------------------------------
            while remaining > 0:
                candidates = []

                for i in range(
                    1,
                    n_cls - 1
                ):
                    # Do not exceed the number of available nodes in this class
                    if (
                        counts_sorted[i] + 1
                        > available_np[i]
                    ):
                        continue

                    # After incrementing, the count must not exceed that of the previous class
                    if (
                        counts_sorted[i] + 1
                        > counts_sorted[i - 1]
                    ):
                        continue

                    # To remain as close as possible to the target,
                    # each class can increase at most to ceil(target)
                    target_upper = int(
                        math.ceil(
                            target_counts[i]
                            - 1e-12
                        )
                    )

                    if (
                        counts_sorted[i] + 1
                        > target_upper
                    ):
                        continue

                    old_error = (
                        counts_sorted[i]
                        - target_counts[i]
                    ) ** 2

                    new_error = (
                        counts_sorted[i] + 1
                        - target_counts[i]
                    ) ** 2

                    error_increase = (
                        new_error - old_error
                    )

                    candidates.append(
                        (
                            error_increase,
                            i
                        )
                    )

                if len(candidates) == 0:
                    break

                # Prioritize the candidate with the smallest increase in error;
                # if tied, prioritize the class appearing earlier
                _, selected_class = min(
                    candidates,
                    key=lambda item: (
                        item[0],
                        item[1]
                    )
                )

                counts_sorted[
                    selected_class
                ] += 1

                remaining -= 1

            if remaining != 0:
                continue

            # -------------------------------------------------
            # 5.6 Final feasibility check
            # -------------------------------------------------
            if int(
                counts_sorted.sum()
            ) != n:
                continue

            if (
                counts_sorted[0]
                != rho
                * counts_sorted[-1]
            ):
                continue

            if np.any(
                counts_sorted[:-1]
                < counts_sorted[1:]
            ):
                continue

            if np.any(
                counts_sorted
                > available_np
            ):
                continue

            # -------------------------------------------------
            # 5.7 Evaluate the deviation of the final integer distribution from the exponential distribution
            #
            # Evaluate adjacent-class ratios in log space,
            # because an exponential sequence is linear in log space.
            # -------------------------------------------------
            log_target_ratio = math.log(
                decay_ratio
            )

            log_ratio_error = 0.0

            for i in range(
                n_cls - 1
            ):
                current_ratio = (
                    counts_sorted[i + 1]
                    / counts_sorted[i]
                )

                log_ratio_error += (
                    math.log(current_ratio)
                    - log_target_ratio
                ) ** 2

            # The closer middle_scale is to 1,
            # the smaller the modification to the original exponential sequence
            scale_error = abs(
                math.log(middle_scale)
            )

            score = (
                log_ratio_error,
                scale_error,
                -minority_count
            )

        # Select the feasible solution closest to the exponential long-tailed distribution
        if (
            best_solution is None
            or score < best_score
        ):
            best_solution = (
                counts_sorted.copy()
            )
            best_score = score

    # ---------------------------------------------------------
    # 6. No feasible integer solution
    # ---------------------------------------------------------
    if best_solution is None:
        raise ValueError(
            f"Unable to simultaneously satisfy:\n"
            f"1. The total training-set size is exactly equal to {n};\n"
            f"2. The maximum/minimum class ratio is exactly equal to {rho};\n"
            f"3. Class counts decrease monotonically;\n"
            f"4. Class counts do not exceed the available counts.\n"
            f"Available class counts (large→small): "
            f"{available_np.tolist()}"
        )

    counts_sorted = best_solution

    # ---------------------------------------------------------
    # 7. Final strict validation
    # ---------------------------------------------------------
    final_total = int(
        counts_sorted.sum()
    )

    final_max = int(
        counts_sorted[0]
    )

    final_min = int(
        counts_sorted[-1]
    )

    if final_total != n:
        raise RuntimeError(
            f"Incorrect total training-set size: "
            f"actual={final_total}, target={n}"
        )

    if final_max != rho * final_min:
        raise RuntimeError(
            f"Incorrect head-to-tail ratio: "
            f"max={final_max}, min={final_min}, "
            f"actual={final_max / final_min}, "
            f"target={rho}"
        )

    if np.any(
        counts_sorted[:-1]
        < counts_sorted[1:]
    ):
        raise RuntimeError(
            "Class counts are not monotonically decreasing"
        )

    if np.any(
        counts_sorted > available_np
    ):
        raise RuntimeError(
            "Training class counts exceed the available node counts"
        )

    if np.any(
        counts_sorted <= 0
    ):
        raise RuntimeError(
            "The training set contains an empty class"
        )

    if debug:
        adjacent_ratios = [
            counts_sorted[i + 1]
            / counts_sorted[i]
            for i in range(n_cls - 1)
        ]

        print("\n" + "=" * 100)
        print("Integer long-tailed split with strict total size and head-to-tail ratio")
        print("=" * 100)

        print(
            f"Available class counts (majority→minority): "
            f"{available_np.tolist()}"
        )

        print(
            f"Final training counts (majority→minority): "
            f"{counts_sorted.tolist()}"
        )

        print(
            f"Final training counts (minority→majority): "
            f"{counts_sorted[::-1].tolist()}"
        )

        print(
            f"Total training-set size: "
            f"{final_total}"
        )

        print(
            f"Head-to-tail ratio: "
            f"{final_max}/{final_min}="
            f"{final_max / final_min:.6f}"
        )

        print(
            f"Theoretical adjacent-class decay factor: "
            f"{decay_ratio:.6f}"
        )

        print(
            "Actual adjacent-class ratios: "
            f"{[round(x, 6) for x in adjacent_ratios]}"
        )

        print("=" * 100 + "\n")

    # ---------------------------------------------------------
    # 8. Restore the original class order
    # ---------------------------------------------------------
    counts_sorted_tensor = torch.tensor(
        counts_sorted,
        dtype=torch.long
    )

    class_num_list_train = (
        counts_sorted_tensor[
            inv_indices_cpu
        ]
    )

    return class_num_list_train


def get_classnum_imb_split(target_data, imb_level, shuffle_seed, debug=True):
    """
    Re-split a Dataset object to construct class imbalance.

    Split rules
    ----
    1. The training set contains 10% of all nodes;
    2. The validation set contains 10% of all nodes;
    3. The test set contains the remaining nodes, approximately 80%;
    4. The training set follows a long-tailed class-imbalanced distribution;
    5. The validation set is kept as class-balanced as possible;
    6. The remaining nodes are used as the test set.

    Note
    ----
    This function overwrites the following original fields in target_data:
    train_masks / val_masks / test_masks
    idx_train / idx_val / idx_test
    mask_train / mask_val / mask_test
    unlabel_masks / idx_unlabel / mask_unlabel
    """
    random.seed(shuffle_seed)
    np.random.seed(shuffle_seed)

    device = target_data.device

    labels_cpu = target_data.labels.detach().cpu().long()
    total_items = labels_cpu.shape[0]
    n_classes = int(labels_cpu.max().item()) + 1

    shuffled_indices = list(range(total_items))
    random.shuffle(shuffled_indices)
    shuffled_indices = torch.tensor(shuffled_indices, dtype=torch.long)

    imb_ratio = resolve_imb_ratio(imb_level)

    class_num_list, indices, inv_indices = sort(data=target_data)

    n_train = int(total_items * 0.1)
    n_val = int(total_items * 0.1)

    class_num_list_train = split_lt(
        class_num_list=class_num_list,
        indices=indices,
        inv_indices=inv_indices,
        imb_ratio=imb_ratio,
        n_cls=n_classes,
        n=n_train
    )

    # Keep the validation set as class-balanced as possible
    n_val_per_class = n_val // n_classes
    class_num_list_val = torch.full((n_classes,), n_val_per_class, dtype=torch.long)

    remainder = n_val % n_classes
    if remainder > 0:
        class_num_list_val[:remainder] += 1

    train_indices = []
    val_indices = []
    test_indices = []

    for cls in range(n_classes):
        class_mask = (labels_cpu[shuffled_indices] == cls)
        class_indices = shuffled_indices[class_mask]

        num_samples_in_class = len(class_indices)

        # Number of training samples in the current class
        n_train_samples = int(class_num_list_train[cls].item())
        n_train_samples = min(n_train_samples, num_samples_in_class)

        train_class_indices = class_indices[:n_train_samples]
        train_indices.extend(train_class_indices.tolist())

        remaining_class_indices = class_indices[n_train_samples:]
        num_remaining = len(remaining_class_indices)

        # Number of validation samples in the current class
        n_val_samples = int(class_num_list_val[cls].item())
        n_val_samples = min(n_val_samples, num_remaining)

        val_class_indices = remaining_class_indices[:n_val_samples]
        val_indices.extend(val_class_indices.tolist())

        # Use all remaining samples as the test set
        test_class_indices = remaining_class_indices[n_val_samples:]
        test_indices.extend(test_class_indices.tolist())

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)

    # Check that the three sets do not overlap
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    assert len(set(train_indices) & set(val_indices)) == 0
    assert len(set(train_indices) & set(test_indices)) == 0
    assert len(set(val_indices) & set(test_indices)) == 0

    # Save indices in NumPy format
    target_data.train_masks = train_indices
    target_data.val_masks = val_indices
    target_data.test_masks = test_indices

    # Save indices in PyTorch format
    target_data.idx_train = torch.from_numpy(train_indices).long().to(device)
    target_data.idx_val = torch.from_numpy(val_indices).long().to(device)
    target_data.idx_test = torch.from_numpy(test_indices).long().to(device)

    # This split uses all nodes, so there are no unlabeled nodes
    all_idx = np.arange(total_items)
    used_idx = np.unique(np.concatenate([train_indices, val_indices, test_indices]))
    unlabel_idx = np.setdiff1d(all_idx, used_idx, assume_unique=True)

    target_data.unlabel_masks = unlabel_idx
    target_data.idx_unlabel = torch.from_numpy(unlabel_idx).long().to(device)

    # Save boolean masks
    target_data.mask_train = torch.zeros(total_items, dtype=torch.bool, device=device)
    target_data.mask_val = torch.zeros(total_items, dtype=torch.bool, device=device)
    target_data.mask_test = torch.zeros(total_items, dtype=torch.bool, device=device)
    target_data.mask_unlabel = torch.zeros(total_items, dtype=torch.bool, device=device)

    target_data.mask_train[target_data.idx_train] = True
    target_data.mask_val[target_data.idx_val] = True
    target_data.mask_test[target_data.idx_test] = True

    if len(unlabel_idx) > 0:
        target_data.mask_unlabel[target_data.idx_unlabel] = True

    # Additionally store class counts for experiment logging
    target_data.class_num_list_train = class_num_list_train

    if debug:
        train_counts = torch.bincount(
            target_data.labels[target_data.mask_train].detach().cpu().long(),
            minlength=n_classes
        ).tolist()

        val_counts = torch.bincount(
            target_data.labels[target_data.mask_val].detach().cpu().long(),
            minlength=n_classes
        ).tolist()

        test_counts = torch.bincount(
            target_data.labels[target_data.mask_test].detach().cpu().long(),
            minlength=n_classes
        ).tolist()

        print("\n" + "=" * 100)
        print(f"Class imbalance split result | imb_ratio={imb_ratio}")
        print("=" * 100)
        print(f"Training samples per class: {train_counts}")
        print(f"Validation samples per class: {val_counts}")
        print(f"Test samples per class: {test_counts}")
        print(f"Total training samples: {sum(train_counts)}")
        print(f"Total validation samples: {sum(val_counts)}")
        print(f"Total test samples: {sum(test_counts)}")
        print("=" * 100 + "\n")

    return target_data


def sample_per_class(labels, num_examples_per_class, forbidden_indices=None):
    num_samples = len(labels)
    num_classes = labels.max() + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [np.random.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])

def get_split(labels, train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None,
              train_size=None, val_size=None, test_size=None):
    num_samples = len(labels)
    num_classes = labels.max() + 1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = np.random.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = np.random.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))

    if test_examples_per_class is not None:
        test_indices = sample_per_class(labels, test_examples_per_class, forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = np.random.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    return train_indices, val_indices, test_indices


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=6789)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]   # This step may not be rigorous

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices