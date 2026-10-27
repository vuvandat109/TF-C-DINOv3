"""Dataloaders for TF-C + DINOv3 with a held-out validation set.

Target train.pt is split BEFORE data augmentation. Target test.pt is never
used to select a checkpoint or tune hyperparameters. Use subject/recording
identifiers when available to keep related windows in the same partition.
"""
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

GROUP_KEYS = ("source_ids", "group_ids", "recording_ids", "subject_ids", "parent_ids")


def _load_series(path, config):
    dataset = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(dataset, dict) or not {"samples", "labels"} <= dataset.keys():
        raise ValueError(f"{path}: expected a dict with 'samples' and 'labels'")
    signals = torch.as_tensor(dataset["samples"], dtype=torch.float32)
    labels = torch.as_tensor(dataset["labels"], dtype=torch.long).flatten()
    if signals.ndim == 2:
        signals = signals.unsqueeze(1)
    elif signals.ndim == 3 and signals.shape[1] > signals.shape[2]:
        signals = signals.transpose(1, 2)
    if signals.ndim != 3 or len(signals) != len(labels):
        raise ValueError(f"{path}: incompatible signals/labels shapes")
    required_length = int(config.TSlength_aligned)
    if signals.shape[-1] < required_length:
        raise ValueError(f"{path}: need at least {required_length} time steps")
    # Retain the original one-channel TF-C benchmark convention.
    signals = signals[:, :1, :required_length].contiguous()
    groups = None
    for name in GROUP_KEYS:
        if name in dataset:
            value = dataset[name]
            groups = np.asarray(value).reshape(-1)
            if len(groups) != len(labels):
                raise ValueError(f"{path}: {name} length mismatch")
            break
    if len(labels) < 2:
        raise ValueError(f"{path}: fewer than two samples")
    return signals, labels, groups


def _subset_indices(indices, labels, max_count, seed):
    if len(indices) <= max_count:
        return np.asarray(indices, dtype=int)
    indices = np.asarray(indices, dtype=int)
    counts = np.unique(labels[indices].numpy(), return_counts=True)[1]
    classes = len(counts)
    use_stratify = counts.min() >= 2 and max_count >= classes
    try:
        selected, _ = train_test_split(
            indices, train_size=max_count, random_state=seed,
            stratify=labels[indices].numpy() if use_stratify else None
        )
    except ValueError:
        selected, _ = train_test_split(indices, train_size=max_count, random_state=seed)
    return np.sort(selected)


def _split_target(labels, groups, validation_fraction, seed):
    indices = np.arange(len(labels))
    if len(indices) < 4:
        raise ValueError("Need at least four target training samples")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if groups is not None:
        if len(np.unique(groups)) < 2:
            raise ValueError("Group-based validation needs at least two groups")
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=validation_fraction, random_state=seed
        )
        train, validation = next(splitter.split(indices, labels.numpy(), groups=groups))
        if set(groups[train]) & set(groups[validation]):
            raise AssertionError("Leaked group between train and validation")
    else:
        warnings.warn(
            "No subject/recording IDs found; record-level split cannot rule "
            "out leakage from overlapping or augmented source windows.",
            RuntimeWarning
        )
        target = labels.numpy()
        classes, counts = np.unique(target, return_counts=True)
        n_validation = int(np.ceil(len(target) * validation_fraction))
        stratify = (
            target if counts.min() >= 2 and n_validation >= len(classes)
            and len(target) - n_validation >= len(classes) else None
        )
        train, validation = train_test_split(
            indices, test_size=validation_fraction, random_state=seed,
            stratify=stratify
        )
    if len(train) < 2 or len(validation) < 1:
        raise ValueError("Train/validation partitions too small")
    return np.sort(train), np.sort(validation)


class Load_Dataset(Dataset):
    """Return (time, label, augmented_time, freq_wave, augmented_freq_wave).

    DINOv3's frequency adapter generates STFT from the waveform internally.
    Augmentation is stochastic ONLY for training datasets.
    """
    def __init__(self, signals, labels, augment=False):
        self.signals, self.labels, self.augment = signals, labels, augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        signal, label = self.signals[index], self.labels[index]
        if not self.augment:
            return signal, label, signal, signal, signal

        std = signal.std(unbiased=False).clamp_min(1e-3)
        gain = 1 + 0.05 * torch.randn((), dtype=signal.dtype)
        time_aug = signal * gain + 0.02 * std * torch.randn_like(signal)

        spectrum = torch.fft.rfft(signal, dim=-1)
        keep = torch.rand(spectrum.shape, device=signal.device) > 0.10
        freq_aug = torch.fft.irfft(spectrum * keep, n=signal.shape[-1], dim=-1)
        return signal, label, time_aug, signal, freq_aug


def data_generator(sourcedata_path, targetdata_path, configs, training_mode,
                   subset=False, seed=42, validation_fraction=0.2):
    """Return four loaders: pretrain, fine_train, validation, held-out test."""
    src_x, src_y, _ = _load_series(Path(sourcedata_path) / "train.pt", configs)
    train_x, train_y, groups = _load_series(Path(targetdata_path) / "train.pt", configs)
    test_x, test_y, _ = _load_series(Path(targetdata_path) / "test.pt", configs)
    train_ids, val_ids = _split_target(train_y, groups, validation_fraction, seed)

    if subset:
        src_ids = _subset_indices(np.arange(len(src_y)), src_y,
                                  int(configs.batch_size) * 10, seed)
        src_x, src_y = src_x[src_ids], src_y[src_ids]
        train_ids = _subset_indices(train_ids, train_y,
                                    int(configs.target_batch_size) * 10, seed)
        val_ids = _subset_indices(val_ids, train_y,
                                  int(configs.target_batch_size) * 2, seed)
        test_ids = _subset_indices(np.arange(len(test_y)), test_y,
                                   int(configs.target_batch_size) * 10, seed)
        test_x, test_y = test_x[test_ids], test_y[test_ids]

    pretrain_set = Load_Dataset(src_x, src_y, augment=True)
    fine_train_set = Subset(Load_Dataset(train_x, train_y, augment=True), train_ids.tolist())
    validation_set = Subset(Load_Dataset(train_x, train_y), val_ids.tolist())
    test_set = Load_Dataset(test_x, test_y)

    def create_loader(dataset, batch_size, shuffle):
        return DataLoader(
            dataset, batch_size=min(batch_size, len(dataset)),
            shuffle=shuffle, drop_last=False, num_workers=0,
            generator=torch.Generator().manual_seed(seed)
        )

    print(
        f"Split sizes | pretrain={len(pretrain_set)}, "
        f"fine_train={len(fine_train_set)}, validation={len(validation_set)}, "
        f"test={len(test_set)}"
    )
    return (
        create_loader(pretrain_set, configs.batch_size, True),
        create_loader(fine_train_set, configs.target_batch_size, True),
        create_loader(validation_set, configs.target_batch_size, False),
        create_loader(test_set, configs.target_batch_size, False)
    )
