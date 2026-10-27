"""CPU-only synthetic regression tests. Run from code/TFC:
python -m unittest test_validated_pipeline.py -v

No DINOv3 download is needed. This checks split isolation, training-only
augmentations, checkpoint selection and exactly one held-out test evaluation.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from dataloader import data_generator
import trainer


class FakeTFC(nn.Module):
    """Tiny stand-in for the two DINOv3 projected embeddings."""
    def __init__(self):
        super().__init__()
        self.t = nn.Linear(1, 128)
        self.f = nn.Linear(1, 128)

    def forward(self, t, f):
        t_features = self.t(t.mean(dim=(1, 2), keepdim=False).unsqueeze(1))
        f_features = self.f(f.mean(dim=(1, 2), keepdim=False).unsqueeze(1))
        return t_features, t_features, f_features, f_features


def write_dataset(path, count, seed, grouped=False):
    rng = np.random.default_rng(seed)
    label = np.tile(np.arange(2), (count + 1) // 2)[:count]
    signal = rng.standard_normal((count, 1, 178), dtype=np.float32)
    signal += label[:, None, None].astype(np.float32) * 0.3
    dataset = {
        "samples": torch.tensor(signal),
        "labels": torch.tensor(label, dtype=torch.long),
    }
    if grouped:
        dataset["group_ids"] = torch.tensor(np.repeat(np.arange(count // 2), 2))
    torch.save(dataset, path)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "SleepEEG").mkdir()
        (self.root / "Epilepsy").mkdir()
        write_dataset(self.root / "SleepEEG/train.pt", 48, 1)
        write_dataset(self.root / "Epilepsy/train.pt", 60, 2, grouped=True)
        write_dataset(self.root / "Epilepsy/test.pt", 30, 3)
        self.config = SimpleNamespace(
            TSlength_aligned=178, batch_size=8, target_batch_size=8,
            num_classes_target=2, num_epoch=2,
            dinov3_loss_temperature=0.2, dinov3_loss_time_weight=0.2,
            dinov3_loss_frequency_weight=0.2, dinov3_loss_cross_weight=1.0,
            dinov3_finetune_ssl_weight=0.1,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_target_split_and_augmentation(self):
        pre, fine, val, test = data_generator(
            self.root / "SleepEEG", self.root / "Epilepsy",
            self.config, "fine_tune_test", seed=42
        )
        fine_ids = set(fine.dataset.indices)
        val_ids = set(val.dataset.indices)
        self.assertFalse(fine_ids & val_ids)
        self.assertEqual(len(fine_ids) + len(val_ids), 60)
        group_ids = np.repeat(np.arange(30), 2)
        self.assertFalse(set(group_ids[list(fine_ids)]) & set(group_ids[list(val_ids)]))
        original, _, train_aug, freq, freq_aug = fine.dataset[0]
        self.assertFalse(torch.allclose(original, train_aug))
        for eval_loader in (val, test):
            x, _, aug_t, wave, aug_f = eval_loader.dataset[0]
            self.assertTrue(torch.equal(x, aug_t))
            self.assertTrue(torch.equal(x, wave))
            self.assertTrue(torch.equal(x, aug_f))

    def test_heldout_test_is_called_once(self):
        pre, fine, val, test = data_generator(
            self.root / "SleepEEG", self.root / "Epilepsy",
            self.config, "fine_tune_test", seed=42
        )
        model = FakeTFC()
        classifier = nn.Linear(256, 2)
        optim_model = torch.optim.Adam(model.parameters(), lr=0.002)
        optim_classifier = torch.optim.Adam(classifier.parameters(), lr=0.002)
        calls = [0]
        evaluate = trainer._evaluate

        def counted_evaluate(model_, classifier_, loader, device, classes):
            if loader is test:
                calls[0] += 1
            return evaluate(model_, classifier_, loader, device, classes)

        class Logger:
            def debug(self, message):
                pass

        with patch.object(trainer, "_evaluate", side_effect=counted_evaluate):
            trainer.Trainer(
                model, optim_model, classifier, optim_classifier,
                pre, fine, val, test, torch.device("cpu"),
                Logger(), self.config, self.root / "results", "fine_tune_test"
            )
        self.assertEqual(calls[0], 1)
        self.assertTrue((self.root / "results/saved_models/best_validation.pt").is_file())
        self.assertTrue((self.root / "results/evaluation.json").is_file())
        self.assertTrue((self.root / "results/history.csv").is_file())
        self.assertTrue((self.root / "results/confusion_matrix.csv").is_file())


if __name__ == "__main__":
    unittest.main()
