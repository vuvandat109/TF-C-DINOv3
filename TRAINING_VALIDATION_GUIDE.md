# TF-C + DINOv3 — validated training and evaluation

This update changes the evaluation protocol, **not** the two DINOv3 encoders
or the DINOv3 contrastive objective.

## What changed

- `code/TFC/dataloader.py`: separate target `train.pt` into disjoint
  fine-tuning and validation sets, *before* training-time augmentation.
  The target `test.pt` remains separate. If `train.pt` includes one of
  `source_ids`, `group_ids`, `recording_ids`, `subject_ids`, or
  `parent_ids`, the validation split keeps whole groups together.
  Otherwise the code warns that overlapping windows cannot be ruled out.
- `code/TFC/trainer.py`: use actual augmented time/frequency waveforms
  during fine-tuning; select the checkpoint using validation macro F1
  (validation loss breaks ties). Compute metrics over all evaluated samples.
  Do **not** read the target test set until the best checkpoint is restored.
  The KNN baseline trains only on fine-tuning train embeddings.
- `code/TFC/main.py`: pass four loaders; honor `--device`, `--seed`,
  and `--validation_fraction`.
- `code/TFC/test_validated_pipeline.py`: synthetic CPU-only regression tests.

## Setup

From the repository root:

```powershell
python -m pip install -r code/TFC/requirements-dinov3.txt
python -m pip install matplotlib
cd code/TFC
python -m unittest test_validated_pipeline.py -v
python test_dinov3_loss.py
python smoke_test_dinov3.py
```

The synthetic tests do not download or execute real DINOv3 weights. Full
training requires Hugging Face access to the DINOv3 backbone.

Expected dataset layout, relative to the repository root:

```text
datasets/
  SleepEEG/train.pt
  Epilepsy/train.pt
  Epilepsy/test.pt
```

Each `.pt` file must hold the tensors `samples` and `labels`. For
subject-safe evaluation, include matching *tensor* group IDs in the target
training data. Ensure that the **original** target train and test files
already separate subjects/source recordings where the dataset requires it:
creating a validation split cannot repair leakage across those two files.

## Commands

Execute from `code/TFC`, in order:

```powershell
python main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epilepsy --device cpu --subset
python main.py --training_mode fine_tune_test --pretrain_dataset SleepEEG --target_dataset Epilepsy --device cpu --subset
```

When this small-data smoke run succeeds, repeat both commands without
`--subset` for a full experiment. Run across multiple seeds, e.g.
`--seed 42`, `--seed 123`, `--seed 2026`, and present mean/std.
The paired pretrain and fine-tune runs MUST use matching seed and
`--run_description` so the fine-tune step loads the right checkpoint.
Using `--unfreeze_backbone` trains both DINOv3 backbones and consumes
substantially more GPU memory. Running on CPU is possible but slow.

## Outputs

Each fine-tune run writes to:

`code/experiments_logs/SleepEEG_2_Epilepsy/<run_description>/fine_tune_test_seed_<seed>_dinov3_both/`

- `saved_models/best_validation.pt`
- `evaluation.json`: validation-selected epoch and final held-out test metrics
- `history.csv`: epoch-level train loss, validation loss, validation accuracy/F1
- `test_predictions.csv`, `confusion_matrix.csv`
- `loss_curve.png`, `validation_f1.png`, `confusion_matrix.png`
  (when matplotlib is installed)

The Pre-training checkpoint is under the corresponding
`pre_train_seed_<seed>_dinov3_both/saved_models/ckp_last.pt`.

**Research caveats:** Test only once per run; do not pick a run/model by its
test score. A subset run is a functional test, not a benchmark. Compare to
the original TF-C with the same split, preprocessing, seed and evaluation
protocol before claiming an improvement. The repository does not include
training data or pretrained weight downloads; no new experimental accuracy
is claimed by this change.
