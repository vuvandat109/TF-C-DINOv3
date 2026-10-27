# Real DINOv3 + SleepEEG/Epilepsy smoke test

The repository now offers a **manual** GitHub Actions workflow that uses
real public benchmark data and real Hugging Face pretrained DINOv3 weights.

- Workflow: [Real pretrained DINOv3 - SleepEEG to Epilepsy smoke run](.github/workflows/real-dinov3-smoke.yml)
- Data downloader: [download_real_benchmarks.py](code/TFC/download_real_benchmarks.py)
- Results: workflow logs and an artifact with evaluation metrics (no checkpoints)

## Required GitHub secret

The model is gated. Approval on your Hugging Face account is **not** forwarded
to a GitHub runner. Create a Hugging Face **read-only** access token on
https://huggingface.co/settings/tokens using the approved account. Set it as
GitHub repository Actions secret `HF_TOKEN` using:

https://github.com/vuvandat109/TF-C-DINOv3/settings/secrets/actions

Do not paste the token in chat, code, issues, logs, or commits.

## How to run

1. Open https://github.com/vuvandat109/TF-C-DINOv3/actions/workflows/real-dinov3-smoke.yml
2. Select **Run workflow** on branch `main`.
3. Wait for the GitHub Actions run to finish; read errors in the run's log.
4. Download artifact `real-dinov3-one-epoch-results` and review `evaluation.json`.

The workflow downloads the public Figshare archives, unpacks required .pt files,
loads gated pretrained weights using `HF_TOKEN`, exercises two real encoders,
pretrains the projectors for **one epoch** on the `--subset` source dataset,
then fine-tunes and evaluates on a subset of Epilepsy. Results are diagnostic,
**not** an accuracy benchmark, and cannot be compared with previously reported
full-dataset results. The job runs on GitHub-hosted CPU and may exceed runner
time limits due to two pretrained DINOv3 models; use a compatible GPU runner
for full training.

If a Figshare ZIP has a different archive structure from expected, the
downloader fails loudly rather than generating synthetic data. The dataset
download script in the repository references these same Figshare sources.

For reproducible full experimentation, execute the existing documented
commands (without `--subset` and without `--epochs 1`) on an appropriately
sized GPU and evaluate multiple random seeds, reporting test accuracy/F1 only
after selection by validation. The full experiment is **not** performed by
this manual smoke workflow.
