**# TF-C + DINOv3 — Time Series Representation Learning through Time–Frequency Contrastive Learning**

**\*\*Status according to the experimental report:\*\*** Tested two actual pretrained DINOv3 encoders; completed 20 epochs of Pre-training, 20 epochs of Fine-tuning, and Testing on the SleepEEG → Epilepsy subset.

**## 1. Project Overview**

The project experiments with replacing the two Transformer Encoders in TF-C (Time-Frequency Consistency) with two independent pretrained DINOv3 ViT-S/16 backbones:

\- One backbone processes the time domain.

\- One backbone processes the frequency domain.

Since DINOv3 is designed to receive images, one-dimensional EEG signals are converted into feature images before being passed to the encoders. The features are combined to perform Epilepsy classification.

**### Objectives**

\- Replace the two TF-C encoders with two independent pretrained DINOv3 encoders.

\- Convert EEG signals into time-domain images and frequency spectrum images.

\- Implement and use the new Loss Function in Pre-training and Fine-tuning.

\- Test the model structure, gradients, training, and evaluation on real data.

**### Main Files**

\| Component | Source Code File |

\| --- | --- |

\| Two DINOv3 encoders, projectors, and classifier | [\`code/TFC/model.py\`]\(code/TFC/model.py) |

\| Signal-to-image conversion | [\`code/TFC/dinov3_adapter.py\`]\(code/TFC/dinov3_adapter.py) |

\| New Loss Function | [\`code/TFC/dinov3_loss.py\`]\(code/TFC/dinov3_loss.py) |

\| Loss integration module | [\`code/TFC/loss.py\`]\(code/TFC/loss.py) |

\| Loss usage during Pre-training/Fine-tuning | [\`code/TFC/trainer.py\`]\(code/TFC/trainer.py) |

\| Configuration parameters and main program | [\`code/TFC/main.py\`]\(code/TFC/main.py) |

\| Loss and gradient testing | [\`code/TFC/test_dinov3_loss.py\`]\(code/TFC/test_dinov3_loss.py) |

\| Encoder testing with a mock backbone | [\`code/TFC/smoke_test_dinov3.py\`]\(code/TFC/smoke_test_dinov3.py) |

\| Testing two actual pretrained encoders | [\`code/TFC/test_two_real_dinov3.py\`]\(code/TFC/test_two_real_dinov3.py) |

**## 2. System Architecture**

\| Stage | Time Branch | Frequency Branch |

\| --- | --- | --- |

\| Input | EEG signal \`[B, 1, 178]\` | EEG signal \`[B, 1, 178]\` |

\| Conversion using SignalImageAdapter | Time-domain image \`[B, 3, 224, 224]\` | STFT spectrum image \`[B, 3, 224, 224]\` |

\| Backbone | DINOv3 Encoder T | DINOv3 Encoder F |

\| Backbone features | 384 dimensions | 384 dimensions |

\| Projector | Projector T | Projector F |

\| Features after the projector | \`z_t\`: 128 dimensions | \`z_f\`: 128 dimensions |

**### Architecture Details**

\- **\*\*Input:\*\*** EEG tensor of shape \`[B, 1, 178]\`; \`B\` is the number of samples in the batch.

\- **\*\*Encoder input images:\*\*** Each image has 3 channels and a size of \`224 × 224\`.

\- **\*\*Encoder:\*\*** Each DINOv3 ViT-S/16 outputs a 384-dimensional vector; the two encoders are separate instances and do not share parameters.

\- **\*\*Projector:\*\*** Each branch maps from 384 to 128 dimensions.

\- **\*\*Classifier:\*\*** Receives a concatenated 256-dimensional vector, using MLP or KNN in the evaluation experiment.

\- **\*\*Environment:\*\*** Python running locally.

**### 2.1. New Loss Function — DINOv3TFCLoss**

The loss function uses the outputs of the trainable projectors (\`z_t\`, \`z_f\`). The notation \`\_aug\` denotes the representation of a data-augmented sample.

\`\`\`text

L_time = NT-Xent(z_t, z_t_aug)

L_freq = NT-Xent(z_f, z_f_aug)

L_cross = mean(

    NT-Xent(z_t, z_f),

    NT-Xent(z_t, z_f_aug),

    NT-Xent(z_t_aug, z_f),

    NT-Xent(z_t_aug, z_f_aug)

)

L_pretrain = 0.2 \* L_time + 0.2 \* L_freq + 1.0 \* L_cross

L_finetune = CrossEntropy + 0.1 \* L_pretrain

Temperature = 0.2

\`\`\`

**## 3. Environment and Data**

**### 3.1. Environment**

\| Component | Requirements / Information |

\| --- | --- |

\| Operating system used | Windows, PowerShell in VS Code |

\| Python | Python 3, using \`.venv\`; check the version with \`python --version\` |

\| PyTorch | \`torch>=2.2\` |

\| Transformers | \`transformers>=4.56.0\` |

\| Hugging Face Hub | \`huggingface_hub>=0.34\` |

\| NumPy | \`numpy>=1.24\` |

\| scikit-learn | \`scikit-learn>=1.3\` |

\| Backbone | \`facebook/dinov3-vits16-pretrain-lvd1689m\` |

\| Tested device | CPU; speed depends on the machine configuration |

**### 3.2. Required Data**

Prepare the following files relative to the repository root directory:

\`\`\`text

datasets/

├── SleepEEG/

│   └── train.pt

└── Epilepsy/

    ├── train.pt

    └── test.pt

\`\`\`

The \`.pt\` files must have a structure compatible with [\`code/TFC/dataloader.py\`]\(code/TFC/dataloader.py).

**## 4. Installation on Windows**

**### Step 1 — Clone the Repository and Open the Project Directory**

\`\`\`powershell

git clone https\://github.com/vuvandat109/TF-C-DINOv3.git

cd TF-C-DINOv3

\`\`\`

The commands in Section 4 are run from the \`TF-C-DINOv3\` root directory.

**### Step 2 — Create and Activate the Virtual Environment**

\`\`\`powershell

python -m venv .venv

.\\.venv\Scripts\Activate.ps1

\`\`\`

If PowerShell does not allow the activation script to run, execute:

\`\`\`powershell

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

.\\.venv\Scripts\Activate.ps1

\`\`\`

**### Step 3 — Install Libraries**

\`\`\`powershell

python -m pip install -r .\code\TFC\requirements-dinov3.txt

\`\`\`

Check versions:

\`\`\`powershell

python --version

python -c "import torch, transformers; print('PyTorch:', torch.\_\_version\_\_); print('Transformers:', transformers.\_\_version\_\_)"

\`\`\`

**### Step 4 — Check the Data**

Run from the repository root directory:

\`\`\`powershell

Test-Path .\datasets\SleepEEG\train.pt

Test-Path .\datasets\Epilepsy\train.pt

Test-Path .\datasets\Epilepsy\test.pt

\`\`\`

All three commands should return:

\`\`\`text

True

\`\`\`

If any result is \`False\`, check the location and names of the data files before continuing.

**### Step 5 — Obtain Access to DINOv3 Weights**

Open the model page:

[facebook/dinov3-vits16-pretrain-lvd1689m on Hugging Face]\(https\://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)

Complete the provider's access requirements and ensure that the account has permission to download the weights.

Then log in using the corresponding account:

\`\`\`powershell

hf auth login

\`\`\`

**## 5. Running and Testing**

The Python commands in this section must be run from the \`code/TFC\` directory.

From the repository root directory, execute:

\`\`\`powershell

cd .\code\TFC

\`\`\`

**### 5.1. Test the Loss Function**

\`\`\`powershell

python test_dinov3_loss.py

\`\`\`

Expected result:

\`\`\`text

PASS: DINOv3-TF-C loss, positive pairing and gradients in all four projected views

\`\`\`

The test uses mock tensors to check loss finiteness, positive pairs, and projector gradients. This test does not demonstrate the performance of the pretrained model.

**### 5.2. Test the Two-Encoder Structure**

\`\`\`powershell

python smoke_test_dinov3.py

\`\`\`

Expected result:

\`\`\`text

PASS: both DINOv3 encoder slots replaced, adapters, output shapes and projector gradients

\`\`\`

**### 5.3. Test Two Actual Pretrained DINOv3 Encoders**

\`\`\`powershell

python test_two_real_dinov3.py

\`\`\`

Results recorded in the report:

\`\`\`text

EEG input: torch.Size([2, 1, 178])

Independent encoders: True

Time features: torch.Size([2, 384])

Time projector: torch.Size([2, 128])

Frequency features: torch.Size([2, 384])

Frequency projector: torch.Size([2, 128])

Combined: torch.Size([2, 256])

Finite: True

SUCCESS: Both real DINOv3 encoders passed!

\`\`\`

**### 5.4. Pre-training with the New Loss Function**

\`\`\`powershell

python main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epilepsy --device cpu --subset

\`\`\`

Signs of correct execution:

\`\`\`text

Mode: pre_train

Pretraining on source dataset

DINOv3-TF-C pretrain | total=... | L_time=... | L_freq=... | L_cross=...

Pre-training Epoch : 1 | Train Loss : ...

\`\`\`

Reported configuration:

\| Parameter | Value |

\| --- | --- |

\| Number of epochs | 20 |

\| Learning rate | \`3e-4\` |

\| Batch size | 8 |

\| Seed | 42 |

\| Quick-test data | 80 SleepEEG samples when using \`--subset\` |

**\*\*Note:\*\*** According to the current source code description, \`main.py\` automatically prioritizes CUDA when available. Therefore, \`--device cpu\` does not guarantee CPU execution if the machine has a CUDA GPU.

**#### Pre-training Checkpoint Location**

Path relative to the repository root directory:

\`\`\`text

code/experiments_logs/SleepEEG_2_Epilepsy/run1/pre_train_seed_42_dinov3_both/saved_models/ckp_last.pt

\`\`\`

**### 5.5. Fine-tuning and Testing**

Run only after Pre-training has saved a new checkpoint.

\`\`\`powershell

python main.py --training_mode fine_tune_test --pretrain_dataset SleepEEG --target_dataset Epilepsy --device cpu --subset

\`\`\`

The reported \`--subset\` configuration uses:

\- 60 Fine-tuning samples.

\- 600 Testing samples.

Signs of correct execution:

\`\`\`text

Mode: fine_tune_test

Fine-tune on Fine-tuning set

Epoch : 1

Finetune: loss = ... | Acc=... | F1=...

MLP Testing: Acc=... | F1=...

KNN Testing: Acc=... | F1=...

...

Best Testing Performance: ...

Best KNN F1 ...

Training is Done!

\`\`\`

**### 5.6. Result Locations**

The following paths are relative to the repository root directory:

\| File / Directory | Purpose |

\| --- | --- |

\| \`code/experiments_logs/SleepEEG_2_Epilepsy/\` | Experiment log directory |

\| \`code/experiments_logs/SleepEEG_2_Epilepsy/run1/pre_train_seed_42_dinov3_both/saved_models/ckp_last.pt\` | Pre-training checkpoint |

\| \`code/TFC/experiments_logs/finetunemodel/sleepedf2eplipsy_model.pt\` | Fine-tuning checkpoint |

\| \`code/TFC/experiments_logs/finetunemodel/sleepedf2eplipsy_classifier.pt\` | MLP classifier checkpoint |

**## 6. Experimental Results**

The following figures were recorded from the report of a trial run on a subset; they are not results from running on the full dataset.

**### 6.1. Functional Test Results**

\| Test | Reported Result |

\| --- | --- |

\| Loss Function and gradients — \`test_dinov3_loss.py\` | PASS |

\| Mock encoder architecture — \`smoke_test_dinov3.py\` | PASS |

\| Two actual pretrained DINOv3 encoders — \`test_two_real_dinov3.py\` | SUCCESS |

**### 6.2. Pre-training — 20 Epochs**

\| Metric | Value |

\| --- | --- |

\| Dataset | SleepEEG, \`--subset\`: 80 samples |

\| Completed epochs | 20/20 |

\| Epoch 1 loss | 3.2115 |

\| Epoch 20 loss | 0.2784 |

\| Epoch 20 \`L_time\` | 0.1390 |

\| Epoch 20 \`L_freq\` | 0.1775 |

\| Epoch 20 \`L_cross\` | 0.2151 |

\| Runtime | \`00:23:57.778917\` |

\| Checkpoint | \`ckp_last.pt\` — reported as verified to exist |

Loss decreased from **\*\*3.2115\*\*** to **\*\*0.2784\*\*** in the reported run.

**### 6.3. Fine-tuning and Testing — 20 Epochs**

The experiment uses Epilepsy with 60 Fine-tuning samples and 600 Testing samples when \`--subset\` is enabled.

\| Metric | MLP Testing | KNN Testing |

\| --- | --- | --- |

\| Accuracy | 18.50% | 90.33% |

\| Precision | 9.25% | 94.70% |

\| Recall | 50.00% | 73.87% |

\| F1-score | 15.61% | 79.52% |

\| AUROC | 67.83% | 91.21% |

\| AUPRC | 72.78% | 89.79% |

**### Discussion of Results**

\- KNN achieved high Accuracy in the trial run, but results on a subset are insufficient to conclude that the model generalizes well.

\- MLP achieved low Accuracy and showed almost no improvement from epoch 2 to epoch 20. The Fine-tuning process, label distribution, and checkpoint loading logic need to be checked before the final evaluation.

\- The logs contain \`UndefinedMetricWarning\` because a class was not predicted. The predictions and metric calculations need to be examined in detail.

\- The DINOv3 variant cannot yet be claimed to outperform the original TF-C without a controlled comparison using the same dataset and conditions.

**## 7. AI Model and Limitations**

**### 7.1. Model Information**

\| Attribute | Description |

\| --- | --- |

\| Model | TF-C + two DINOv3 ViT-S/16 encoders |

\| Pretrained backbone | \`facebook/dinov3-vits16-pretrain-lvd1689m\` |

\| Loss Function | \`DINOv3TFCLoss\` |

\| Experimental input | EEG \`[B, 1, 178]\` |

\| Encoder outputs | Two 384-dimensional vectors |

\| Projector outputs | Two 128-dimensional vectors |

\| Classifier input | Concatenated 256-dimensional vector |

\| MLP classifier output | Two-class logits in the described Epilepsy configuration |

\| Experimental configuration | Seed 42, \`--subset\`, 20 epochs |

\| Deployment | Runs locally using Python |

**### 7.2. Limitations**

\- DINOv3 is pretrained on images. Converting EEG into images is an experimental method that has not been demonstrated to be optimal.

\- According to the experiment description, \`dataloader.py\` takes the first signal channel.

\- The two DINOv3 backbones are frozen by default; the subsequent trainable layers are optimized at each stage.

\- Full evaluation on the entire dataset, across multiple seeds, or through a controlled comparison with the original TF-C has not yet been performed.

\- The experimental repository does not yet provide a prediction API, confidence threshold, or fallback for end users.

**## 8. Usage**

The project is intended for researchers and developers.

After installation:

1\. Run the tests in Sections 5.1–5.3.

2\. Run **\*\*Pre-training → Fine-tuning → Testing\*\*** in the order described in Sections 5.4–5.5.

3\. Read the logs and monitor Loss, Accuracy, Precision, Recall, F1, AUROC, and AUPRC in the Terminal.

4\. Use the checkpoint for subsequent experiments with the same architecture and a compatible configuration.

**### Check the Pre-training Checkpoint**

From the \`code/TFC\` directory, run:

\`\`\`powershell

Test-Path "..\experiments_logs\SleepEEG_2_Epilepsy\run1\pre_train_seed_42_dinov3_both\saved_models\ckp_last.pt"

\`\`\`

Expected result after Pre-training has successfully saved the checkpoint:

\`\`\`text

True

\`\`\`

**## 9. Troubleshooting**

\| Symptom | Common Cause | Solution |

\| --- | --- | --- |

\| \`ModuleNotFoundError: transformers\` | Libraries have not been installed in the correct Python environment | Activate \`.venv\` and reinstall \`requirements-dinov3.txt\` |

\| \`401\` / \`403\` errors when downloading DINOv3 | Access has not been granted or the wrong account is logged in | Complete the access requirements on Hugging Face and run \`hf auth login\` using the authorized account |

\| \`FileNotFoundError\` related to \`train.pt\` | Missing data or an incorrect path | Check the three \`.pt\` files and run \`main.py\` from \`code/TFC\` |

\| \`ckp_last.pt\` not found | Pre-training has not been run or the experiment directory does not match | Run Pre-training first; check the dataset, seed, and run name |

\| Epochs take a long time | DINOv3 and STFT use substantial resources | Monitor Task Manager; use \`--subset\` for a quick test or consider a GPU |

\| \`UndefinedMetricWarning\` | A class is not predicted by the classifier | Check predictions, label distribution, and the confusion matrix |

\| MLP Accuracy is low and does not improve | The model has not converged or there is an issue in the training/evaluation flow | Check \`trainer.py\`, the checkpoint, optimizer, and data |

\| \`can't open file ...main.py\` | The Terminal is in the wrong directory | From the root directory, run \`cd .\code\TFC\` before calling \`python main.py\` |

**## 10. Experiment Reproducibility and References**

**### 10.1. Reproducibility Configuration**

For others to reproduce the experiment, preserve the following:

\- Source code.

\- Data version and dataset splits.

\- Experimental configuration.

\- Pretrained weights.

\- Seed.

\- Library environment.

Reported configuration:

\| Parameter | Value |

\| --- | --- |

\| Seed | 42 |

\| Learning rate | \`3e-4\` |

\| Pre-training batch size | 8 |

\| Number of Pre-training epochs | 20 |

\| Number of Fine-tuning epochs | 20 |

\| \`L_time\` weight | 0.2 |

\| \`L_freq\` weight | 0.2 |

\| \`L_cross\` weight | 1.0 |

\| SSL coefficient during Fine-tuning | 0.1 |

\| Temperature | 0.2 |

To record library versions in the currently activated environment:

\`\`\`powershell

python -m pip freeze > environment_versions.txt

\`\`\`

**### 10.2. References**

\- **\*\*TF-C:\*\*** Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency, NeurIPS 2022.

\- **\*\*DINOv3 pretrained:\*\*** [facebook/dinov3-vits16-pretrain-lvd1689m]\(https\://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).