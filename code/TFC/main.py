"""Updated implementation for TF-C -- Xiang Zhang, Jan 16, 2023"""

import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger
from model import *
from dataloader import data_generator
from trainer import Trainer




# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()
######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=42, type=int, help='seed value')
parser.add_argument('--backbone_name', default='facebook/dinov3-vits16-pretrain-lvd1689m', type=str)
parser.add_argument('--unfreeze_backbone', action='store_true', help='Fine-tune both DINOv3 backbones (GPU memory intensive)')
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--n_fft', default=64, type=int)
parser.add_argument('--subset', action='store_true', help='Use small subset for debugging')
parser.add_argument('--validation_fraction', default=0.2, type=float,
                    help='Fraction of target train.pt reserved for validation')
parser.add_argument('--epochs', default=None, type=int,
                    help='Override number of epochs (1 for CI smoke tests)')
parser.add_argument('--dinov3_loss_time_weight', default=0.2, type=float,
                    help='Intra-time NT-Xent weight on DINOv3 projected embeddings')
parser.add_argument('--dinov3_loss_frequency_weight', default=0.2, type=float,
                    help='Intra-frequency NT-Xent weight on DINOv3 projected embeddings')
parser.add_argument('--dinov3_loss_cross_weight', default=1.0, type=float,
                    help='Symmetric time-frequency NT-Xent weight')
parser.add_argument('--dinov3_loss_temperature', default=0.2, type=float,
                    help='InfoNCE temperature')
parser.add_argument('--dinov3_finetune_ssl_weight', default=0.1, type=float,
                    help='Weight of self-supervised objective during fine-tuning')

# 1. self_supervised pre_train; 2. finetune (itself contains finetune and test)
parser.add_argument('--training_mode', default='fine_tune_test', type=str,
                    help='pre_train, fine_tune_test')

parser.add_argument('--pretrain_dataset', default='SleepEEG', type=str,
                    help='Dataset of choice: SleepEEG, FD_A, HAR, ECG')
parser.add_argument('--target_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: Epilepsy, FD_B, Gesture, EMG')

parser.add_argument('--logs_save_dir', default='../experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args, unknown = parser.parse_known_args()

if args.device == 'cuda' and not torch.cuda.is_available():
    print('CUDA unavailable: falling back to CPU')
    device = torch.device('cpu')
else:
    device = torch.device(args.device)
print('We are using %s now.' %device)

pretrain_dataset = args.pretrain_dataset
targetdata = args.target_dataset
experiment_description = str(pretrain_dataset) + '_2_' + str(targetdata)


method = 'TF-C'
training_mode = args.training_mode
run_description = args.run_description
logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)
exec(f'from config_files.{pretrain_dataset}_Configs import Config as Configs')
configs = Configs()
if args.epochs is not None:
    if args.epochs < 1:
        parser.error('--epochs must be a positive integer')
    configs.num_epoch = args.epochs
# Expose DINOv3-specific objective coefficients to the training loop.
configs.dinov3_loss_time_weight = args.dinov3_loss_time_weight
configs.dinov3_loss_frequency_weight = args.dinov3_loss_frequency_weight
configs.dinov3_loss_cross_weight = args.dinov3_loss_cross_weight
configs.dinov3_loss_temperature = args.dinov3_loss_temperature
configs.dinov3_finetune_ssl_weight = args.dinov3_finetune_ssl_weight

# # ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}_dinov3_both")
# 'experiments_logs/Exp1/run1/train_linear_seed_0'
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
# 'experiments_logs/Exp1/run1/train_linear_seed_0/logs_14_04_2022_15_13_12.log'
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Pre-training Dataset: {pretrain_dataset}')
logger.debug(f'Target (fine-tuning) Dataset: {targetdata}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
sourcedata_path = f"../../datasets/{pretrain_dataset}"
targetdata_path = f"../../datasets/{targetdata}"
subset = args.subset  # full data by default, use --subset for debugging
pretrain_dl, fine_train_dl, validation_dl, test_dl = data_generator(
    sourcedata_path, targetdata_path, configs, training_mode,
    subset=subset, seed=SEED, validation_fraction=args.validation_fraction
)
logger.debug("Data loaded ...")

# Load Model
"""Here are two models, one basemodel, another is temporal contrastive model"""
TFC_model = TFC(configs, backbone_name=args.backbone_name,
                freeze_backbone=not args.unfreeze_backbone,
                image_size=args.image_size, n_fft=args.n_fft).to(device)
classifier = target_classifier(configs).to(device)
temporal_contr_model = None


if training_mode == "fine_tune_test":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description,
    f"pre_train_seed_{SEED}_dinov3_both", "saved_models"))
    print("The loading file path", load_from)
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device, weights_only=True)
    pretrained_dict = chkpoint["model_state_dict"]
    TFC_model.load_state_dict(pretrained_dict)

model_optimizer = torch.optim.Adam(TFC_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

# Trainer
Trainer(TFC_model, model_optimizer, classifier, classifier_optimizer,
        pretrain_dl, fine_train_dl, validation_dl, test_dl,
        device, logger, configs, experiment_log_dir, training_mode)

logger.debug(f"Training time is : {datetime.now()-start_time}")
