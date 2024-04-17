import os
import yaml
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from main import pipeline
from utils import override_yaml

def masking_and_epochs(max_epochs=10):
    best_ICM = 0
    best_epochs = 0
    best_masking_percentage = 0

    for epochs in range(1, max_epochs+1):
        for masking_percentage in np.arange(0.1, 1.1, 0.1):
            print(f"Evaluating model with {epochs} epochs and masking percentage {masking_percentage}...")
            config.training_settings['epochs'] = epochs
            config.model_conf['modality_masking_prob'] = masking_percentage

            val_output = pipeline(args, config)

        # Check if this model's accuracy is better than the previous best
        if val_output['icm-norm'] > best_ICM:
            best_ICM = val_output['icm-norm']
            best_epochs = epochs
            best_masking_percentage = masking_percentage

        # Save the best hyperparameters to a text file
        with open("./best_hyperparameters_masking_epochs.txt", "a") as file:
            file.write(f"Epochs: {epochs}\n")
            file.write(f"Masking Percentage: {masking_percentage}\n")
            file.write(f"ICM-norm: {val_output['icm-norm'] }\n")

    # Save the best hyperparameters to a text file
    with open("./best_hyperparameters_masking_epochs.txt", "a") as file:
        file.write(f"Best Hyperparameters:\n")
        file.write(f"Epochs: {best_epochs}\n")
        file.write(f"Masking Percentage: {best_masking_percentage}\n")
        file.write(f"ICM-norm: {best_ICM}\n")

if __name__ == "__main__":

    # -- setting seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Training and/or evaluation of models for EXIST2024.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', required=True, type=str, help='Configuration file to build, train, and evaluate the model')
    parser.add_argument('--training-dataset', required=True, type=str, help='CSV file representing the training dataset')
    parser.add_argument('--validation-dataset', required=True, type=str, help='CSV file representing the validation dataset')
    parser.add_argument('--mode', default='both', type=str, help='Choose between: "training", "evaluation", or "both"')
    parser.add_argument('--load-checkpoint', default='', type=str, help='Choose between: "training", "evaluation", or "both"')
    parser.add_argument("--yaml-overrides", metavar="CONF:[KEY]:VALUE", nargs='*', help="Set a number of conf-key-value pairs for modifying the yaml config file on the fly.")
    parser.add_argument("--use-modalities", nargs='+', default=['all_modalities'], help="It allows you to choose which modalities will be used.")
    parser.add_argument('--save', default=1, type=int, help='Do you want to save checkpoints and output reports?')
    parser.add_argument('--output-dir', required=True, type=str, help='Path where to save model checkpoints and predictions')
    parser.add_argument('--output-name', required=True, type=str, help='Choose between: "validation", or "test"')

    args = parser.parse_args()

    # -- loading configuration file
    config_file = Path(args.config)
    with config_file.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config = override_yaml(config, args.yaml_overrides)
    config = argparse.Namespace(**config)

    if 'all_modalities' not in args.use_modalities:
        new_modalities = []
        for modality in config.modalities:
            if modality['name'] in args.use_modalities:
                new_modalities.append(modality)

        config.modalities = new_modalities
        config.use_modalities = args.use_modalities

    assert len(config.modalities) > 0, f'Ensure you specified the modalities you expected to use'

    ###
    masking_and_epochs()
