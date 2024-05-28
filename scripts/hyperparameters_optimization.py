import os
import yaml
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from main import pipeline
from utils import override_yaml

def complete():
    best_ICM = 0
    best_lr = 0
    best_attnheads = 0
    best_epochs = 0
    best_masking = 0

    for lr in [0.0002, 0.0005]:
        for layers in [1, 2, 4]:
            for attnheads in [4, 8]:
                for epochs in [3, 5, 7]:
                    for masking_percentage in [0.7, 0.8, 0.9]:
                        print(f"Evaluating model with: LR: {lr} | LAYERS: {layers} | ATTN_HEADS: {attnheads} | EPOCHS: {epochs} | MASKING: {masking_percentage}...")

                        config.training_settings['learning_rate'] = lr
                        config.model_conf['num_encoder_layers'] = layers
                        config.model_conf['n_heads'] = attnheads
                        config.training_settings['epochs'] = epochs
                        config.model_conf['modality_masking_prob'] = masking_percentage

                        val_output = pipeline(args, config)

                        # Check if this model's accuracy is better than the previous best
                        if val_output['icm-norm'] > best_ICM:
                            best_ICM = val_output['icm-norm']
                            best_lr = lr
                            best_layers = layers
                            best_attnheads = attnheads
                            best_epochs = epochs
                            best_masking = masking_percentage

                        # Save the best hyperparameters to a text file
                        with open("./best_simple_soft_hyperparameters.txt", "a") as file:
                            file.write(f"LR: {lr}\n")
                            file.write(f"LAYERS: {layers}\n")
                            file.write(f"ATTN_HEADS: {attnheads}\n")
                            file.write(f"EPOCHS: {epochs}\n")
                            file.write(f"MASKING: {masking_percentage}\n")
                            file.write("---------------------------------")
                            file.write(f"ICM-norm: {val_output['icm-norm'] }\n\n")

    # Save the best hyperparameters to a text file
    with open("./best_simple_soft_hyperparameters.txt", "a") as file:
        file.write(f"Best Hyperparameters:\n")
        file.write(f"Best LR: {best_lr}\n")
        file.write(f"Best LAYERS: {best_layers}\n")
        file.write(f"Best ATTN_HEADS: {best_attnheads}\n")
        file.write(f"Best EPOCHS: {best_epochs}\n")
        file.write(f"Best MASKING: {best_masking}\n")
        file.write("---------------------------------")
        file.write(f"ICM-norm: {best_ICM}\n")


if __name__ == "__main__":

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Training and/or evaluation of models for EXIST2024.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', required=True, type=str, help='Configuration file to build, train, and evaluate the model')
    parser.add_argument('--training-dataset', required=True, type=str, help='CSV file representing the training dataset')
    parser.add_argument('--validation-dataset', required=True, type=str, help='CSV file representing the validation dataset')
    parser.add_argument('--test-dataset', required=True, type=str, help='CSV file representing the test dataset')
    parser.add_argument('--mode', default='both', type=str, help='Choose between: "training", "evaluation", or "both"')
    parser.add_argument('--load-checkpoint', default='', type=str, help='Choose between: "training", "evaluation", or "both"')
    parser.add_argument("--yaml-overrides", metavar="CONF:[KEY]:VALUE", nargs='*', help="Set a number of conf-key-value pairs for modifying the yaml config file on the fly.")
    parser.add_argument("--use-modalities", nargs='+', default=['all_modalities'], help="It allows you to choose which modalities will be used.")
    parser.add_argument('--save', default=1, type=int, help='Do you want to save checkpoints and output reports?')
    parser.add_argument('--output-dir', required=True, type=str, help='Path where to save model checkpoints and predictions')

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
    complete()
