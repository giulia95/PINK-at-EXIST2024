import os
import yaml
import torch
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from colorama import Fore
from utils import *
from sklearn.metrics import accuracy_score, classification_report

def train(args, config):
    model.train()
    optimizer.zero_grad()
    train_stats = {'loss': 0.0} # , 'icm-soft-norm': 0.0}

    for batch in tqdm(train_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        batch = {k: v.to(device=config.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}

        # -- forward pass
        model_output = model(batch)

        # report = evaluate_model(args.output_dir, model_output)

        # -- optimization
        model_output['loss'].backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_stats['loss'] += model_output['loss'].item()
        # train_stats['icm-soft-norm'] += report.report['metrics']['ICMSoftNorm']['results']['average_per_test_case']

        # in case it is hard labels
        # accuracy_score(model_output['preds'].detach().cpu().numpy(), model_output['labels'].detach().cpu().numpy())

    train_stats['loss'] = train_stats['loss'] / len(train_loader)
    # train_stats['icm-soft-norm'] = (train_stats['icm-soft-norm'] / len(train_loader)) * 100.0

    return train_stats

def evaluate(args, config):
    model.eval()
    eval_output = {'sample_id': [], 'logits': [], 'probs': [], 'preds': [], 'labels': [], 'loss': 0.0}

    with torch.no_grad():
        for batch in tqdm(val_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
            batch = {k: v.to(device=config.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}

            # -- forward pass
            model_output = model(batch)

            # -- gathering statistics
            eval_output['loss'] += model_output['loss'].item()
            for eval_key in eval_output.keys():
                eval_output[eval_key] += model_output[eval_key].detach().cpu().numpy().tolist()

    report = evaluate_model(args.output_dir, config.task, eval_output)
    shutil.rmtree(args.output_dir)

    eval_output['loss'] = eval_output['loss'] / len(val_loader)
    eval_output['pyevall-report'] = report

    icm_key = 'ICMSoftNorm' if 'soft' in config.task else 'ICMNorm'
    eval_output['icm-norm'] = report.report['metrics'][icm_key]['results']['average_per_test_case'] # accuracy_score(model_output['preds'].detach().cpu().numpy(), model_output['labels'].detach().cpu().numpy())

    return eval_output

def pipeline(args, config):

    # -- building model architecture
    global model, optimizer, scheduler, train_loader, val_loader
    model = build_model(config)
    print(model)
    print(f'Model Parameters: {sum([param.nelement() for param in model.parameters()])}')

    # -- loading model checkpoint
    if args.load_checkpoint:
        model.load_state_dict(checkpoint)

    # -- training process
    if args.mode in ['training', 'both']:

        # -- creating data loaders
        train_loader = get_dataloader(config, args.training_dataset, is_training=True)
        val_loader = get_dataloader(config, args.validation_dataset, is_training=False)

        # -- defining the optimizer and its scheduler
        optimizer, scheduler = get_optimizer_and_scheduler(config, model, train_loader)

        for epoch in range(1, config.training_settings['epochs']+1):
            train_stats = train(args, config)
            val_output = evaluate(args, config)

            # -- saving model checkpoint
            if args.save:
                save_checkpoint(model, args.output_dir, f'epoch_{str(epoch).zfill(3)}')
            print(f"Epoch {epoch}: TRAIN LOSS={round(train_stats['loss'],4)} || VAL LOSS={round(val_output['loss'],4)} | VAL ICM-NORM={round(val_output['icm-norm'],2)}%")

    if args.mode in ['evaluation', 'both']:
        val_loader = get_dataloader(config, args.validation_dataset, is_training=False)
        val_output = evaluate(args, config)

        # -- saving model output
        if args.save:
            save_model_output(val_output, args.output_dir, args.output_name)

        # -- displaying final report
        # eval_report = classification_report(
        #     val_output['preds'],
        #     val_output['labels'],
        # )
        #     target_names=config.class_names,
        print()
        # print(eval_report)
        print(val_output['pyevall-report'].print_report())

    return val_output