import os
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from colorama import Fore
from scripts.utils import *
from sklearn.metrics import classification_report

def train():
    model.train()
    optimizer.zero_grad()
    train_stats = {'loss': 0.0, 'f1-score': 0.0}

    for batch in tqdm(train_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.GREEN, Fore.RESET))):
        # -- forward pass
        model_output = model(batch)

        # -- optimization
        model_output['loss'].backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # -- metric report
        train_metrics = classification_report(
            model_output['preds'],
            model_output['labels'],
            target_names=config.class_names,
            output_dict=True,
        )

        train_stats['loss'] += model_output['loss'].item()
        train_stats['f1-score'] += train_metrics['misoginy']['f1-score']

  train_stats['loss'] = train_stats['loss'] / len(train_loader)
  train_stats['f1-score'] = (train_stats['f1-score'] / len(train_loader)) * 100.0

  return train_stats

def evaluate():
    model.eval()
    eval_stats = {'logits': [], 'probs': [], 'preds': [], 'labels': [], 'loss': 0.0, 'f1-score': 0.0}

    with torch.no_grad():
        for batch in tqdm(val_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.BLUE, Fore.RESET))):
            # -- forward pass
            model_output = model(batch)

            # -- gathering statistics
            eval_stats['loss'] += model_output['loss'].item()
            for eval_key in eval_output.keys():
                eval_stats[eval_key] += model_output[eval_key].detach.cpu().numpy().tolist()

    # -- metric report
    eval_metrics = classification_report(
        eval_output['preds'],
        eval_output['labels'],
        target_names=config.class_names,
        output_dict=True,
    )

    eval_stats['loss'] = eval_stats['loss'] / len(val_loader)
    eval_stats['f1-score'] = eval_metrics['misoginy']['f1-score']

  return eval_stats

if __name__ == "__main__":

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Training and/or evaluation of models for EXIST2024.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', required=True, type=str, help='Configuration file to build, train, and evaluate the model')
    parser.add_argument('--training-dataset', required=True, type=str, help='CSV file representing the training dataset')
    parser.add_argument('--validation-dataset', required=True, type=str, help='CSV file representing the validation dataset')
    parser.add_argument('--mode', default='both', type=str, help='Choose between: "training", "evaluation", or "both"')
    parser.add_argument('--load-checkpoint', default='', type=str, help='Choose between: "training", "evaluation", or "both"')
    parser.add_argument('--output-dir', required=True, type=str, help='Path where to save model checkpoints and predictions')
    parser.add_argument('--output-name', required=True, type=str, help='Choose between: "validation", or "test"')

    args = parser.parse_args()

    # -- loading configuration file
    config_file = Path(args.config)
    with config_file.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config = argparse.Namespace(**config)

    # -- building model architecture
    model = build_model(config)

    # -- loading model checkpoint
    if args.load_checkpoint:
        model.load_state_dict(checkpoint)

    # -- training process
    if args.mode in ['training', 'both']:

        # -- creating data loaders
        train_loader = get_dataloader(config, args.training_dataset, is_training=True)
        val_loader = get_dataloader(config, args.validation_dataset, is_training=False)

        # -- defining the optimizer and its scheduler
        optimizer, scheduler = get_optimizer_and_scheduler(config, train_loader)

        for epoch in range(1, config.training_settings['epochs']+1):
            train_stats = train()
            val_stats = evaluate()

            # -- saving model checkpoint
            save_checkpoint(model, args.output_dir, f'epoch_{int(epoch).zfill(3)}')
            print(f"Epoch {epoch}: TRAIN LOSS={round(train_stats['loss'],4)} | TRAIN F1={round(train_stats['f1-score'],2)}% || VAL LOSS={round(val_stats['loss'],4)} | VAL F1={round(val_stats['f1-score'],2)}%")

    if args.mode in ['evaluation', 'both']:
        val_loader = get_dataloader(config, args.validation_dataset, is_training=False)
        val_stats = evaluate()

        # -- saving model output
        save_model_output(val_stats, args.output_dir, args.output_name)

        # -- displaying final report
        eval_report = classification_report(
            val_stats['preds'],
            val_stats['labels'],
            target_names=config.class_names,
        )
        print()
        print(eval_report)
