import torch
import pickle

from models import PinkBaseline
from datasets import PinkDataset

def build_model(config):
    if config.model == 'pink_baseline':
        model = PinkBaseline(config).to(config.device)
    else:
        raise ValueError(f'unknown {config.model} model architecture')

    return model

def get_dataloader(config, dataset_path, is_training=False):

    dataset = PinkDataset(dataset_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=is_training,
        batch_size=config.training_settings['batch_size'],
        num_workers=config.training_settings['num_workers'],
        pin_memory=True,
    )

    return dataloader

def get_optimizer_and_scheduler(config, train_loader):
    # -- optimizer
    if config.training_settings['optimizer'] == "adamw":
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, our_amazing_model.parameters()),
            config.training_settings['learning_rate'],
        )
    else:
        raise ValueError('unknown {args.training_settings["optimizer"]} optimizer')

    # -- scheduler
    if config.training_settings['scheduler'] == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.training_settings['learning_rate'],
            epochs=config.training_settings['epochs'],
            steps_per_epoch=len(train_loader),
            anneal_strategy=args.training_settings['anneal_strategy'],
        )
    else:
        raise ValueError('unknown {args.training_settings["scheduler"]} scheduler')

    return optimizer, scheduler

def save_checkpoint(model, output_dir, suffix):
    # -- creating output directories
    model_checkpoints_dir = os.path.join(output_dir, 'model_checkpoints')
    os.makedirs(model_checkpoints_dir, exist_ok=True)

    # -- saving model checkpoint
    model_checkpoint_path = os.path.join(model_checkpoints_dir, f'{suffix}.pth')
    print(f'Saving model checkpoint in {model_checkpoint_path}...')
    torch.save(model.state_dict(), model_checkpoint_path)

def save_model_output(output_stats, output_dir, suffix):
    # -- creating output directories
    model_outputs_dir = os.path.join(output_dir, 'model_output')
    os.makedirs(model_outputs_dir, exist_ok=True)

    # -- saving model output
    model_output_path = os.path.join(model_outputs_dir, f'{suffix}.pkl')
    print(f'Saving model output in {model_output_path}...')
    with open(model_output_path, 'wb') as handle:
        pickle.dump(output_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
