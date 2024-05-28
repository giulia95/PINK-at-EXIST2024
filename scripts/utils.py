import os
import sys
import json
import torch
import pickle
import numpy as np

from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from datasets import PinkDataset
from models import PinkSimple, PinkTransformer

from pyevall.utils.utils import PyEvALLUtils
from pyevall.evaluation import PyEvALLEvaluation

def build_model(config):
    if config.model == 'pink_transformer':
        model = PinkTransformer(config).to(config.device)
    elif config.model == 'pink_simple':
        model = PinkSimple(config).to(config.device)
    else:
        raise ValueError(f'unknown {config.model} model architecture')

    return model

def get_dataloader(config, dataset_path, is_training=False):

    dataset = PinkDataset(dataset_path, task_id=config.task, language_filter=config.language_filter, is_training=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=is_training,
        batch_size=config.training_settings['batch_size'],
        num_workers=config.training_settings['num_workers'],
        pin_memory=True,
    )

    return dataloader

def get_optimizer_and_scheduler(config, model, train_loader):
    # -- optimizer
    if config.training_settings['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            config.training_settings['learning_rate'],
        )
    else:
        raise ValueError(f'unknown {config.training_settings["optimizer"]} optimizer')

    # -- scheduler
    if config.training_settings['scheduler'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.training_settings['learning_rate'],
            epochs=config.training_settings['epochs'],
            steps_per_epoch=len(train_loader),
            anneal_strategy=config.training_settings['anneal_strategy'],
        )
    else:
        raise ValueError(f'unknown {config.training_settings["scheduler"]} scheduler')

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

def override_yaml(yaml_config, to_override):
    if to_override is not None:
        for new_setting in to_override:
            if new_setting.count(':') == 1:
                key, value = new_setting.split(':')
                value_type_func = type(yaml_config[key])
                if value_type_func == bool:
                    yaml_config[key] = value == 'true'
                else:
                    yaml_config[key] = value_type_func(value)

            elif new_setting.count(':') == 2:
                conf, key, value = new_setting.split(':')
                value_type_func = type(yaml_config[conf][key])
                if value_type_func == bool:
                    yaml_config[conf][key] = value == 'true'
                else:
                    yaml_config[conf][key] = value_type_func(value)

    return yaml_config

def soft_label_postprocessing(prob, is_yes=True):
    no_choices = [0.0, 0.1667, 0.3334, 0.5, 0.6667, 0.8334, 1.0]
    yes_choices = [0.0, 0.1666, 0.3333, 0.5, 0.6666, 0.8333, 1.0]

    return min(yes_choices if is_yes else no_choices, key=lambda x: abs(x-prob))

def soft_output_to_json(output_path, out_key, model_output):
    tojson = []
    for idx, sample in enumerate(model_output[out_key]):
        tojson.append({
            'test_case': 'EXIST2024',
            'id': str(model_output['sample_id'][idx]),
            'value': {
                'YES': soft_label_postprocessing(sample[-1]),
                'NO': soft_label_postprocessing(sample[0], is_yes=False),
            }
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tojson, f, ensure_ascii=False, indent=2)

def hard_label_postprocessing(probs):
    if type(probs) is list:
        pred = np.argmax(probs)
    else:
        pred = probs

    return 'YES' if pred else 'NO'

def hard_output_to_json(output_path, out_key, model_output):
    tojson = []
    for idx, sample in enumerate(model_output[out_key]):
        tojson.append({
            'test_case': 'EXIST2024',
            'id': str(model_output['sample_id'][idx]),
            'value': hard_label_postprocessing(sample),
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tojson, f, ensure_ascii=False, indent=2)

def evaluate_model(output_dir, task_id, model_output, preds_output_name='validation_preds_last_epoch.json', golds_output_name='validation_golds.json', only_output_json=False):
    '''Implementation for Task 4 in a soft-soft evaluation.
    '''
    # -- creating output directory
    os.makedirs(output_dir, exist_ok=True)

    output_to_json_func = soft_output_to_json if 'soft' in task_id else hard_output_to_json

    # -- creating temporary files
    prediction_path = os.path.join(output_dir, preds_output_name)
    output_to_json_func(prediction_path, 'probs', model_output)

    gold_path = os.path.join(output_dir, golds_output_name)
    output_to_json_func(gold_path, 'labels', model_output)

    # -- setting up PyEvALL
    if not only_output_json:
        pyevall = PyEvALLEvaluation()
        params = dict()
        params[PyEvALLUtils.PARAM_REPORT] = PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED

        if 'soft' in task_id:
            metrics=['ICMSoft', 'ICMSoftNorm', 'CrossEntropy']
        else:
            metrics=['ICM', 'ICMNorm' ,'FMeasure']

        # -- evaluating model results
        report = pyevall.evaluate(prediction_path, gold_path, metrics, **params)

        return report

    return None
