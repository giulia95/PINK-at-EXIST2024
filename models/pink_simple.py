import torch
import numpy as np
import torch.nn.functional as F
from einops.layers.torch import Reduce

class PinkSimple(torch.nn.Module):

  def __init__(self, args):
    super(PinkSimple, self).__init__()

    self.args = args

    # Annotators' Metadata Layer
    if args.model_conf['annotators_emb_type'] == 'embedding':
        self.annotators_emb = torch.nn.ModuleDict({
            metadata['name']: torch.nn.Embedding(metadata['num_items'], args.model_conf['latent_dim'])
            for metadata in self.args.annotators_metadata
        })
    elif args.model_conf['annotators_emb_type'] == 'linear':
        self.annotators_emb = torch.nn.Linear(24, args.model_conf['latent_dim'])
    else:
        self.annotators_emb = None

    # Output Classification
    input_dim = 0
    for modality in self.args.modalities:
        input_dim += modality['input_dim']

    self.classifier = torch.nn.Sequential(
        torch.nn.Linear(
            input_dim,
            args.model_conf['latent_dim'],
            bias=False,
        ),
        torch.nn.LeakyReLU(),#GELU(),
        torch.nn.Linear(
            args.model_conf['latent_dim'],
            args.num_classes,
            bias=False,
        ),
    )

    # 6. Computing Loss Function
    if args.training_settings['loss_criterion'] == 'cross_entropy':
        self.loss_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    elif args.training_settings['loss_criterion'] == 'nll':
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.loss_criterion = torch.nn.NLLLoss(reduction='mean')
    elif args.training_settings['loss_criterion'] == 'mse':
        self.loss_criterion = torch.nn.MSELoss()
    elif args.training_settings['loss_criterion'] == 'kl':
        self.loss_criterion = torch.nn.KLDivLoss(reduction='batchmean')
    else:
      raise ValueError(f'unknown loss criterion {args.training_settings["loss_criterion"]}')

  def forward(self, batch, test_for_submission=False):
    model_output = {}
    all_modalities = []

    for modality in self.args.modalities:
      data = batch[modality['name']] # -- (batch, 1, input_dim)
      all_modalities.append( data )

    # Modality Masking
    if self.training and (self.args.model_conf['modality_masking_prob'] > 0.0) and (torch.rand(1).item() < self.args.model_conf['modality_masking_prob']):
        random_modality_idx = np.random.randint(len(self.args.modalities), size=1)[0]
        all_modalities[random_modality_idx] = all_modalities[random_modality_idx] * 0.0

    cat_data = torch.cat(all_modalities, dim=-1).squeeze(1) # -- (batch, input_flatten_dim)

    logits = self.classifier(cat_data)

    model_output['sample_id'] = batch['sample_id']
    model_output['logits'] = logits
    model_output['probs'] = torch.nn.functional.softmax(logits, dim = -1)
    model_output['preds'] = logits.argmax(dim = -1)
    model_output['labels'] = batch['label']
    if not test_for_submission:
        if self.args.training_settings['loss_criterion'] == 'nll':
            log_logits = self.log_softmax(logits)
            model_output['loss'] = self.loss_criterion(log_logits, batch['label'])
        elif self.args.training_settings['loss_criterion'] == 'kl':
            log_logits = F.log_softmax(logits, dim=1)
            log_targets = F.softmax(batch['label'], dim=1)
            model_output['loss'] = self.loss_criterion(log_logits, log_targets)
        else:
            model_output['loss'] = self.loss_criterion(logits, batch['label'])

    return model_output
