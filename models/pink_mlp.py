import torch
import numpy as np
from einops.layers.torch import Reduce

class PinkMLP(torch.nn.Module):

  def __init__(self, args):
    super(PinkMLP, self).__init__()

    self.args = args

    # 0. Embedding Projection Layer
    self.emb_projection = torch.nn.Linear(
        args.model_conf['input_dim'],
        args.model_conf['latent_dim'],
        bias=False,
    )

    # 1. Modality Embedding Layer
    self.modality_map = {mod_name:mod_id for mod_id, mod_name in enumerate(sorted(args.modalities))}
    self.modality_emb = torch.nn.Embedding(
        len(args.modalities),
        args.model_conf['latent_dim'],
    )

    # 2. Language Embedding Layer
    self.language_map = {lang_name:lang_id for lang_id, lang_name in enumerate(sorted(args.languages))}
    self.language_emb = torch.nn.Embedding(
        len(args.languages),
        args.model_conf['latent_dim'],
    )

    # 3. Multi-Modal Multi-Lingual MLP Backbone
    dim_after_flatten = args.model_conf['latent_dim'] * len(args.modalities)
    self.mlp_backbone = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(dim_after_flatten, args.model_conf['mlp_dim']),
        torch.nn.SiLU(),
        torch.nn.Linear(args.model_conf['mlp_dim'], args.model_conf['latent_dim']),
    )


    # 4. Output Classification
    self.classifier = torch.torch.nn.Linear(
        args.model_conf['latent_dim'],
        args.num_classes,
        bias=False,
    )

    # 5. Computing Loss Function
    if args.training_settings['loss_criterion'] == 'cross_entropy':
        self.loss_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    else:
      raise ValueError(f'unknown loss criterion {args.training_settings["loss_criterion"]}')

  def forward(self, batch):
    model_output = {}
    all_modalities = []

    for modality_id in self.modality_map.keys():
      data = batch[modality_id] # -- (batch, 1, input_dim)

      # 0. Embedding Projection Layer
      data = self.emb_projection(data) # -- (batch, 1, latent_dim)

      # 1. Modality Embedding Layer
      modality_id = torch.LongTensor([self.modality_map[modality_id]]).to(data.device)
      data = data + self.modality_emb(modality_id)

      # 2. Language Embedding Layer
      language_ids = torch.LongTensor(list( map(self.language_map.get, batch['language']) )).to(data.device)
      data = data + self.language_emb(language_ids).unsqueeze(1)

      all_modalities.append( data )

    # Modality Masking
    if self.training and (self.args.model_conf['modality_masking_prob'] > 0.0) and (torch.rand(1).item() < self.args.model_conf['modality_masking_prob']):
        random_modality_idx = np.random.randint(len(self.modality_map.keys()), size=1)[0]
        all_modalities[random_modality_idx] = all_modalities[random_modality_idx] * 0.0

    cat_data = torch.cat(all_modalities, dim=1) # -- (batch, num_modalities, latent_dim)

    # 3. Multi-Modal Multi-Lingual MLP Backbone
    output = self.mlp_backbone(cat_data)

    # 4. Output Classification
    logits = self.classifier(output)

    model_output['logits'] = logits
    model_output['probs'] = torch.nn.functional.softmax(logits, dim = -1)
    model_output['preds'] = logits.argmax(dim = -1)
    model_output['labels'] = batch['label']
    model_output['loss'] = self.loss_criterion(logits, batch['label'])

    return model_output
