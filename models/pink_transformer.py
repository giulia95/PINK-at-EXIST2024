import torch
import numpy as np
from einops.layers.torch import Reduce

class PinkTransformer(torch.nn.Module):

  def __init__(self, args):
    super(PinkTransformer, self).__init__()

    self.args = args

    # 0. Embedding Projection Layer
    self.emb_projection = torch.nn.ModuleDict({
        modality['name']: LinearModalityEncoder(modality['input_dim'], args.model_conf['latent_dim'])
        for modality in self.args.modalities
    })

    # 1. Modality Embedding Layer
    self.modality_map = {modality['name']:id for id, modality in enumerate(sorted(self.args.modalities, key = lambda x: x['name']))}
    if args.model_conf['use_modality_emb']:
        self.modality_emb = torch.nn.Embedding(
            len(args.modalities),
            args.model_conf['latent_dim'],
        )
    else:
        self.modality_emb = None

    # 2. Language Embedding Layer
    self.language_map = {lang_name:lang_id for lang_id, lang_name in enumerate(sorted(args.languages))}
    if args.model_conf['use_language_emb']:
        self.language_emb = torch.nn.Embedding(
            len(args.languages),
            args.model_conf['latent_dim'],
        )
    else:
        self.language_emb = None

    # 3. Multi-Modal Multi-Lingual Transformer Backbone
    self.transformer_backbone = torch.nn.TransformerEncoder(
        encoder_layer=torch.nn.TransformerEncoderLayer(
            d_model=args.model_conf['latent_dim'],
            nhead=args.model_conf['n_head'],
            dim_feedforward=args.model_conf['dim_feedforward'],
            dropout=args.model_conf['dropout'],
            activation=torch.nn.SiLU(),
            batch_first = True,
        ),
        num_layers=args.model_conf['num_encoder_layers'],
        enable_nested_tensor=False,
    )

    # 4. Output Classification
    self.classifier = torch.nn.Sequential(
        torch.torch.nn.Linear(
            args.model_conf['latent_dim'],
            args.num_classes,
            bias=False,
        ),
    )

    # 5. Computing Loss Function
    if args.training_settings['loss_criterion'] == 'cross_entropy':
        self.loss_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    else:
      raise ValueError(f'unknown loss criterion {args.training_settings["loss_criterion"]}')

  def forward(self, batch, test_for_submission=False):
    model_output = {}
    all_modalities = []

    for modality_id in self.modality_map.keys():
      data = batch[modality_id] # -- (batch, 1, input_dim)

      # 0. Embedding Projection Layer
      data = self.emb_projection[modality_id](data) # -- (batch, 1, latent_dim)

      # 1. Modality Embedding Layer
      if self.modality_emb is not None:
          modality_id = torch.LongTensor([self.modality_map[modality_id]]).to(data.device)
          data = data + self.modality_emb(modality_id)

      # 2. Language Embedding Layer
      if self.language_emb is not None:
          language_ids = torch.LongTensor(list( map(self.language_map.get, batch['language']) )).to(data.device)
          data = data + self.language_emb(language_ids).unsqueeze(1)

      all_modalities.append( data )

    # Modality Masking
    if self.training and (self.args.model_conf['modality_masking_prob'] > 0.0) and (torch.rand(1).item() < self.args.model_conf['modality_masking_prob']):
        random_modality_idx = np.random.randint(len(self.modality_map.keys()), size=1)[0]
        all_modalities[random_modality_idx] = all_modalities[random_modality_idx] * 0.0

    cat_data = torch.cat(all_modalities, dim=1) # -- (batch, num_modalities, latent_dim)

    # 3. Multi-Modal Multi-Lingual Transformer Backbone
    output = self.transformer_backbone(cat_data)

    # 4. Output Classification
    output = Reduce('b n d -> b d', 'mean')(output)
    logits = self.classifier(output)

    model_output['sample_id'] = batch['sample_id']
    model_output['logits'] = logits
    model_output['probs'] = torch.nn.functional.softmax(logits, dim = -1)
    model_output['preds'] = logits.argmax(dim = -1)
    model_output['labels'] = batch['label']
    if not test_for_submission:
        model_output['loss'] = self.loss_criterion(logits, batch['label'])

    return model_output

class LinearModalityEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModalityEncoder, self).__init__()

        # 0. Batch Normalization
        self.batch_norm = torch.nn.BatchNorm1d(
            input_dim,
        )

        # 1. Linear Projection
        self.projection = torch.nn.Linear(
            input_dim,
            output_dim,
            bias=False,
        )

    def forward(self, x):
        x = self.batch_norm(x.permute(0,2,1)).permute(0,2,1)
        x = self.projection(x)

        return x


