import torch
import numpy as np
import torch.nn.functional as F
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

    # [Optional] Use CLS token
    if args.model_conf['use_cls_token']:
        self.cls_emb = torch.nn.Embedding(
            1,
            args.model_conf['latent_dim'],
        )
    else:
        self.cls_emb = None

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

    # 3. Annotators' Metadata Layer
    if args.model_conf['annotators_emb_type'] == 'embedding':
        self.annotators_emb = torch.nn.ModuleDict({
            metadata['name']: torch.nn.Embedding(metadata['num_items'], args.model_conf['latent_dim'])
            for metadata in self.args.annotators_metadata
        })
    elif args.model_conf['annotators_emb_type'] == 'linear':
        self.annotators_emb = torch.nn.Linear(24, args.model_conf['latent_dim'])
    else:
        self.annotators_emb = None

    # 4. Multi-Modal Multi-Lingual Transformer Backbone
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

    # 5. Output Classification
    self.classifier = torch.nn.Sequential(
        torch.torch.nn.Linear(
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

    # 3. Annotators' Metadata
    if self.args.model_conf['annotators_emb_type'] == 'embedding':
        all_annotators = []
        for metadata in self.args.annotators_metadata:
            metadata_id = metadata['name']
            metadata_emb = self.annotators_emb[metadata_id](batch[metadata_id])
            all_annotators.append(metadata_emb)

        annotators_data = torch.cat(all_annotators, dim=1)
        cat_data = torch.cat([cat_data, annotators_data], dim=1)
    elif self.args.model_conf['annotators_emb_type'] == 'linear':
        all_annotators = []
        for metadata in self.args.annotators_metadata:
            metadata_id = metadata['name']
            all_annotators.append( batch[metadata_id].unsqueeze(1) )
        annotators_data = torch.cat(all_annotators, dim=-1)
        annotators_data = self.annotators_emb(annotators_data.type(torch.float32))
        cat_data = torch.cat([cat_data, annotators_data], dim=1)

    # [Optional] CLS token
    if self.cls_emb is not None:
        cls_token = self.cls_emb( torch.LongTensor([0]*cat_data.shape[0]).to(cat_data.device) ).unsqueeze(1)
        cat_data = torch.cat([cat_data, cls_token], dim=1)

    # 4. Multi-Modal Multi-Lingual Transformer Backbone
    output = self.transformer_backbone(cat_data)

    # 5. Output Classification
    if self.cls_emb is not None:
        output = output[:, -1, :]
    else:
        output = Reduce('b n d -> b d', 'mean')(output)

    logits = self.classifier(output)

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



