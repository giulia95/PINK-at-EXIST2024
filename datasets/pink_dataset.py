import json
import torch
import numpy as np

class PinkDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path, task='hard_label_task4', is_training=True):

      self.task = task
      with open(dataset_path, 'r') as f:
          self.dataset = json.load(f)

  def __len__(self):
    return len(self.dataset.keys())

  def __getitem__(self, index):
    sample = self.dataset[list(self.dataset.keys())[index]]

    batch_sample = {}
    batch_sample['image'] = np.load( sample['img_emb_path'] )['emb']
    batch_sample['text'] = np.load( sample['text_emb_path'] )['emb']
    batch_sample['caption'] = np.load( sample['caption_emb_path'] )['emb']
    batch_sample['language'] = sample['lang'].strip().lower()
    batch_sample['label'] = sample[self.task]

    return batch_sample
