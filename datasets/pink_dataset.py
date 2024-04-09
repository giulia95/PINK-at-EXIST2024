import torch
import pandas as pd

class PinkDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path, is_training=True):

      self.dataset = pd.read_csv(dataset_path, delimiter=",")
      self.dataset['label'] = self.dataset['label'].apply(lambda x: 1 if x == 'misonigy' else 0)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    sample = self.dataset.iloc[index]

    batch_sample = {}
    batch_sample['caption'] = np.load( sample['caption_embed_path'] )['emb']
    batch_sample['image'] = np.load( sample['img_embed_path'] )['emb']
    batch_sample['text'] = np.load( sample['text_embed_path'] )['emb']
    batch_sample['language'] = sample['language'].strip().lower()
    batch_sample['label'] = sample['label']

    return batch_sample
