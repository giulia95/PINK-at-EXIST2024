import torch
import numpy as np
import pandas as pd

class PinkDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, task_id='soft_label_task4', language_filter='both', is_training=True):

        self.task_id = task_id
        self.dataset = pd.read_csv(dataset_path, sep='\t')
        if language_filter in ['en', 'es']:
            self.dataset = self.dataset[self.dataset['lang'] == language_filter]

    def __len__(self):
      return len(self.dataset)

    def __getitem__(self, index):
      sample = self.dataset.iloc[index]

      batch_sample = {}
      batch_sample['sample_id'] = sample['sample_id']
      batch_sample['clip_image'] = np.load( sample['clip_image_emb_path'] )['emb']
      batch_sample['clip_text'] = np.load( sample['clip_text_emb_path'] )['emb']
      batch_sample['clip_caption'] = np.load( sample['clip_caption_emb_path'] )['emb']
      batch_sample['bert_text'] = np.load( sample['bert_text_emb_path'] )['emb']
      batch_sample['bert_caption'] = np.load( sample['bert_caption_emb_path'] )['emb']
      batch_sample['language'] = sample['lang'].strip().lower()
      batch_sample['label'] = self.__getlabel__(self.task_id, sample)

      return batch_sample

    def __getlabel__(self, task_id, sample):
        if 'soft' in task_id:
            if task_id == 'soft_label_task4':
                yes_prob = sample[task_id]
                no_prob = round(1 - yes_prob, len(str(yes_prob).split('.')[-1]))
                return np.array([no_prob, yes_prob])
        else:
            return sample[task_id]
