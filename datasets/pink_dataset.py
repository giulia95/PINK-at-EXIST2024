import torch
import numpy as np
import pandas as pd

SOFT_LABEL_TASK4_MAPPING = {
    0.0: 0,
    0.1666: 1,
    0.3333: 2,
    0.5: 3,
    0.6666: 4,
    0.8333: 5,
    1.0: 6
}

class PinkDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, task_id='soft_label_task4', is_training=True):

        self.task_id = task_id
        self.dataset = pd.read_csv(dataset_path, sep='\t')

    def __len__(self):
      return len(self.dataset)

    def __getitem__(self, index):
      sample = self.dataset.iloc[index]

      batch_sample = {}
      batch_sample['clip_image'] = np.load( sample['clip_image_emb_path'] )['emb']
      batch_sample['clip_text'] = np.load( sample['clip_text_emb_path'] )['emb']
      batch_sample['clip_caption'] = np.load( sample['clip_caption_emb_path'] )['emb']
      batch_sample['language'] = sample['lang'].strip().lower()
      batch_sample['label'] = self.__getlabel__(self.task_id, sample)

      return batch_sample

    def __getlabel__(self, task_id, sample):
        if 'soft' in task_id:
            if task_id == 'soft_label_task4':
                return SOFT_LABEL_TASK4_MAPPING[sample[task_id]]
        else:
            return sample[task_id]
