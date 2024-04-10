import json
import torch

class PinkDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path, task='task4', is_training=True):

      with open(dataset_path, 'r') as f:
          self.dataset = json.load(f)

      if task == 'task4':
          # -- task 4: sexism detection in memes
          self.label2id = {
              'NO': 0,
              'YES': 1,
          }

      elif task == 'task5':
          # -- task 5: source intention detection in memes
          self.label2id = {
              'DIRECT': 0,
              'JUDGEMENTAL': 1,
          }

      elif task == 'task6':
          # -- task 6: sexism categorization in memes
          self.label2id = {
              'IDEOLOGICAL AND INEQUALITY': 0,
              'STEREOTYPING AND DOMINANCE': 1,
              'OBJECTIFICATION': 2,
              'SEXUAL VIOLENCE': 3,
              'MISOGYNY AND NON-SEXUAL VIOLENCE': 4,
          }

      else:
          raise ValueError(f'unknown task {task}. It should be "task4", "task5", or "task6"')

  def __len__(self):
    return len(self.dataset.keys())

  def __getitem__(self, index):
    sample = self.dataset[self.dataset.keys()[index]]

    batch_sample = {}
    batch_sample['image'] = np.load( sample['img_emb_path'] )['emb']
    batch_sample['text'] = np.load( sample['text_emb_path'] )['emb']
    batch_sample['caption'] = np.load( sample['caption_emb_path'] )['emb']
    batch_sample['language'] = sample['lang'].strip().lower()

    # TODO: 6 annotators => 6 labels & sometimes one annotator provides more than one label
    # batch_sample['label'] = sample['label']

    return batch_sample
