import torch
from transformers import BertTokenizer, BertModel
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd

def get_bert_embeddings(sentence, max_length=512):
    # Tokenize input text
    tokens = tokenizer.tokenize(sentence)
    # Truncate or pad the sequence to match the maximum sequence length
    tokens = tokens[:max_length - 2]  # Account for [CLS] and [SEP] tokens
    # Add [CLS] and [SEP] tokens
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    # Convert tokens to vocabulary indices
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_length = len(input_ids)
    # Pad sequences if necessary
    padding_length = max_length - input_length
    input_ids += [tokenizer.pad_token_id] * padding_length

    attn_mask = torch.zeros(max_length)
    attn_mask[:(input_length+1)] = 1.0

    # Convert input_ids to PyTorch tensor
    input_tensor = torch.tensor([input_ids]).to(device)
    attn_mask_tensor = attn_mask.unsqueeze(0).to(device)

    # Forward pass, get hidden states
    with torch.no_grad():
        outputs = model(input_tensor, attn_mask_tensor)
        # Extract hidden states of all layers (last_hidden_state) for the first token (CLS token)
        hidden_states = outputs[0]
        # cls_embedding = hidden_states[:, 0, :].cpu().numpy()
        cls_embedding = hidden_states[:, :input_length, :].cpu().numpy()
    # Return the embedding for the [CLS] token
    return cls_embedding

def save_embedding(embedding, output_path):
  """
  Save the embedding provided in input in a .npz file.
  ________________Input________________
  output_path: path to the file that will store the embedding
  embedding: embedding file to save
  """
  # embedding = embedding.detach().cpu().numpy()
  np.savez_compressed(output_path, emb=embedding)


##Set random values
seed_val = 42
#random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed_val)

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

if __name__ == "__main__":

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Extraction of BERT-based embeddings.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--split-path', required=True, type=str, help='Path to the JSON representing the dataset split from which the embeddings will be extracted')
    parser.add_argument('--output-dir', required=True, type=str, help='Path where to save the extracted embeddings')
    parser.add_argument('--type', required=True, type=str, help='Types of embeddings to extract (text, capt or text+capt)')
    parser.add_argument('--dataset', default='EXIST2024', type=str, help='Dataset name')

    args = parser.parse_args()

    # -- building BERT model
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)

    # -- creating output embedding directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'bert_token_text'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'bert_token_caption'), exist_ok=True)

    # -- processing dataset
    with open(args.split_path) as f:
        split = json.load(f)

    for sample_id in tqdm(split.keys()):
        sample = split[sample_id]

        # -- extracting embeddings
        if args.type != 'capt':
            text = sample['text'].strip()
            text_emb = get_bert_embeddings(text)
            save_embedding(text_emb, os.path.join(args.output_dir, 'bert_token_text', f'{sample_id}.npz'))

        if args.type != 'text':
            if 'test' in str(args.split_path):
                capt_df = pd.read_csv('/home/dgimeno/EXIST2024/PINK-at-EXIST2024/data/EXIST2024/EXIST_2024_Memes_Dataset/blip_captions_test.csv', sep='\t')
            else:
                capt_df = pd.read_csv('/home/dgimeno/EXIST2024/PINK-at-EXIST2024/data/EXIST2024/EXIST_2024_Memes_Dataset/blip_caption_train.csv', sep='\t')
            capt_df['image_name'] = capt_df['image_name'].astype(str)
            caption = capt_df.loc[capt_df['image_name']==str(sample_id), 'caption_free'].values[0]
            caption_emb = get_bert_embeddings(caption)
            save_embedding(caption_emb, os.path.join(args.output_dir, 'bert_token_caption', f'{sample_id}.npz'))
