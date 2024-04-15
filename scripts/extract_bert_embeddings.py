import torch
from transformers import BertTokenizer, BertModel
import os
import json
import argparse
import numpy as np
from tqdm import tqdm

# Define a function to get BERT sentence embeddings
def get_bert_embeddings(sentence):
    # Tokenize input text
    tokens = tokenizer.tokenize(sentence)
    # Add [CLS] and [SEP] tokens
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    # Convert tokens to vocabulary indices
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Convert input_ids to PyTorch tensor
    input_tensor = torch.tensor([input_ids]).to(device)
    # Forward pass, get hidden states
    with torch.no_grad():
        outputs = model(input_tensor)
        # Extract hidden states of all layers (last_hidden_state) for the first token (CLS token)
        hidden_states = outputs[0]
        cls_embedding = hidden_states[:, 0, :].cpu().numpy()
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

    args = parser.parse_args()

    # -- building BERT model
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)

    # -- creating output embedding directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'BERT-text'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'BERT-caption'), exist_ok=True)

    # -- processing dataset
    with open(args.split_path) as f:
        split = json.load(f)

    for sample_id in tqdm(split.keys()):
        sample = split[sample_id]

        # -- extracting embeddings
        text = sample['text'].strip()
        #caption = sample['blip_caption'].strip()

        text_emb = get_bert_embeddings(text)
        #caption_emb = get_bert_embeddings(caption)

        # -- saving embeddings
        save_embedding(text_emb, os.path.join(args.output_dir, 'BERT-text', f'{sample_id}.npz'))
        #save_embedding(caption_emb, os.path.join(args.output_dir, 'BERT-caption', f'{sample_id}.npz'))



