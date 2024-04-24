import os
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def build_clip():

  image_encoder = SentenceTransformer('clip-ViT-B-32')
  text_encoder = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

  return image_encoder, text_encoder

def extract_embeddings(image, text, caption):

  img_emb = clip_image_encoder.encode([image])
  text_emb = clip_text_encoder.encode([text])
  caption_emb = clip_text_encoder.encode([caption])

  print(img_emb.shape, text_emb.shape, caption_emb.shape)

  return img_emb, text_emb, caption_emb

def save_embedding(embedding, output_path):
  """
  Save the embedding provided in input in a .npz file.
  ________________Input________________
  output_path: path to the file that will store the embedding
  embedding: embedding file to save
  """
  # embedding = embedding.detach().cpu().numpy()
  np.savez_compressed(output_path, emb=embedding)

if __name__ == "__main__":

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Extraction of CLIP-based embeddings.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--split-path', required=True, type=str, help='Path to the JSON representing the dataset split from which the embeddings will be extracted')
    parser.add_argument('--caption-csv-path', default='', type=str, help='Path to the CSV containing the extracted captions for the dataset')
    parser.add_argument('--output-dir', required=True, type=str, help='Path where to save the extracted embeddings')

    args = parser.parse_args()

    split_dir = os.sep.join(args.split_path.split(os.sep)[:-1])

    # -- building CLIP model
    clip_image_encoder, clip_text_encoder = build_clip()

    # -- creating output embedding directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'text'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'caption'), exist_ok=True)

    # -- processing dataset
    with open(args.split_path) as f:
        split = json.load(f)

    if args.caption_csv_path:
        caption_df = pd.read_csv(args.caption_csv_path, sep='\t')

    for sample_id in tqdm(split.keys()):
        sample = split[sample_id]

        # -- extracting embeddings
        image = Image.open( os.path.join(split_dir, sample['path_memes']) )
        text = sample['text'].strip()
        caption = sample['blip_caption'].strip() if not args.caption_csv_path else caption_df.loc[caption_df['image_name']==int(sample_id)].iloc[0]['caption_free'].strip()

        print(f'{text}##{caption}')
        img_emb, text_emb, caption_emb = extract_embeddings(image, text, caption)

        # -- saving embeddings
        save_embedding(img_emb, os.path.join(args.output_dir, 'image', f'{sample_id}.npz'))
        save_embedding(text_emb, os.path.join(args.output_dir, 'text', f'{sample_id}.npz'))
        save_embedding(caption_emb, os.path.join(args.output_dir, 'caption', f'{sample_id}.npz'))

