import os
import argparse
import numpy as np
import pandas as pd
from PIL import image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

def build_clip(checkpoint):
  model = CLIPModel.from_pretrained(checkpoint)
  processor = CLIPProcessor.from_pretrained(checkpoint)

  return model, processor

def extract_embeddings(image, text, caption):
  """
  This function processes given image and text inputs through the CLIP model, generating embeddings for both.
  It utilizes a CLIPProcessor instance to prepare inputs for the model, considering both text and image inputs.
  If a caption is provided, it incorporates it along with the text data for processing.
  The processed inputs are then fed into the CLIP model to obtain embeddings, which are returned as outputs.
  ________________Input________________
  image: Image data to be processed and embedded.
  text: Text data to be processed and embedded.
  caption (optional): Additional text data, typically a caption related to the image (default is an empty string).
  ________________Output________________
   image_embeds: Embeddings generated from processing the input image.
   text_embeds: Embeddings generated from processing the input text(s) (including caption if provided).
  """
  inputs = clip_processor(
      images=image,
      text=[text, caption],
      padding=True,
      return_tensors='pt',
  )

  outputs = clip_model(**inputs)

  img_emb = outputs.image_embeds
  text_emb = outputs.text_embeds[0].unsqueeze(0)
  caption_emb = outputs.text_embeds[1].unsqueeze(0)

  return img_emb, text_emb, caption_emb

def save_embedding(embedding, output_path):
  """
  Save the embedding provided in input in a .npz file.
  ________________Input________________
  output_path: path to the file that will store the embedding
  embedding: embedding file to save
  """
  embedding = embedding.detach().cpu().numpy()
  np.savez_compressed(output_path, emb=embedding)

if __name__ == "__main__":

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Extraction of CLIP-based embeddings.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', required=True, type=str, help='Path to the CSV representing the dataset from which the embeddings will be extracted')
    parser.add_argument('--clip-checkpoint', default='openai/multilingual-clip-vit-large-patch14', type=str, help='Checkpoint of the CLIP model to use when extracting the embeddings')
    parser.add_argument('--output-dir', required=True, type=str, help='Path where to save the extracted embeddings')

    args = parser.parse_args()

    # -- building CLIP model
    clip_model, clip_processor = build_clip(args.clip_checkpoint)

    # -- creating output embedding directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'text'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'caption'), exist_ok=True)

    # -- processing dataset
    dataset = pd.read_csv(args.dataset)
    for idx, sample in tqdm(df.iterrows()):
        sample_id = sample['sample_id']

        # -- extracting embeddings
        image = Image(sample['image_path']
        text = sample['text'].strip().lower()
        caption = sample['caption'].strip().lower()

        img_emb, text_emb, caption_emb = extract_embeddings(image, text, caption)

        # -- saving embeddings
        save_embedding(img_emb, os.path.join(args.output_dir, 'image', f'{sample_id}.npz'))
        save_embedding(text_emb, os.path.join(args.output_dir, 'text', f'{sample_id}.npz'))
        save_embedding(caption_emb, os.path.join(args.output_dir, 'caption', f'{sample_id}.npz'))

