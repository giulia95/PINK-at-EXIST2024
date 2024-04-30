import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":

    # Run example: python scripts/create_splits.py --split-dir ./data/MAMI/TRAINING/ --split-name training.csv --embeddings-dir ./data/MAMI/embeddings/training/ --output-path ./splits/MAMI/training.csv

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Create a CSV from the original split adding new necessary items for our code.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--split-dir', default='data/MAMI/TRAINING/', type=str, help='Directory where the dataset split is stored')
    parser.add_argument('--split-name', default='training.csv', type=str, help='Name of the original CSV split')
    parser.add_argument('--caption-csv', default='data/MAMI/blip_captions_MAMI_train.csv', type=str, help='Path to the CSV containing captions')
    parser.add_argument('--embeddings-dir', default='data/MAMI/embeddings/training/', type=str, help='Directory where the embeddings are stored')
    parser.add_argument('--output-path', required=True, type=str, help='Path where to save the modified split')

    args = parser.parse_args()

    # -- reading json
    split_path = os.path.join(args.split_dir, args.split_name)
    original_split = pd.read_csv(split_path, sep='\t')

    # -- reading caption csv
    # -- TODO: add code to obtain the BLIP captions (only for the test set)
    caption_df = pd.read_csv(args.caption_csv, sep="\t")

    # -- iterating samples
    new_split = []
    for idx, sample in tqdm(original_split.iterrows()):
        lang = 'en'
        sample_id = sample['file_name'].split('.')[0]

        clip_image_emb_path = os.path.join(args.embeddings_dir, 'clip_image', f'{sample_id}.npz')
        clip_text_emb_path = os.path.join(args.embeddings_dir, 'clip_text', f'{sample_id}.npz')
        clip_caption_emb_path = os.path.join(args.embeddings_dir, 'clip_caption', f'{sample_id}.npz')
        bert_text_emb_path = os.path.join(args.embeddings_dir, 'bert_text', f'{sample_id}.npz')
        bert_caption_emb_path = os.path.join(args.embeddings_dir, 'bert_caption', f'{sample_id}.npz')

        hard_label_task4, soft_label_task4 = sample['misogynous'], 0.5

        new_split.append( (sample_id, lang, clip_image_emb_path, clip_text_emb_path, clip_caption_emb_path, bert_text_emb_path, bert_caption_emb_path, hard_label_task4, soft_label_task4) )

    new_split = pd.DataFrame(new_split, columns=['sample_id', 'lang', 'clip_image_emb_path', 'clip_text_emb_path', 'clip_caption_emb_path', 'bert_text_emb_path', 'bert_caption_emb_path', 'hard_label_task4', 'soft_label_task4'])
    new_split.to_csv(args.output_path, sep='\t', index=False)
