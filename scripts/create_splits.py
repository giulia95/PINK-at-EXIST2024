import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":

    # Run example: python scripts/create_splits.py --split-dir ./data/EXIST2024/EXIST_2024_Memes_Dataset/training/ --split-name EXIST2024_training.json --embeddings-dir ./data/EXIST2024/embeddings/ --output-path ./splits/EXIST2024/training.json

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Create a JSON from the original split adding new necessary items for our code.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--split-dir', default='data/EXIST2024/EXIST_2024_Memes_Dataset/training/', type=str, help='Directory where the dataset split is stored')
    parser.add_argument('--split-name', default='EXIST2024_training.json', type=str, help='Name of the original JSON split')
    parser.add_argument('--embeddings-dir', default='data/EXIST2024/embeddings/', type=str, help='Directory where the embeddings are stored')
    parser.add_argument('--output-path', required=True, type=str, help='Path where to save the modified split')

    args = parser.parse_args()

    # -- reading json
    split_path = os.path.join(args.split_dir, args.split_name)
    with open(split_path) as f:
        original_split = json.load(f)

    # -- reading caption csv
    # -- TODO: add code to obtain the BLIP captions (only for the test set)
    caption_df = pd.read_csv('data/EXIST2024/EXIST_2024_Memes_Dataset/blip_captions.csv', sep="\t")

    # -- iterating samples
    for image_id in tqdm(original_split.keys()):
        original_split[image_id]['image_path'] = os.path.join(args.split_dir, original_split[image_id]['path_memes'])

        # TODO: replace this line with the code adequate to the previous todo comment
        original_split[image_id]['blip_caption'] = caption_df.loc[caption_df['image_name'] == image_id]['caption_free'].values[0].strip()

        original_split[image_id]['img_emb_path'] = os.path.join(args.embeddings_dir, 'image', f'{image_id}.npz')
        original_split[image_id]['text_emb_path'] = os.path.join(args.embeddings_dir, 'text', f'{image_id}.npz')
        original_split[image_id]['caption_emb_path'] = os.path.join(args.embeddings_dir, 'caption', f'{image_id}.npz')

    # -- write new training set json
    with open(args.output_path, 'w') as f:
        json.dump(original_split, f, indent=2)

