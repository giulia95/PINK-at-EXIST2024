import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

LABEL_MAPPINGS = {
    # -- task 4: sexism detection in memes
    'task4': {
        'NO': 0,
        'YES': 1
    },

    # -- task 5: source detection in memes
    'task5': {
        'DIRECT': 0,
        'JUDGEMENTAL': 1,
    },

    # -- task 6: sexism categorization in memes
    'task6': {
        'IDEOLOGICAL AND INEQUALITY': 0,
        'STEREOTYPING AND DOMINANCE': 1,
        'OBJECTIFICATION': 2,
        'SEXUAL VIOLENCE': 3,
        'MISOGYNY AND NON-SEXUAL VIOLENCE': 4,
    }

}

def get_labels(task_id, labels):
    label_mapping = LABEL_MAPPINGS[task_id]
    mapped_labels = list( map(label_mapping.get, labels)  )

    # TODO: special case for task5 and task6
    if task_id in ['task4']:
        count_zeros = mapped_labels.count(0)
        count_ones = mapped_labels.count(1)

        hard_label = 0 if count_zeros > count_ones else 1
        soft_label = count_ones / len(labels) if count_ones > 0 else 0

        return hard_label, soft_label

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

        for task_id in ['task4']: # TODO: task5 and task6
            hard_label, soft_label = get_labels(task_id, original_split[image_id][f'labels_{task_id}'])
            original_split[image_id][f'hard_label_{task_id}'] = hard_label
            original_split[image_id][f'soft_label_{task_id}'] = soft_label

    # -- write new training set json
    with open(args.output_path, 'w') as f:
        json.dump(original_split, f, indent=2)

