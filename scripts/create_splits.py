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
        # -- trick to normalize the soft labels to allocate the actual integers labels for the model
        soft_label = float(str(soft_label)[:6])

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
    new_split = []
    for sample_id in tqdm(original_split.keys()):
        lang = original_split[sample_id]['lang']
        # image_path = os.path.join(args.split_dir, original_split[sample_id]['path_memes'])

        clip_image_emb_path = os.path.join(args.embeddings_dir, 'clip_image', f'{sample_id}.npz')
        clip_text_emb_path = os.path.join(args.embeddings_dir, 'clip_text', f'{sample_id}.npz')
        clip_caption_emb_path = os.path.join(args.embeddings_dir, 'clip_caption', f'{sample_id}.npz')
        bert_text_emb_path = os.path.join(args.embeddings_dir, 'bert_text', f'{sample_id}.npz')
        bert_caption_emb_path = os.path.join(args.embeddings_dir, 'bert_caption', f'{sample_id}.npz')

        # text = original_split[sample_id]['text']
        # blip_caption = caption_df.loc[caption_df['image_name'] == sample_id]['caption_free'].values[0].strip()

        hard_label_task4, soft_label_task4 = get_labels('task4', original_split[sample_id]['labels_task4']) if 'test' not in args.split_name else (-1, -1)

        new_split.append( (sample_id, lang, clip_image_emb_path, clip_text_emb_path, clip_caption_emb_path, bert_text_emb_path, bert_caption_emb_path, hard_label_task4, soft_label_task4) )

    new_split = pd.DataFrame(new_split, columns=['sample_id', 'lang', 'clip_image_emb_path', 'clip_text_emb_path', 'clip_caption_emb_path', 'bert_text_emb_path', 'bert_caption_emb_path', 'hard_label_task4', 'soft_label_task4'])
    new_split.to_csv(args.output_path, sep='\t', index=False)
