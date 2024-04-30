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

def mapping_gender(gender_annotators):
    gender_map = {'F': 0, 'M': 1}
    return list(map(gender_map.get, gender_annotators))

def mapping_ethnicities(ethnicities_annotators):
    ethnicities_map = {'Asian': 0, 'Hispano or Latino': 1, 'other': 2, 'Middle Eastern': 3, 'Multiracial': 4, 'White or Caucasian': 5, 'Black or African American': 6}
    return list(map(ethnicities_map.get, ethnicities_annotators))

def mapping_study_levels(study_levels_annotators):
    study_levels_map = {'Bachelor’s degree': 0, 'Less than high school diploma': 1, 'other': 2, 'Doctorate': 3, 'High school degree or equivalent': 4, 'Master’s degree': 5}
    return list(map(study_levels_map.get, study_levels_annotators))

def mapping_countries(countries_annotators):
    countries_map = {'Uruguay': 0, 'New Zealand': 1, 'Afghanistan': 2, 'Greece': 3, 'Peru': 4, 'Canada': 5, 'Israel': 6, 'United Kingdom': 7, 'Czech Republic': 8, 'Colombia': 9, 'Morocco': 10, 'Latvia': 11, 'Bangladesh': 12, 'Germany': 13, 'Ecuador': 14, 'Serbia': 15, 'Brazil': 16, 'France': 17, 'Chile': 18, 'United Arab Emirates': 19, 'Nigeria': 20, 'Argentina': 21, 'Hungary': 22, 'Slovenia': 23, 'Mexico': 24, 'United States': 25, 'China': 26, 'Italy': 27, 'Romania': 28, 'Venezuela': 29, 'Finland': 30, 'Portugal': 31, 'South Africa': 32, 'Ireland': 33, 'Cuba': 34, 'Sweden': 35, 'Poland': 36, 'Netherlands': 37, 'India': 38, 'Spain': 39, 'Croatia': 40, 'Russian Federation': 41, 'Zimbabwe': 42, 'Austria': 43, 'Estonia': 44, 'Bulgaria': 45, 'Turkey': 46, 'Belgium': 47, 'Nicaragua': 48, 'Dominican Republic': 49, 'Viet Nam': 50, 'Australia': 51, 'Azerbaijan': 52, 'Iran, Islamic Republic of': 53}
    return list(map(countries_map.get, countries_annotators))

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

        gender_annotators = str(mapping_gender(original_split[sample_id]['gender_annotators']))
        ethnicities_annotators = str(mapping_ethnicities(original_split[sample_id]['ethnicities_annotators']))
        study_levels_annotators = str(mapping_study_levels(original_split[sample_id]['study_levels_annotators']))
        countries_annotators = str(mapping_countries(original_split[sample_id]['countries_annotators']))

        # text = original_split[sample_id]['text']
        # blip_caption = caption_df.loc[caption_df['image_name'] == sample_id]['caption_free'].values[0].strip()

        hard_label_task4, soft_label_task4 = get_labels('task4', original_split[sample_id]['labels_task4']) if 'test' not in args.split_name else (-1, -1)

        new_split.append( (sample_id, lang, clip_image_emb_path, clip_text_emb_path, clip_caption_emb_path, bert_text_emb_path, bert_caption_emb_path, hard_label_task4, soft_label_task4, gender_annotators, ethnicities_annotators, study_levels_annotators, countries_annotators) )

    new_split = pd.DataFrame(new_split, columns=['sample_id', 'lang', 'clip_image_emb_path', 'clip_text_emb_path', 'clip_caption_emb_path', 'bert_text_emb_path', 'bert_caption_emb_path', 'hard_label_task4', 'soft_label_task4', 'gender_annotators', 'ethnicities_annotators', 'study_levels_annotators', 'countries_annotators'])
    new_split.to_csv(args.output_path, sep='\t', index=False)
