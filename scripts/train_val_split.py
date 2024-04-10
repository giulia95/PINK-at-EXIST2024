import os
import random
import json
import argparse
from tqdm import tqdm

def create_json(image_ids, output_path):
    new_split = {}
    for image_id in image_ids:
        new_split[image_id] = fulltraining_split[image_id]

    # -- write new training set json
    with open(output_path, 'w') as f:
        json.dump(new_split, f, indent=2)

if __name__ == "__main__":

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='From a full training dataset, this script creates the training and validation sets corresponding to a percentage',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fulltraining-path', required=True, type=str, help='Path to the JSON representing the full-training split')
    parser.add_argument('--percentage', default=0.2, type=float, help='Percentage (between 0 and 1) to split the full-training split')
    parser.add_argument('--output-training-path', required=True, type=str, help='Path where to save the modified training split')
    parser.add_argument('--output-validation-path', required=True, type=str, help='Path where to save the modified validation split')

    args = parser.parse_args()

    # -- reading json
    split_path = os.path.join(args.fulltraining_path)
    with open(split_path) as f:
        fulltraining_split = json.load(f)

    image_ids = list(fulltraining_split.keys())

    num_val_samples = int(len(image_ids) * args.percentage)

    new_validation_ids = random.sample(image_ids, num_val_samples)
    new_training_ids = list( set(image_ids) - set(new_validation_ids)  )

    print(f'The split of {args.percentage*100}% resulted in a TRAIN SET of {len(new_training_ids)} samples and a VALIDATION SET of {len(new_validation_ids)} samples')

    create_json(new_training_ids, args.output_training_path)
    create_json(new_validation_ids, args.output_validation_path)
