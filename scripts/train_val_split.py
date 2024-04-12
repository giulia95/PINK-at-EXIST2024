import os
import random
import argparse
import pandas as pd
from tqdm import tqdm

def create_csv(sample_ids, output_path):
    split_df = fulltrain_df[fulltrain_df['sample_id'].isin(sample_ids)]
    split_df.to_csv(output_path, sep='\t', index=False)

if __name__ == "__main__":

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='From a full training dataset, this script creates the training and validation sets corresponding to a percentage',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fulltraining-path', required=True, type=str, help='Path to the CSV representing the full-training split')
    parser.add_argument('--percentage', default=0.2, type=float, help='Percentage (between 0 and 1) to split the full-training split')
    parser.add_argument('--output-training-path', required=True, type=str, help='Path where to save the modified training split')
    parser.add_argument('--output-validation-path', required=True, type=str, help='Path where to save the modified validation split')

    args = parser.parse_args()

    fulltrain_df = pd.read_csv(args.fulltraining_path, sep='\t')

    fulltrain_sample_ids = fulltrain_df['sample_id'].tolist()

    num_val_samples = int(len(fulltrain_sample_ids) * args.percentage)

    new_validation_ids = random.sample(fulltrain_sample_ids, num_val_samples)
    new_training_ids = list( set(fulltrain_sample_ids) - set(new_validation_ids)  )

    print(f'The split of {args.percentage*100}% resulted in a TRAIN SET of {len(new_training_ids)} samples and a VALIDATION SET of {len(new_validation_ids)} samples')

    create_csv(new_training_ids, args.output_training_path)
    create_csv(new_validation_ids, args.output_validation_path)
