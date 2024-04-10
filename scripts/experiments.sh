#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
  --config ./configs/pink_transformer.yaml \
  --training-dataset ./splits/EXIST2024/training.json \
  --validation-dataset ./splits/EXIST2024/validation.json \
  --output-dir ./exps/EXIST2024/hard_label_task4/pink_transformer/no_modality_masking/ \
  --output-name validation
