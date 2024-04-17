#!/bin/bash

# BASELINE
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config ./configs/pink_transformer.yaml \
  --training-dataset ./splits/EXIST2024/training.csv \
  --validation-dataset ./splits/EXIST2024/validation.csv \
  --output-dir ./exps/EXIST2024/hard_label_task4/pink_transformer/baseline/ \
  --output-name validation \
  --yaml-overrides task:hard_label_task4 > ./logs/baseline.log

# BASELINE + MODALITY
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config ./configs/pink_transformer.yaml \
  --training-dataset ./splits/EXIST2024/training.csv \
  --validation-dataset ./splits/EXIST2024/validation.csv \
  --output-dir ./exps/EXIST2024/hard_label_task4/pink_transformer/baseline+modality/ \
  --output-name validation \
  --yaml-overrides task:hard_label_task4 model_conf:use_modality_emb:true > ./logs/baseline+modality.log

# BASELINE + MODALITY + LANGUAGE
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config ./configs/pink_transformer.yaml \
  --training-dataset ./splits/EXIST2024/training.csv \
  --validation-dataset ./splits/EXIST2024/validation.csv \
  --output-dir ./exps/EXIST2024/hard_label_task4/pink_transformer/baseline+modality+language/ \
  --output-name validation \
  --yaml-overrides task:hard_label_task4 model_conf:use_modality_emb:true model_conf:use_language_emb:true > ./logs/baseline+modality+language.log

# BASELINE + MODALITY + LANGUAGE + MASKING
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config ./configs/pink_transformer.yaml \
  --training-dataset ./splits/EXIST2024/training.csv \
  --validation-dataset ./splits/EXIST2024/validation.csv \
  --output-dir ./exps/EXIST2024/hard_label_task4/pink_transformer/baseline+modality+language+masking/ \
  --output-name validation \
  --yaml-overrides task:hard_label_task4 model_conf:use_modality_emb:true model_conf:use_language_emb:true model_conf:modality_masking_prob:0.7 > ./logs/baseline+modality+language+masking.log


