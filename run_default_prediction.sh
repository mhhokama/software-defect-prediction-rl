#!/bin/bash

# Run the Python script for fault prediction with specified parameters
python3 -m scripts.fault-prediction \
  --tb_name custom-lucen-2-9_3-0_auc_logistic_SMOTE-test \
  --state custom \
  --max_features 10 \
  --min_criteria 94 \
  --reward_mode accuracy \
  --features_dim 16 \
  --n_steps 10 \
  --total_timesteps 100 \
  --classifier random_forest
