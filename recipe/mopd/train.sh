#!/usr/bin/env bash
set -x

STUDENT=${STUDENT:-"Qwen/Qwen2.5-7B-Instruct"}
TEACHER=${TEACHER:-"Qwen/Qwen2.5-14B-Instruct"}
DATA_DIR=${DATA_DIR:-"$HOME/data/gsm8k"}
TRAIN_FILES="['${DATA_DIR}/train.parquet']"
VAL_FILES="['${DATA_DIR}/test.parquet']"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=mopd \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  critic.enable=False \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.train_batch_size=512 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=${STUDENT} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  teacher.enable=True \
  teacher.config.actor_rollout_ref.model.path=${TEACHER} \
  teacher.config.actor_rollout_ref.actor.strategy=fsdp \
  teacher.config.actor_rollout_ref.rollout.name=vllm \
  teacher.config.actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  teacher.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=mopd-demo \
  trainer.experiment_name=qwen2_7b_mopd \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.val_before_train=False \
  trainer.save_freq=10 \
  trainer.test_freq=10