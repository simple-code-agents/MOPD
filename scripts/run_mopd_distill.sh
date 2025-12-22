#!/usr/bin/env bash
# Minimal on-policy distillation launch for VERL (student on-policy samples, teacher dense logprobs).
# Fill the placeholders below before running.

set -euo pipefail

# === Required paths (edit) ===
STUDENT_CKPT="/path/to/student/init"   # e.g., SFT checkpoint or latest student
TEACHER_CKPT="/path/to/teacher"        # teacher weights; must share tokenizer/chat template
TRAIN_DATA="/path/to/train.jsonl"      # prompt-only RLHF-format data (ignored labels)
VAL_DATA="/path/to/val.jsonl"          # optional; can mirror TRAIN_DATA

# === Cluster settings (edit to your env) ===
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export RAY_memory_monitor_refresh_ms=0

python -m verl.trainer.main_ppo \
  trainer.project_name=mopd \
  trainer.experiment_name=mopd_on_policy \
  trainer.default_local_dir=outputs/mopd_on_policy \
  trainer.nnodes=${NNODES} \
  trainer.n_gpus_per_node=${GPUS_PER_NODE} \
  trainer.master_addr=${MASTER_ADDR} trainer.master_port=${MASTER_PORT} \
  data.train_files=[${TRAIN_DATA}] \
  data.val_files=[${VAL_DATA}] \
  actor_rollout_ref.model.path=${STUDENT_CKPT} \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.max_new_tokens=512 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.loss_scale_factor=1.0 \
  actor_rollout_ref.actor.use_kl_loss=false \
  teacher_enable=true \
  teacher.model.path=${TEACHER_CKPT} \
  teacher.strategy=fsdp \
  algorithm.adv_estimator=mopd \
  algorithm.gamma=1.0 algorithm.lam=1.0 \
  reward_model.enable=false \
  critic.enable=false \
  global_profiler.tool=null \
  +trainer.ray_wait_register_center_timeout=300

# Notes:
# - For FSDP teacher/student, ensure consistent tokenizer/template; if using vLLM teacher, swap strategy and worker class accordingly.
# - Set rollout.n, max_new_tokens, and micro_batch sizes to match your GPU memory; above is conservative.
# - If using multiple nodes, launch via ray/torch run launcher in your environment; this script assumes a single driver invoking Ray inside.
