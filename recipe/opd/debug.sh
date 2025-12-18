python -m verl.trainer.main_ppo \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  data.train_batch_size=1 actor_rollout_ref.rollout.n=1 \
  algorithm.on_policy_distill.enable=true algorithm.adv_estimator=on_policy_distill \
  reward_model.enable=false critic.enable=false \
  actor_rollout_ref.model.path=/mnt/bn/douyin-ai4se-general-wl/model/Qwen2.5-Coder-1.5B \
  actor_rollout_ref.ref_model_path=/mnt/bn/douyin-ai4se-general-wl/model/Qwen2.5-Coder-3B \
  data.train_files=/mnt/bn/douyin-ai4se-general-wl/lht/project/MOPD/data/examples.jsonl \
  trainer.total_training_steps=10