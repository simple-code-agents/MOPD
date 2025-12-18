python -m verl.trainer.main_ppo \
  trainer.default_local_dir=./ckpts/distill_run \
  data.train_files="[/path/to/prompts.jsonl]" \
  actor_rollout_ref.model.path=/path/to/student_init \
  actor_rollout_ref.ref_model_path=/path/to/teacher_model \
  algorithm.on_policy_distill.enable=true \
  algorithm.adv_estimator=on_policy_distill \
  algorithm.gamma=0.0 algorithm.lam=1.0 \
  reward_model.enable=false \
  critic.enable=false \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0