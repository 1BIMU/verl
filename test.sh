set -x


export CUDA_VISIBLE_DEVICES=0

train_path=data/dapo-math-17k_dedup.parquet
aime_test_path=data/offline_eval/math__aime_repeated_8x_240.parquet
math_test_path=data/offline_eval/math__math_500.parquet

train_files="['$train_path']"
test_files="['$aime_test_path', '$math_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=sequence_level_adv \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/workspace/models/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.model.path=/workspace/models/Qwen2.5-0.5B-Instruct \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.default_local_dir='/workspace/Reasoning360/SQPPO' \
    trainer.project_name='PPO' \
    trainer.experiment_name='qwen0.5B_PPO_SEQUENCE' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 $@
