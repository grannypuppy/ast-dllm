# python eval/gen_eval.py \
#     --model_path ../dLLM-RL/local_models/dream-7b-base \
#     --output_dir results/dream-7b-base_len512_steps512 \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 512 \
#     --top_p None \
#     --threshold None \
#     --num_generations 1 \
#     --use_cache False

# python eval/gen_eval.py \
#     --model_path ../dLLM-RL/projects/sft_dream_py/checkpoint-200 \
#     --output_dir results/sft_dream_py_len512_steps512 \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 512 \
#     --top_p None \
#     --threshold None \
#     --num_generations 1 \
#     --use_cache False


# export CUDA_VISIBLE_DEVICES=2

# python eval/gen_eval.py \
#     --model_path ../dLLM-RL/projects/sft_dream_dag/1epoch_1e-5lr_1lm_8ga_4gpus/checkpoint-200 \
#     --output_dir results/sft_dream_dag_len512_steps512 \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 512 \
#     --top_p None \
#     --threshold None \
#     --num_generations 1 \
#     --use_cache False

# torchrun --nproc_per_node=4 eval/gen_eval.py \
#     --model_path ../dLLM-RL/projects/sft_dream_cfg/1epoch_1e-5lr_1lm_8ga_4gpus/checkpoint-200 \
#     --output_dir results/sft_dream_cfg_len512_steps512_ckpt200 \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 512 \
#     --top_p None \
#     --threshold None \
#     --num_generations 1 \
#     --use_cache False

torchrun --nproc_per_node=4 eval/gen_eval.py \
    --model_path ../dLLM-RL/projects/sft_dream_py-official/global_step_50 \
    --output_dir results/sft_dream_py-official_len512_steps512_ckpt50 \
    --max_new_tokens 512 \
    --diffusion_steps 512 \
    --block_size 512 \
    --top_p None \
    --threshold None \
    --num_generations 1 \
    --use_cache False