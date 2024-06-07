
# --include localhost:0,1,2,3,4,5,6,7
 deepspeed  --num_gpus 8  run.py \
    --deepspeed ds_zero.json \
    --embedding_model_name qwen2 \
    --output_dir output \
    --model_name_or_path  /data3/home/llm/test/Qwen1.5-7B-Chat \
    --data_dir ./data\
    --cache_dir_data cache_data \
    --learning_rate 2e-5 \
    --fp16 true \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --query_max_len 64 \
    --passage_max_len 512 \
    --remove_unused_columns False \
    --save_strategy epoch \
    --save_total_limit 3 \
    --temperature 0.05 \
    --logging_steps 5 



# CUDA_VISIBLE_DEVICES=0,1,2,3 python hz_run_self.py \
#     --output_dir modeloutput \
#     --embedding_model_name bert \
#     --model_name_or_path model/roberta \
#     --data_dir data/random_neg \
#     --cache_dir_data cache_data \
#     --learning_rate 2e-5 \
#     --fp16 False \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 20 \
#     --query_max_len 64 \
#     --passage_max_len 256 \
#     --remove_unused_columns False \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 3 \
#     --temperature 0.05 \
#     --logging_steps 5 #
