# # --include localhost:0,1,2,3,4,5,6,7
num_processes=1 deepspeed  --num_gpus 8 train_sft.py \
    --deepspeed ds_zero_no_affload.json \
    --model_name_or_path /data3/home/llm/test/Qwen1.5-7B-Chat \
    --use_lora true \
    --data_path "./data" \
    --bf16 true \
    --fp16 false \
    --output_dir output_qwen\
    --num_train_epochs 5 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 4e-4 \
    --logging_steps 10 \
    --tf32 False \


