PRE_SEQ_LEN=128
LR=5e-3
NUM_GPUS=1

# --train_file AdvertiseGen/train.json \
# --validation_file AdvertiseGen/dev.json \
#    --quantization_bit 4
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --do_predict \
    --train_file /root/autodl-tmp/LLMs/data/win64-texts-codes-texts/train.json \
    --validation_file /root/autodl-tmp/LLMs/data/win64-texts-codes-texts/test.json \
    --test_file /root/autodl-tmp/LLMs/data/win64-texts-codes-texts/test.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path ../GLM2-6b-model \
    --output_dir output/fullfeatures \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 512 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --max_steps 300000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --report_to tensorboard \