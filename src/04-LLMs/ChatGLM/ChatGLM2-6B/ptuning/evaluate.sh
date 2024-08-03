PRE_SEQ_LEN=128
CHECKPOINT=win64-rawfeatures-test
STEP=10000
NUM_GPUS=1

# --validation_file AdvertiseGen/dev.json \
# --test_file AdvertiseGen/dev.json \
# --quantization_bit 4
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_predict \
    --validation_file /root/autodl-tmp/LLMs/data/win64-rawfeatures/test.json \
    --test_file /root/autodl-tmp/LLMs/data/win64-rawfeatures/test.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path ../GLM2-6b-model \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 128 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
