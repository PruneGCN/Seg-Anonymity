python ../../../examples/eval_long_ppl.py \
    --neighboring_size 400 \
    --initial_size 4 \
    --cache_size 516 \
    --sep_cache_size 32 \
    --enable_kv_cache_manager True \
    --enable_segmented_LLM True \
    --enable_pos_shift True \
    --num_samples 500000000000000000 \
    --num_eval_tokens 5120 \
    --dataset_name pg19 \
    --task default \
    --split test\
    --output_dir ../../../outputs/debug/llama3_seg_debug   2>&1 | tee ../../../my_logs/debug/llama3_seg_debug.log

# --output_dir ../../../outputs/llama3/pg19/seg_len10M_sink4_seg32_win400_ca516   2>&1 | tee ../../../my_logs/llama3/pg19/seg_len10M_sink4_seg32_win400_ca516.log

