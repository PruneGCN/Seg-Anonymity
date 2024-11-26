python ../examples/eval_long_ppl.py \
    --recent_size 320 \
    --start_size 4 \
    --cache_size 324 \
    --sep_cache_size 0 \
    --enable_start_recent_kv_cache True \
    --enable_segmented_LLM False \
    --enable_pos_shift True \
    --num_samples 5000000 \
    --num_eval_tokens 20480 \
    --dataset_name wikitext \
    --task wikitext-2-raw-v1 \
    --output_dir ../outputs/wikitext/streamingllm_len20k_sink4_324   2>&1 | tee ../my_logs/wikitext/streamingllm_len20k_sink4_324.log