python ../../../main/evaluate_streaming_perplexity.py \
    --neighboring_size 224 \
    --initial_size 4 \
    --cache_size 324 \
    --sep_cache_size 48 \
    --enable_kv_cache_manager True \
    --enable_segmented_LLM True \
    --enable_pos_shift True \
    --num_samples 5000000 \
    --num_eval_tokens 20480 \
    --dataset_name wikitext \
    --task wikitext-2-raw-v1 \
    --output_dir ../../../outputs/debug/seg_len20k_sink4_seg48_win224_ca324_debug   2>&1 | tee ../../../my_logs/debug/seg_len20k_sink4_seg48_win224_ca324_debug.log


# parser.add_argument("--dataset_name", type=str, default="wikitext")  ##deafult
# parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")   ## default