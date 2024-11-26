CUDA_VISIBLE_DEVICES=0  python ./main/evaluate_streaming_perplexity.py \
    --local_size 256 \
    --initial_size 4 \
    --cache_size 800 \
    --sep_cache_size 64 \
    --enable_kv_cache_manager False \
    --enable_segmented_LLM False \
    --enable_pos_shift True \
    --num_samples 5000000 \
    --num_eval_tokens 65535 \
    --dataset_name pg19 \
    --task default \
    --split test\
    --output_dir ./outputs/debug/pg19_seg_len65535_sink4_seg64_win256_ca800_shift_infinite_timetest_detailed_2cards_90G_Vis1_debug   2>&1 | tee ./my_logs/debug/pg19_seg_len65535_sink4_seg64_win256_ca800_shift_infinite_timetest_detailed_2cards_90G_Vis1_debug.log


# --dataset_name pg19 \
# --task default \
# --split test\

# --dataset_name wikitext \
# --task wikitext-2-raw-v1 \

# parser.add_argument("--dataset_name", type=str, default="wikitext")  ##deafult
# parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")   ## default