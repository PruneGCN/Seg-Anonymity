python ../../examples/eval_long_ppl.py \
    --recent_size 224 \
    --start_size 0 \
    --cache_size 324 \
    --sep_cache_size 32 \
    --enable_start_recent_kv_cache True \
    --enable_segmented_LLM True \
    --enable_pos_shift True \
    --num_samples 5000000 \
    --num_eval_tokens 20480 \
    --dataset_name wikitext \
    --task wikitext-2-raw-v1 \
    --output_dir ../../outputs/wikitext/seg_len20k_sink0_seg32_win224_ca324_nosink_add_all_SOS   2>&1 | tee ../../my_logs/wikitext/seg_len20k_sink0_seg32_win224_ca324_nosink_add_all_SOS.log