
# Abstract
Large Language Models (LLMs) have exhibited exceptional performance across a spectrum of natural language processing tasks. However, their substantial sizes pose considerable challenges, particularly in terms of computational demands and inference speed, due to its quadratic complexity. In this work, we have identified a noteworthy pattern: certain seemingly meaningless special tokens (i.e., separators) contribute disproportionately to attention scores compared to other semantically meaningful tokens. This insight has led us to hypothesize that the information of the segment between these special tokens can be condensed into these tokens without significant loss of information. Based on this hypothesis, we introduce SegLLM, a plug-and-play framework for inference acceleration by compressing these segments and dropping redundant tokens. The experimental results on training-free, training-from-scratch and post-training settings substantiate the effectiveness of SegLLM. Notably, SegLLM achieves a remarkable reduction of over 50\% in KV cache on the GSM8K-CoT benchmark, utilizing the Llama-3-8B backbone, with negligible compromise in performance. Additionally, in streaming settings, SegLLM is capable of delivering consistent and effective language modeling across up to 4 million tokens or even more.  This advancement underscores the potential of our approach to optimize LLMs for practical applications.

![image](https://hackmd.io/_uploads/BkOQCLEX1e.png)

# Usage

You can install the required package in the requirements.txt. You are recommended to build a independent conda environment (or pyenv, etc.) to do this. Our code is based on the code framework [GPTNeoX](https://github.com/EleutherAI/gpt-neox).

## For training

![image](https://hackmd.io/_uploads/r18jZD47Jg.png)


All the code corresponding to training is in the Training-SegLLM folder. If you want to use the fused operators, just run:
```
cd Training-SegLLM
pip install -r requirements/requirements.txt
python ./megatron/fused_kernels/setup.py install # optional if not using fused kernels
```
*Note: If you want to use the Seg-Attention module, please make sure your Pytorch>=2.5.0. And set "USE_FLEX=True" in your training config file.*

You can start training by:
```
python ./deepy.py train.py [path/to/config.yml]
```
The sample configuration yml files are in ./Training-SegLLM/sample_configs.

### Parameter Settings for SegLLM Training

```
@dataclass ###my
class SegmentMaskArgs(NeoXArgsTemplate):
    """
    Our segment mask args when training
    """


    special_tokens_id: list = None
    """
    The token ids for the special tokens:  ['.', ',', '?', '!', ';', ":", ' ', '\t','\n'] 
    """
    
    PADDING_ID:  int = 0  # For pythia (GPT_NeoX)    
    """
    The id for padding token of pythia (GPT_NeoX)
    """


    prefill_k: int = 0  ## NOT implemented yet; From old version: Deprecated          
    decode_k: int  = 0  ## NOT implemented yet; From old version: Deprecated
    """
    The max layers (excluded) that use the original attention masks when prefilling and generating respectively. These two args are NOT implemented yet and deprecated.
    For now, put a large number (>max_seq_len) for the corresponding layers in prefill_win_size_list (or decode_win_size_list) if you want to keep the entire layer's KV and attentions
    """

    prefill_window_size: int  = 256  
    """
    The local window size when prefilling (tokens inside the window are kept)

    Only take effect when USE_DYNAMIC_PREFILL_WINDOW_SIZE=False
    """
    
    decode_window_size: int  = 256 
    """
    The local window size when generating (tokens inside the window are kept)
    
    Only take effect when USE_DYNAMIC_DECODE_WINDOW_SIZE=False        
    """


    USE_DYNAMIC_PREFILL_WINDOW_SIZE: bool = False
    """
    If True: the prefilling window sizes for different decoder layers are different.
    If True: should set 'prefill_win_size_list', else: should set 'prefill_window_size'
    """

    USE_DYNAMIC_DECODE_WINDOW_SIZE: bool = False 
    """
    If True: the decoding window sizes for different decoder layers are different.
    If True: should set 'decode_win_size_list', else: should set 'decode_window_size'
    """



    prefill_win_size_list: list = None
    """
    The prefilling window sizes for different decoder layers
    """

    decode_win_size_list: list = None
    """
    The decoding window sizes for different decoder layers
    """

    att_sink_max_idx: int = 2 
    """
    The largest index for attention sink tokens. E.g., if att_sink_max_idx==2, it means we use 3 tokens (idx: 0,1,2) as the attention sinks
    """


    ######################################There should be at most 1 True for the following 2 args ##############################################
    original_flag: bool = False  
    """
    Run the model without any modification if True
    """

    streamingLLM: bool = False  
    """
    Run streamingLLM. Only takes effect when original_flag=False. 
    """
    ######################################There should be at most 1 True for the above 2 args ##############################################
    

    BATCH_DYNAMIC_ATT_SINK: bool = False 
    """
    If True: use the floating attension sink positions since when evaluating, LLM usually add paddings on the left side of the shorter seqs in a batch for alignment (i.e., left padding).
    
    Can be False when pretraining since the attention sinks are at the beginning of each seq in batch for pretraining (i.e., right padding)
    """



    PRINT_KV_RATIO: bool = False 
    """
    If True, print the KV cache preservation ratio (especially for generating). When pretraining, it will print the retention ratio for the computational complexity of calculating the attention map if it is set True
    """

    print_KV_intervals: int = 4000
    """    
    Print the retention ratio for the computational complexity of calculating the attention map once after every 'print_KV_intervals' forward passes. It only takes effect when PRINT_KV_RATIO=True.    
    """

    USE_FLEX: bool = False 
    """
    If True, use Flex-attention. MUST be set True if you want to use Seg-Attention module to accelerate training.
    """

    EXCLUDE_DIAGONAL: bool = True ## From old version: Deprecated
    """
    True by default. Only take effects when self.floating_window = False. When True, it means when we choose fixed window to process the prefilling mask, the diagonal elements in the prefilling mask could be set negative (like yihang's code). When False: would keep the prefilling mask's diagonal positive
    """

    floating_window: bool = True ## NOT fully implemented yet; From old version: Deprecated
    """
    
    True by default, which means using floating window when prefilling. False means using fixed window (on the rightmost side for all rows of the prefilling mask matrix) when prefilling, which generates better performance sometimes however :)
    """
```

Remember to save your training process checkpoints, so that if the training is interrupted unexpectedly, you can resume the training. You can set the save dir in the config yml file.
```
  "save": "path/to/checkpoints",
  "load": "path/to/checkpoints",
```


After the training is completed, we can convert the training checkpoints to the Hugging Face format, so that we can test them on downstream tasks （e.g. using [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness)）.

```
python ./tools/ckpts/convert_neox_to_hf.py --input_dir path/to/checkpoints/global_stepXXX --config_file your_config.yml --output_dir hf_model/save/dir
```

# Long Streaming Test
Our long streaming evaluation is following [StreamingLLM](https://github.com/mit-han-lab/streaming-llm/).

## Usage

```
conda create -yn streaming-segllm python=3.8
conda activate streaming-segllm 

pip install torch torchvision torchaudio # we use torch==2.1.0+cu121 for streaming test.
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

python setup.py develop
```
And to evaluate Streaming-SegLLM, you can follow this example:
```
CUDA_VISIBLE_DEVICES=0 python ./main/evaluate_streaming_perplexity.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B\
    --neighboring_size 224 \
    --initial_size 4 \
    --cache_size 324 \
    --sep_cache_size 32 \
    --enable_kv_cache_manager True \
    --enable_segmented_LLM True \
    --enable_pos_shift True \
    --num_samples 5000000 \
    --num_eval_tokens 20480 \
    --dataset_name pg19 \
    --task default \
    --split test\
    --output_dir ./outputs/sth   2>&1 | tee ./logs/sth.log
```
You can see other examples under ./Streaming-SegLLM/scripts/
