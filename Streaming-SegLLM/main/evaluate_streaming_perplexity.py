import torch
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_segllm.kv_cache_manager import SegLLM_KVCache_Manager
from streaming_segllm.utils import parse_args, load, print_args
import time


device = "cuda:0"

args = parse_args()
if args.dataset_name.lower() == 'pg19':
    args.dataset_name = '/lustre/fast/fast/txiao/shihan/pg19/deepmind-gutenberg/deepmind-gutenberg'
data = load_dataset(args.dataset_name, args.task, split=args.split)
# data = load_dataset('PG19', split='test')


model, tokenizer = load(args.model_name_or_path)

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None

if args.enable_kv_cache_manager:
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "pythia" in model.config.model_type or "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
    else:
        raise ValueError(f"got {model.config.model_type}")
    kv_cache = SegLLM_KVCache_Manager(
        initial_size=args.initial_size,
        local_size=args.local_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
        cache_size = args.cache_size,
        sep_cache_size = args.sep_cache_size,
        model_type = model.config.model_type
    )
else:
    kv_cache = None

if args.enable_pos_shift:
    if "llama" in model.config.model_type:
        from streaming_segllm.pos_shift.modify_llama import enable_llama_pos_shift_attention

        enable_llama_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        from streaming_segllm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    elif "gpt_neox" in model.config.model_type:
        from streaming_segllm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        pass
    else:
        raise ValueError(f"got {model.config.model_type}")


os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")


# print("f##################data: {data}#####################")##my

print_args(args)

num_eval_tokens = 0
total_infer_time0 = 0
total_infer_time1 = 0
total_infer_time2 = 0
for text in data["text"][: args.num_samples]:
    
    # print(f"text: {text}")
        
    
    encodings = tokenizer(text, return_tensors="pt")
    
    # #######################################my############################################
    # for tok_id in encodings.input_ids.tolist()[0]:
    #     print("^^^^^^^^^^^^^^^^^^^^^^^^")
    #     print(f"tok_id {tok_id}:  {tokenizer.decode(tok_id)} \n")
    #     print("************************")

    # # print(encodings.input_ids[:, :10])
    # print(f"encodings.input_ids: {encodings.input_ids}") ##my
    # # special_tokens_id = [13, 11, 30, 0, 26, 25, 198, 220, 128000, 662, 1174, 949, 758, 2652, 551, 720, 256,262] # llama3 8b

    # special_tokens_id = [15, 13, 32, 2, 28, 27, 209, 186, 187] # pythia-14b
    # special_tokens = ['.', ',', '?', '!', ';', ':', '\n']
    
    # special_tokens_new1 = [' .', ' ,', ' ?', ' !', ' ;', ' :', ' \n', ' ', '  ']
    # special_tokens_new2 = ['. ', ', ', '? ', '! ', '; ', ': ', '\n ', ' ', '  ']
    # special_tokens_new3 = [' . ', ' , ', ' ? ', ' ! ', ' ; ', ' : ', ' \n ', ' '  , '  ', '   ']
    
    ####################################################################################
    # for sep in special_tokens_id:
    #     print(f"^^^^^^^^^^^^^for id {sep}^^^^^^^^^^^^^^^^^")
    #     print(tokenizer.decode(sep))
    #     print(f"^^^^^^^^^^^##for id {sep}##^^^^^^^^^^^^^^^")
    
    
    
    
    # ^^^^^^^^^^^^^in special_tokens_new1 for tok  .^^^^^^^^^^^^^^^^^
    # [128000, 662]
    # ^^^^^^^^^^^##in special_tokens_new1 for tok  .##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new1 for tok  ,^^^^^^^^^^^^^^^^^
    # [128000, 1174]
    # ^^^^^^^^^^^##in special_tokens_new1 for tok  ,##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new1 for tok  ?^^^^^^^^^^^^^^^^^
    # [128000, 949]
    # ^^^^^^^^^^^##in special_tokens_new1 for tok  ?##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new1 for tok  !^^^^^^^^^^^^^^^^^
    # [128000, 758]
    # ^^^^^^^^^^^##in special_tokens_new1 for tok  !##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new1 for tok  ;^^^^^^^^^^^^^^^^^
    # [128000, 2652]
    # ^^^^^^^^^^^##in special_tokens_new1 for tok  ;##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new1 for tok  :^^^^^^^^^^^^^^^^^
    # [128000, 551]
    # ^^^^^^^^^^^##in special_tokens_new1 for tok  :##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new1 for tok  
    # ^^^^^^^^^^^^^^^^^
    # [128000, 720]
    # ^^^^^^^^^^^##in special_tokens_new1 for tok  
    # ##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new2 for tok . ^^^^^^^^^^^^^^^^^
    # [128000, 13, 220]
    # ^^^^^^^^^^^##in special_tokens_new2 for tok . ##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new2 for tok , ^^^^^^^^^^^^^^^^^
    # [128000, 11, 220]
    # ^^^^^^^^^^^##in special_tokens_new2 for tok , ##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new2 for tok ? ^^^^^^^^^^^^^^^^^
    # [128000, 30, 220]
    # ^^^^^^^^^^^##in special_tokens_new2 for tok ? ##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new2 for tok ! ^^^^^^^^^^^^^^^^^
    # [128000, 0, 220]
    # ^^^^^^^^^^^##in special_tokens_new2 for tok ! ##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new2 for tok ; ^^^^^^^^^^^^^^^^^
    # [128000, 26, 220]
    # ^^^^^^^^^^^##in special_tokens_new2 for tok ; ##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new2 for tok : ^^^^^^^^^^^^^^^^^
    # [128000, 25, 220]
    # ^^^^^^^^^^^##in special_tokens_new2 for tok : ##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new2 for tok 
    # ^^^^^^^^^^^^^^^^^
    # [128000, 198, 220]
    # ^^^^^^^^^^^##in special_tokens_new2 for tok 
    # ##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new3 for tok  . ^^^^^^^^^^^^^^^^^
    # [128000, 662, 220]
    # ^^^^^^^^^^^##in special_tokens_new3 for tok  . ##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new3 for tok  , ^^^^^^^^^^^^^^^^^
    # [128000, 1174, 220]
    # ^^^^^^^^^^^##in special_tokens_new3 for tok  , ##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new3 for tok  ? ^^^^^^^^^^^^^^^^^
    # [128000, 949, 220]
    # ^^^^^^^^^^^##in special_tokens_new3 for tok  ? ##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new3 for tok  ! ^^^^^^^^^^^^^^^^^
    # [128000, 758, 220]
    # ^^^^^^^^^^^##in special_tokens_new3 for tok  ! ##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new3 for tok  ; ^^^^^^^^^^^^^^^^^
    # [128000, 2652, 220]
    # ^^^^^^^^^^^##in special_tokens_new3 for tok  ; ##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new3 for tok  : ^^^^^^^^^^^^^^^^^
    # [128000, 551, 220]
    # ^^^^^^^^^^^##in special_tokens_new3 for tok  : ##^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^in special_tokens_new3 for tok  
    # ^^^^^^^^^^^^^^^^^
    # [128000, 720, 220]
    # ^^^^^^^^^^^##in special_tokens_new3 for tok  
    # ##^^^^^^^^^^^^^^^
    
    
    
    
    
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    
    # for tok in special_tokens_new1:
    #     print(f"^^^^^^^^^^^^^in special_tokens_new1 for tok {tok}^^^^^^^^^^^^^^^^^")
    #     print(tokenizer.encode(tok))
    #     print(f"^^^^^^^^^^^##in special_tokens_new1 for tok {tok}##^^^^^^^^^^^^^^^")
    
    # for tok in special_tokens_new2:
    #     print(f"^^^^^^^^^^^^^in special_tokens_new2 for tok {tok}^^^^^^^^^^^^^^^^^")
    #     print(tokenizer.encode(tok))
    #     print(f"^^^^^^^^^^^##in special_tokens_new2 for tok {tok}##^^^^^^^^^^^^^^^")

    # for tok in special_tokens_new3:
    #     print(f"^^^^^^^^^^^^^in special_tokens_new3 for tok {tok}^^^^^^^^^^^^^^^^^")
    #     print(tokenizer.encode(tok))
    #     print(f"^^^^^^^^^^^##in special_tokens_new3 for tok {tok}##^^^^^^^^^^^^^^^")

    

    seq_len = encodings.input_ids.size(1)
    # print(f"seq_len: {seq_len}")

    pbar = tqdm(range(0, seq_len - 1))
    for idx in pbar:
    # for idx in range(0, seq_len - 1):
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        
        ##########################my###############################
        # if past_key_values is not None:
        #     # #past_key_values[0][0].shape:torch.Size([1, 8, 129, 128])  before
        #     print(f"###########################past_key_values:{past_key_values[0][0].shape}  before##########################################") 
            
        #     ##{len(past_key_values),type(past_key_values)} : (32, <class 'list'>)  before##
        #     print(f"###########################past_key_values:{len(past_key_values),type(past_key_values)}  before##########################################") 
            
            
        #     #######{len(past_key_values[0]),type(past_key_values[0])}:(2, <class 'tuple'>)  before######  // ######past_key_values[0]:(2, <class 'list'>)  before#######
        #     print(f"###########################past_key_values[0]:{len(past_key_values[0]),type(past_key_values[0])}  before##########################################") 
            
        #     ###{len(past_key_values[0][0]),type(past_key_values[0][0]),past_key_values[0][0].shape}  : (1, <class 'torch.Tensor'>, torch.Size([1, 8, 129, 128]))  before###
        #     print(f"###########################past_key_values[0]:{len(past_key_values[0][0]),type(past_key_values[0][0]),past_key_values[0][0].shape}  before##########################################") 
        
        ###########################################################
        start_stp = time.time()
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            end_stp0 = time.time()    
            
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            if kv_cache is not None:
                kv_cache.update_past_tok_ids(input_ids)
            # ##############################my##################################
            # print(f"###########################past_key_values[0][0]:{len(past_key_values[0][0]),type(past_key_values[0][0]),past_key_values[0][0].shape}  before##########################################") 
            # print(f"###############input_ids: {input_ids.shape, input_ids}, idx: {idx}#######################")
            
            # ##################################################################
            
            end_stp1 = time.time()    
            
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)
            if kv_cache is not None:
                # past_key_values = kv_cache(past_key_values)
                if args.enable_segmented_LLM:
                    past_key_values = kv_cache.evict_except_for_seg(past_key_values, SEP_ACCUMULATION=True, USE_MAX_SEP_CACHE=True)
                else:
                    past_key_values = kv_cache(past_key_values)
                    
            # print(f"########## num_eval_tokens:{num_eval_tokens}#############", flush=True)    
        ##############################my####################################
        # print(f"#################num_eval_tokens:{num_eval_tokens}#####################")
        if args.enable_segmented_LLM:
            if past_key_values is not None:
                # past_key_values[0][0].shape:torch.Size([1, 8, 129, 128])  after
                # print(f"###########################past_key_values[0][0].shape:{past_key_values[0][0].shape}  after##########################################")
                # print(f"###########################kv_cache.past_tok_ids.shape:{kv_cache.past_tok_ids.shape}  after##########################################")
                # # print(f"###########################kv_cache.past_tok_ids:{kv_cache.past_tok_ids}  after##########################################")
                # print(f"###########################kv_cache.sep_exrange:{kv_cache.sep_exrange}  after##########################################")
                pass
        ##################################################################
        end_stp2 = time.time()

        total_infer_time0 += (end_stp0 - start_stp)
        total_infer_time1 += (end_stp1 - start_stp)
        total_infer_time2 += (end_stp2 - start_stp)

        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        print(neg_log_likelihood.item(), file=f, flush=True)
        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

f.close()

ppl = torch.exp(torch.stack(nlls).mean())
print(f"\nOverall PPL: {ppl.item()}\n")
print(f"total_infer_time0:{total_infer_time0},total_infer_time1:{total_infer_time1}, total_infer_time2:{total_infer_time2}")

# print(f"Detailed time analysis:")
# print(f"In LlamaModel: llama_tot_time1: {model.model.llama_tot_time1}, llama_tot_time2: {model.model.llama_tot_time2}, llama_tot_time3: {model.model.llama_tot_time3} ")
# sum_layers_t1 = 0
# sum_layers_t2 = 0
# sum_layers_t3 = 0
# sum_layers_t4 = 0
# sum_layers_t8 = 0
# sum_layers_t9 = 0


# sum_layers_att_t1 = 0
# sum_layers_att_t2 = 0
# sum_layers_att_t3 = 0
# sum_layers_att_t4 = 0
# sum_layers_att_t5 = 0
# sum_layers_att_t6 = 0


# sum_layers_inLlamaModel_t8 = model.model.llama_tot_time8
# sum_layers_inLlamaModel_t9 = model.model.llama_tot_time9
# for i in range(len(model.model.layers)):
#     print(f"In LlamaDecoderLayer {i}: dec_tot_time1: {model.model.layers[i].dec_tot_time1},  dec_tot_time2: {model.model.layers[i].dec_tot_time2},  dec_tot_time3: {model.model.layers[i].dec_tot_time3}")
#     print(f"In LlamaAttention {i}: dec_tot_time1: {model.model.layers[i].self_attn.att_tot_time1},  dec_tot_time2: {model.model.layers[i].self_attn.att_tot_time2},  dec_tot_time3: {model.model.layers[i].self_attn.att_tot_time3},  dec_tot_time4: {model.model.layers[i].self_attn.att_tot_time4},  dec_tot_time5: {model.model.layers[i].self_attn.att_tot_time5},  dec_tot_time6: {model.model.layers[i].self_attn.att_tot_time6},  dec_tot_time7: {model.model.layers[i].self_attn.att_tot_time7}")
    
#     sum_layers_t1 += model.model.layers[i].dec_tot_time1
#     sum_layers_t2 += model.model.layers[i].dec_tot_time2
#     sum_layers_t3 += model.model.layers[i].dec_tot_time3
#     sum_layers_t4 += model.model.layers[i].dec_tot_time4


#     sum_layers_att_t1 += model.model.layers[i].self_attn.att_tot_time1
#     sum_layers_att_t2 += model.model.layers[i].self_attn.att_tot_time2
#     sum_layers_att_t3 += model.model.layers[i].self_attn.att_tot_time3
#     sum_layers_att_t4 += model.model.layers[i].self_attn.att_tot_time4
#     sum_layers_att_t5 += model.model.layers[i].self_attn.att_tot_time5
#     sum_layers_att_t6 += model.model.layers[i].self_attn.att_tot_time6

# print(f"##############################################################################################")
# print(f"Sum LlamaDecoderLayer : sum_layers_t1: {sum_layers_t1},  sum_layers_t2: {sum_layers_t2},  sum_layers_t3: {sum_layers_t3},  sum_layers_t4: {sum_layers_t4}, sum_layers_inLlamaModel_t8: {sum_layers_inLlamaModel_t8},  sum_layers_inLlamaModel_t9: {sum_layers_inLlamaModel_t9}")
# print(f"Sum LlamaAttention : sum_layers_att_t1: {sum_layers_att_t1},  sum_layers_att_t2: {sum_layers_att_t2},  sum_layers_att_t3: {sum_layers_att_t3},  sum_layers_att_t4: {sum_layers_att_t4},  sum_layers_att_t5: {sum_layers_att_t5},  sum_layers_att_t6: {sum_layers_att_t6}")


# self_attn
with open(f"{args.output_dir}/ppl.txt", "w") as f:
    f.write(f"{ppl.item()}\n")
