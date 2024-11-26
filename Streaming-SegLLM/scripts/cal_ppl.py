import torch

# ctx_len = int(1048576 * 4)
ctx_len = int(1024 * 20)

# log_path = '../outputs/pg19/seg_len10M_sink4_seg32_win224_ca324/'
# log_path = '../outputs/pg19/seg_len10M_sink4_seg32_win400_ca516/' # be killed
# log_path = '../outputs/pg19/seg_len10M_sink4_seg64_win224_ca324/' 

# log_path = '../outputs/pg19/streamingllm_len10M_sink4_ca324/' 
# log_path = '../outputs/pg19/streamingllm_len10M_sink4_ca516/' 



# log_path = '../outputs/wikitext/streamingllm_len20k_sink4_324/' 
# log_path = '../outputs/wikitext/streamingllm_len20k_sink4_516/' 
# log_path = '../outputs/wikitext/streamingllm_len20k_sink4_800/' 
# log_path = '../outputs/wikitext/streamingllm_len20k_sink4_ca324_noshift/' 
# log_path = '../outputs/wikitext/streamingllm_len20k_sink0_ca324_noshift/' 



# log_path = '../outputs/wikitext/seg_len20k_sink4_seg64_win224_ca324/' 
# log_path = '../outputs/wikitext/seg_len20k_sink4_seg64_win320_ca516/' 
# log_path = '../outputs/wikitext/seg_len20k_sink4_seg64_win512_ca800/' 
# log_path = '../outputs/wikitext/seg_len20k_sink4_seg48_win224_ca324/' 

log_path = '../outputs/wikitext/streamingllm_len20k_sink0_ca324_nosink/' 






# streamingllm_len20k_sink0_ca324_noshift


ppl_log_file = 'log.txt'
out_file_name = f"ppl_len{ctx_len}.txt"

nlls = []
with open(log_path+ppl_log_file, 'r') as f:
    for i in range(ctx_len):
        cur_ppl_str = f.readline()
        if  len(cur_ppl_str) <=0:
            print(f"err: for i: {i}, cur_ppl_str: {cur_ppl_str}")
            exit(0)
            continue
        
        neg_log_likelihood =  torch.tensor(float(cur_ppl_str))
        
        nlls.append(neg_log_likelihood)


ppl = torch.exp(torch.stack(nlls).mean())
print(f"ppl: {ppl.item()}")

with open(log_path + out_file_name, 'w') as fout:
    fout.write(f"{ppl.item()}\n")