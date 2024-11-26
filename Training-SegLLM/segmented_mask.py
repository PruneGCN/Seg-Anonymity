import torch
#################################################my########################################################
class TrainingMask:
    def __init__(self, neox_args=None):
        self.past_considered_seps_idx = [-1]  # Store the list of the considered seps after attention sink and before non-local end. Must have "-1" as the initial placeholder. Only take effects when running random experiments
        self.past_kept_tok_idx = []  # Store the list of the random substitute kept tokens after attention sink and before non-local end. Initially empty. Only take effects when running random experiments
        self.past_ids = [] # Store the past ids. Clear it when a new question coming.
        self.kept_tokens_count_layer = [] # Store the kept tokens and total tokens of different layers.
        self.kept_tokens_count_seq =  (0,0) # Store the kept tokens and  total tokens of the final generated seq. It will be cleared every new seq
        self.kept_tokens_count_total = (0,0) # Store the kept tokens and total tokens of the total dataset. 
        
        self.kept_attmap_count_layer = [] # Store the kept tokens and total tokens of different layers.
        self.kept_attmap_count_seq =  (0,0) # Store the kept tokens and  total tokens of the final generated seq. It will be cleared every new seq
        self.kept_attmap_count_total = (0,0) # Store the kept tokens and total tokens of the total dataset. 

        self.batch_prefill_max_seq_len = -1 # Store the length of the longest seq in a batch when prefilling.
        self.FLOAT_ATT_MASK_DTYPE = torch.float32 # The availble float type for 910B3 NPU
        self.dtype = torch.float16  # for compatibility
        self.NPU_MIN = -1e37  # The min value for 910B3
        self.print_KV_intervals = 4000
        self.print_KV_count = 0

        assert self.NPU_MIN > torch.finfo(self.FLOAT_ATT_MASK_DTYPE).min

        if not (neox_args is None):
            self.Layer_num = neox_args.num_layers 
        else:
            self.Layer_num = 12 
        
                ## special_tokens = ['.', ',', '?', '!', ';', ":", '\n'] 
                
        # self.special_tokens_id = [13, 11, 30 ,0, 26, 25, 185, 25, 207] # deepseek v1
        # self.special_tokens_id = [13, 11, 30 ,0, 26, 25, 185, 25] # deepseek v2, remove 207
        # self.special_tokens_id = [13, 11, 30 ,0, 26, 25, 185, 25, 48, 32] # deepseek v3, remove 207, add 48 'Q', 32 'A'


        ##  Llama        
        # self.special_tokens_id = [13, 11, 30 ,0, 26, 25, 198, 220, 128000] # llama3 v1
        # self.special_tokens_id = [13, 30 ,0, 26, 198, 220, 128000] # llama3 v2,                   remove 11 ',',   25 ':' 
        # self.special_tokens_id = [13, 30 ,0, 26, 198, 220, 128000, 5380, 382] # llama3 v3,      remove 11 ',',   25 ':' . add 5380 '?\n', 382 '.\n'
        # self.special_tokens_id = [13, 11, 30 ,0, 26, 25, 198, 220, 128000, 5380, 382 ]  ## v4 all + 5380 '?\n', 382 '.\n'


        if neox_args is None:

            ## Pythia: GPTNeoX
            ## special_tokens = ['.', ',', '?', '!', ';', ":", ' ', '\t','\n'] 
            self.special_tokens_id = [15, 13, 32, 2, 28, 27, 209, 186, 187]


            # self.PADDING_ID = 100001  # deepseek math instruct
            # self.PADDING_ID = 128009  # llama3 8b instuct
            self.PADDING_ID = 0  # pythia
            self.prefill_k = 0
            self.decode_k = 0
            # self.k = min(self.prefill_k, self.decode_k)

            self.prefill_window_size = 10  # Only take effect when self.USE_DYNAMIC_PREFILL_WINDOW_SIZE=False
            self.decode_window_size = 10 # Only take effect when self.USE_DYNAMIC_DECODE_WINDOW_SIZE=False        

            self.USE_DYNAMIC_PREFILL_WINDOW_SIZE = False # If True: the prefilling window size for different decoder layer is different.
            self.USE_DYNAMIC_DECODE_WINDOW_SIZE = False # If True: the decoding window size for different decoder layer is different.
            
            self.prefill_win_size_list = [512,  512,  512,  512,  512,  400,  400,  400,  400, 400,
                                          350,  350
                                                    
            ] # V1

            # self.prefill_win_size_list = [20,   19,   18,   17,   16,   15,  14,   13,  12, 11,
            #                             10,  9
                                                    
            # ] # test


            # self.prefill_win_size_list = list( range(800, 200, -20) )    # V5




            # self.decode_win_size_list =  [64,  32,  16,  8,  4,  2,  1,  1,  1, 1,
            #                               1,  1,  1,  1,  1,  1,  1,  1,  1, 1,
            #                               1,  1,  1,  1,  1,  1,  1,  1,  1, 1
                                                    
            # ]


            self.decode_win_size_list =  self.prefill_win_size_list
            # self.prefill_win_size_list =  self.decode_win_size_list

            self.att_sink_max_idx = 2 # The largest index for attention sink tokens

            self.original_flag =False  # Run the pretrained model without any modification if True
            self.streamingLLM = False  # Run streamingLLM. Only takes effect when self.original_flag=False
            self.random_special_tokens_uniform = False # Keep random selected tokens to replace seps. Select one other token between two seps
            self.random_special_tokens_general = False # Keep random selected tokens to replace seps. Randomly choose tokens generally
            

            self.BATCH_DYNAMIC_ATT_SINK = False # If True: use the floating attension sink positions. Can be False when pretraining since the attention sinks are at the beginning of seqs in batch for pretraining
            self.PRINT_KV_RATIO = False # If True, print the KV cache preservation ratio. Can be False when pretraining
            self.print_KV_intervals = 4000  # Print the retention ratio for the computational complexity of calculating the attention map once after every 'print_KV_intervals' forward passes. It only takes effect when PRINT_KV_RATIO=True.    


            self.USE_BiPE = False   ## False by default. If True (must also set pos_emb='rotary' or 'alibi'), use Bilevel Positional Encoding.
            self.BiPE_seps = self.special_tokens_id ## The token ids of the seperator tokens for BiPE
            self.USE_SA_SOFTMAX = False # If True, use Self-Adjusting Softmax Attention.

            self.NOT_AVOID_SEP = False # False by default. Only take effects when self.random_special_tokens_general = True. When self.NOT_AVOID_SEP=True, it is possible to randomly select another sep to replace a sep.        
            self.EXCLUDE_DIAGONAL = True # True by default. Only take effects when self.floating_window = False. When True, it means when we choose fixed window to process the prefilling mask, the diagonal elements in the prefilling mask could be set negative (like yihang's code). When False: would keep the prefilling mask's diagonal positive

            self.floating_window = True # True by default, which means using floating window when prefilling. False means using fixed window (on the rightmost side for all rows of the prefilling mask matrix) when prefilling, which is better sometimes however :)

        else:
            ## Pythia: GPTNeoX
            ## special_tokens = ['.', ',', '?', '!', ';', ":", ' ', '\t','\n'] 
            self.special_tokens_id = neox_args.special_tokens_id

            self.PADDING_ID = neox_args.PADDING_ID
            self.prefill_k = neox_args.prefill_k
            self.decode_k = neox_args.decode_k

            self.prefill_window_size = neox_args.prefill_window_size  # Only take effect when self.USE_DYNAMIC_PREFILL_WINDOW_SIZE=False
            self.decode_window_size = neox_args.decode_window_size # Only take effect when self.USE_DYNAMIC_DECODE_WINDOW_SIZE=False        

            self.USE_DYNAMIC_PREFILL_WINDOW_SIZE = neox_args.USE_DYNAMIC_PREFILL_WINDOW_SIZE # If True: the prefilling window size for different decoder layer is different.
            self.USE_DYNAMIC_DECODE_WINDOW_SIZE = neox_args.USE_DYNAMIC_DECODE_WINDOW_SIZE # If True: the decoding window size for different decoder layer is different.
            

            self.prefill_win_size_list = neox_args.prefill_win_size_list


            self.decode_win_size_list =  neox_args.decode_win_size_list
        

            self.att_sink_max_idx = neox_args.att_sink_max_idx # The largest index for attention sink tokens

            self.original_flag =neox_args.original_flag  # Run the pretrained model without any modification if True
            self.streamingLLM = neox_args.streamingLLM  # Run streamingLLM. Only takes effect when self.original_flag=False
            self.random_special_tokens_uniform = neox_args.random_special_tokens_uniform # Keep random selected tokens to replace seps. Select one other token between two seps
            self.random_special_tokens_general = neox_args.random_special_tokens_general # Keep random selected tokens to replace seps. Randomly choose tokens generally
            

            self.BATCH_DYNAMIC_ATT_SINK = neox_args.BATCH_DYNAMIC_ATT_SINK # If True: use the floating attension sink positions. Can be False when pretraining since the attention sinks are at the beginning of seqs in batch for pretraining
            self.PRINT_KV_RATIO = neox_args.PRINT_KV_RATIO # If True, print the KV cache preservation ratio. Can be False when pretraining
            self.print_KV_intervals = neox_args.print_KV_intervals  # Print the retention ratio for the computational complexity of calculating the attention map once after every 'print_KV_intervals' forward passes. It only takes effect when PRINT_KV_RATIO=True.    

            self.USE_BiPE = neox_args.USE_BiPE   ## False by default. If True (must also set pos_emb='rotary' or 'alibi'), use Bilevel Positional Encoding.
            self.BiPE_seps = neox_args.BiPE_seps  ## The token ids of the seperator tokens for BiPE

            self.USE_SA_SOFTMAX =neox_args.USE_SA_SOFTMAX # If True, use Self-Adjusting Softmax Attention.


            self.NOT_AVOID_SEP = neox_args.NOT_AVOID_SEP # False by default. Only take effects when self.random_special_tokens_general = True. When self.NOT_AVOID_SEP=True, it is possible to randomly select another sep to replace a sep.        
            self.EXCLUDE_DIAGONAL = neox_args.EXCLUDE_DIAGONAL # True by default. Only take effects when self.floating_window = False. When True, it means when we choose fixed window to process the prefilling mask, the diagonal elements in the prefilling mask could be set negative (like yihang's code). When False: would keep the prefilling mask's diagonal positive

            self.floating_window = neox_args.floating_window # True by default, which means using floating window when prefilling. False means using fixed window (on the rightmost side for all rows of the prefilling mask matrix) when prefilling, which is better sometimes however :)





        EXPERIMENT_NUM = int(self.original_flag)+int(self.streamingLLM)+int(self.random_special_tokens_uniform)+int(self.random_special_tokens_general)
        UNIQUE_EXP_FLAG = EXPERIMENT_NUM <= 1
        assert (self.streamingLLM and (self.decode_k < 1) and (self.prefill_k < 1)  ) or (not self.streamingLLM), "decode_k and prefill_k must be less than 1 when running streamingLLM"                
        assert UNIQUE_EXP_FLAG, "We can only run one experiment at one time"

        if self.USE_DYNAMIC_PREFILL_WINDOW_SIZE:
            assert self.Layer_num == len(self.prefill_win_size_list)
        if self.USE_DYNAMIC_DECODE_WINDOW_SIZE:
            assert  self.Layer_num == len(self.decode_win_size_list)



        print("###########################k and window_size, etc.############################################")
        print(f"prefill_k: {self.prefill_k}")
        print(f"decode_k: {self.decode_k}")
        if self.USE_DYNAMIC_PREFILL_WINDOW_SIZE:
            print(f"self.prefill_win_size_list: {self.prefill_win_size_list}")
            self.prefill_window_size = None
        else:        
            print(f"self.prefill_window_size: {self.prefill_window_size}")

        if self.USE_DYNAMIC_DECODE_WINDOW_SIZE:            
            print(f"self.decode_win_size_list: {self.decode_win_size_list}")
            self.decode_window_size = None
        else:
            print(f"self.decode_window_size: {self.decode_window_size}")

        print(f"self.Layer_num: {self.Layer_num}")
        print(f"self.att_sink_max_idx: {self.att_sink_max_idx}")
        print(f"self.original_flag:  {self.original_flag}" )
        print(f"self.streamingLLM:  {self.streamingLLM}" )
        print(f"self.random_special_tokens_uniform:  {self.random_special_tokens_uniform}" )
        print(f"self.random_special_tokens_general:  {self.random_special_tokens_general}" )


        print(f"self.NOT_AVOID_SEP:  {self.NOT_AVOID_SEP}" )
        print(f"self.EXCLUDE_DIAGONAL:  {self.EXCLUDE_DIAGONAL}" )

        print(">>> Please be careful of the special_tokens_id, Make sure they are correct for the current LLM")
        print(f"self.special_tokens_id: {self.special_tokens_id}")
        print(f"self.floating_window: {self.floating_window}")

        print(f"self.USE_BiPE: {self.USE_BiPE}")
        if self.USE_BiPE:
            print(f"self.BiPE_seps: {self.BiPE_seps}")
        print(f"self.USE_SA_SOFTMAX: {self.USE_SA_SOFTMAX}")


        if not self.EXCLUDE_DIAGONAL:
            if self.floating_window:
                print(f"Warnings: self.EXCLUDE_DIAGONAL={self.EXCLUDE_DIAGONAL} only take effects when self.floating_window=False. self.EXCLUDE_DIAGONAL is True by default ")
            else:
                print(f"Note: self.EXCLUDE_DIAGONAL={self.EXCLUDE_DIAGONAL} only take effects when self.floating_window=False ")


        if self.NOT_AVOID_SEP:
            print(f">>>>>>Warning: When sample substitute tokens to replace seps, it is possible to use another sep to replace a sep since self.NOT_AVOID_SEP={self.NOT_AVOID_SEP}<<<<<")
            if not self.random_special_tokens_general:
                print(f"Warnings: self.NOT_AVOID_SEP={self.NOT_AVOID_SEP} only take effects when self.random_special_tokens_general=True ")
            else:
                print(f"Note: self.NOT_AVOID_SEP={self.NOT_AVOID_SEP} only take effects when self.random_special_tokens_general=True ")

        if self.decode_k < 1:
            if self.streamingLLM:
                print(f"This is about streamingLLM since self.decode_k ={self.decode_k}, self.prefill_k = {self.prefill_k} and self.streamingLLM: {self.streamingLLM} ")
            else:
                print(f"This is about NOT streamingLLM since self.decode_k ={self.decode_k}, self.prefill_k = {self.prefill_k} and self.streamingLLM: {self.streamingLLM}")

        if int(self.original_flag)+int(self.streamingLLM)+int(self.random_special_tokens_uniform)+int(self.random_special_tokens_general) <= 0:
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>------------------ Running our version of the mask strategy-------------------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
        
        elif self.original_flag:
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>------------------ Running the original baseline (no changing) ---------------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
        elif self.streamingLLM:
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>------------------ Running streamingLLM (decode_k,prefill_k = 0)--------------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
        elif self.random_special_tokens_uniform:
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>------------------ Running random test (keep a token near a sep) -------------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")

        elif self.random_special_tokens_general:            
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>-------------Running \"Yihang's + Diagonal\" prefilling random version--------<<<<<<<<")                       
            if self.NOT_AVOID_SEP:                
                print(">>>>>>>>------------------           ( NOT Avoiding Seps )               -------------<<<<<<<<")                    
            else:
                print(">>>>>>>>------------------             ( Avoiding Seps )                 -------------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
    
        elif self.random_special_tokens_general:            
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>------------------ Running random test (generally sample tokens) -------------<<<<<<<<")                        
            if self.NOT_AVOID_SEP:                
                print(">>>>>>>>------------------           ( NOT Avoiding Seps )               -------------<<<<<<<<")                    
            else:
                print(">>>>>>>>------------------             ( Avoiding Seps )                 -------------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
        else:
            print(">>>>>>>>>>>>>>>>>>>>>ERROR, ERROR, ERROR<<<<<<<<<<<<<<<<<<<<<<<<<")
            pass


    def count_decode_kept_element(self, mask, layer_id):
        total_toks = mask.numel()

        if  mask.dtype == torch.bool:
            kept_toks = (~(~mask)).int().sum().item()
        else:
            kept_toks = (mask>-1).int().sum().item()

        self.kept_tokens_count_layer.append( (kept_toks, total_toks) )

        if layer_id == (self.Layer_num - 1):
            self.kept_tokens_count_seq = tuple([sum(x) for x in zip(*self.kept_tokens_count_layer)])  # skip the 1st since it is for prefilling. The last element for this list is for the final seq
            self.kept_tokens_count_layer = []

    def count_prefill_kept_element(self, mask, layer_id):
        total_toks = mask.numel()

        if  mask.dtype == torch.bool:
            kept_toks = (~(~mask)).int().sum().item()
        else:
            kept_toks = (mask>-1).int().sum().item()

        self.kept_tokens_count_layer.append( (kept_toks, total_toks) )

        if layer_id == (self.Layer_num - 1):
            self.kept_tokens_count_seq = tuple([sum(x) for x in zip(*self.kept_tokens_count_layer)])  # skip the 1st since it is for prefilling. The last element for this list is for the final seq
            self.kept_tokens_count_layer = []


    def count_prefill_kept_attmap(self, mask, layer_id):
        # total_toks = mask.numel()
        total_entries = float(  mask.shape[0] * ( (1 + mask.shape[-1]) * mask.shape[-2] / 2 )   ) ## B * (1+ seq_len) * seq /2
        # print("########################total_toks################################")
        # print(mask.shape)
        # print(mask.shape[0], mask.shape[-1],mask.shape[-2] )
        # print(total_toks)

        if  mask.dtype == torch.bool:
            kept_entries = (~(~mask)).int().sum().item()
        else:
            kept_entries = (mask>-1).int().sum().item()

        self.kept_attmap_count_layer.append( (kept_entries, total_entries) )

        if layer_id == (self.Layer_num - 1):
            self.kept_attmap_count_seq = tuple([sum(x) for x in zip(*self.kept_attmap_count_layer)])  # skip the 1st since it is for prefilling. The last element for this list is for the final seq
            self.kept_attmap_count_layer = []
   
    def count_prefill_kept_kv_all_layers(self, attention_mask):
        for layer_id in range(self.Layer_num):
            if isinstance(attention_mask, (list, tuple)) or len(attention_mask.shape) > 4:           
                mask = attention_mask[layer_id]
            else:
                mask = attention_mask
            
            mask_last_row = mask[:, :, -1, :]
            # print(f"####################for layer {layer_id}, {mask_last_row.shape}###########################")

            self.count_prefill_kept_element(mask_last_row,  layer_id)
    
    def count_prefill_kept_attmap_all_layers(self, attention_mask):

        for layer_id in range(self.Layer_num):
            if isinstance(attention_mask, (list, tuple)) or len(attention_mask.shape) > 4:           
                mask = attention_mask[layer_id]
            else:
                mask = attention_mask
            self.count_prefill_kept_attmap(mask,  layer_id)
    


    def build_prefill_mask(self, past_ids, causal_mask2, special_tokens_id, att_sink_size, window_sizeS, DYNAMIC_ATT_SINK=False, att_sink_idx_tensor=None, PAD_TOK_ID=-100):
        # import copy

        if causal_mask2.dtype == torch.bool:
            ori_causal_mask2 = causal_mask2.clone().detach()
        else:
            ori_causal_mask2 = (causal_mask2.clone().detach() > -1)

        # causal_mask2 = torch.zeros_like(ori_causal_mask2).bool().detach()  # B x 1 x seq_len x seq_len
        if isinstance(past_ids, list):
            past_ids_tensor = torch.tensor(past_ids).int().cuda()  # B x seq_len
        else:
            past_ids_tensor = past_ids.clone().detach().int().cuda()  # B x seq_len
            # past_ids_tensor = past_ids.int()  # B x seq_len ## pythia OOM. Not really useful

        ## For some code, causal_mask2'shape is B x 1 x seq_len x (seq_len+1),  last col is a pad
        if past_ids_tensor.shape[-1] != ori_causal_mask2.shape[-1]:  # Have some bugs for MMLU. I fixed it here but have not been tested.
            # print("##############paddding here##################")
            sep_index_tensor = torch.zeros( ori_causal_mask2.shape[0], ori_causal_mask2.shape[-1]  ).bool().to(past_ids_tensor.device)  # B x seq_len or B x (seq_len + 1)
            # pad_tensor = (torch.ones(ori_causal_mask2.shape[0], ori_causal_mask2.shape[-1]) * PAD_TOK_ID).int().to(past_ids_tensor.device)     ## ori      
            assert not (PAD_TOK_ID in special_tokens_id),   f"PAD_TOK_ID: {PAD_TOK_ID} should not be in the special_tokens_id: {special_tokens_id}"
            assert ori_causal_mask2.shape[-1] > past_ids_tensor.shape[-1] ## For some code, causal_mask2'shape is B x 1 x seq_len x (seq_len+1  (or + n)),  last col is a pad

            pad_tensor = (torch.ones(ori_causal_mask2.shape[0], ori_causal_mask2.shape[-1] - past_ids_tensor.shape[-1] ).int() * PAD_TOK_ID).int().to(past_ids_tensor.device)     ## new version. Shape: B x 1 (or x n)
            past_ids_tensor = torch.cat([past_ids_tensor, pad_tensor], dim=-1 )  ## past_ids_tensor shoud have the same shape as sep_index_tensor. And PAD_TOK_ID should not be in the special_tokens_id
        
        else:
            sep_index_tensor = torch.zeros_like(past_ids_tensor).bool().to(past_ids_tensor.device)  # B x seq_len


        for sp_id in special_tokens_id:
            # sep_index_tensor = sep_index_tensor + ( past_ids_tensor == sp_id ) # B x seq_len
            sep_index_tensor = sep_index_tensor | ( past_ids_tensor == sp_id )# B x seq_len or B x (seq_len + 1)

        ## Set the attentions for seps to positive
        # causal_mask2.permute(1,0,3,2)[:,sep_index_tensor, : ]  = True #  1 x B x seq_len (col) x seq_len (row)

        # cau_per =  causal_mask2.permute(1,0,3,2)
        # cau_per[:,sep_index_tensor, : ]  = True #  1 x B x seq_len (col) x seq_len (row)

        #  logically correct
        # sep_index_tensor_exp = sep_index_tensor[:, None, None, :].expand(-1,-1,causal_mask2.shape[-1],-1).clone().detach()
        # causal_mask2[sep_index_tensor_exp] = True
        del causal_mask2 # pythia
        causal_mask2 = sep_index_tensor[:, None, None, :].expand(-1,-1,ori_causal_mask2.shape[-2],-1).clone().detach() # B x 1 x seq_len x seq_len  OR  B x 1 x seq_len x (seq_len+1)
        del sep_index_tensor # pythia


        # Attention sink
        if DYNAMIC_ATT_SINK  and ( not (att_sink_idx_tensor is None) ):
            # causal_mask2[att_sink_idx_tensor] = True
            assert causal_mask2.shape == att_sink_idx_tensor.shape
            assert att_sink_idx_tensor.dtype == torch.bool
            assert causal_mask2.dtype == torch.bool
            causal_mask2 = att_sink_idx_tensor | causal_mask2

        else:
            causal_mask2[:, :, : , :att_sink_size] = True

        ## lower triangular mask. 
        # lower_tri_mask = torch.tril(torch.ones_like(causal_mask2),diagonal=0).bool() * ori_causal_mask2
        assert ori_causal_mask2.dtype == torch.bool
        lower_tri_mask =  ori_causal_mask2

        res = [] # results

        ## Local window
        if self.USE_DYNAMIC_PREFILL_WINDOW_SIZE:
            assert isinstance(window_sizeS, list), "window_sizeS must be a list when self.USE_DYNAMIC_PREFILL_WINDOW_SIZE: is True"
            for ly in range(self.Layer_num):
                if (ly > 0) and ( window_sizeS[ly] == window_sizeS[ly-1]  ):
                    # res.append( copy.deepcopy( res[-1] ) )
                    res.append( res[-1] )
                    continue

                w_size = - (window_sizeS[ly] - int(self.EXCLUDE_DIAGONAL))

                win_mask = torch.triu(torch.ones_like(causal_mask2),diagonal=w_size).bool() 
                
                # res.append(copy.deepcopy( (causal_mask2 | win_mask) & lower_tri_mask  ))
                res.append( ( (causal_mask2 | win_mask) & lower_tri_mask  ).clone().detach())

        else:
            window_size = window_sizeS
            w_size = - (window_size - int(self.EXCLUDE_DIAGONAL))

            win_mask = torch.triu(torch.ones_like(causal_mask2),diagonal=w_size).bool() 
                            
            # res = copy.deepcopy( (causal_mask2 | win_mask) & lower_tri_mask  )
            res = (causal_mask2 | win_mask) & lower_tri_mask   ## pythia
            # del lower_tri_mask ##pythia
            # del causal_mask2  ##pythia
            # del win_mask ##pythia
        # ###########################################################################    
        # torch.set_printoptions(profile="full")
        # print("#####################prefill_mask res###########################")
        # print(res.shape)
        # for row in range(100):
        #     print(f"###########################prefill_mask row {row}#####################################")
        #     print(res[:, :, row, :])
        # torch.set_printoptions(profile="default") # reset
        # ###########################################################################
        return res

    def build_decode_mask(self, past_ids, causal_mask2, special_tokens_id, att_sink_size, window_sizeS, DYNAMIC_ATT_SINK=False, att_sink_idx_tensor=None ):
        # import copy
        # print("##########################in build_decode_mask causal_mask2###################################", causal_mask2.shape)
        # print(causal_mask2)

        causal_mask2 = torch.zeros_like(causal_mask2).bool()  # B x 1 x 1 x seq_len
        
        if isinstance(past_ids, list):
            past_ids_tensor = torch.tensor(past_ids).int()  # B x seq_len
        else:
            past_ids_tensor = past_ids.clone().detach().int()  # B x seq_len

        sep_index_tensor = torch.zeros_like(past_ids_tensor).bool()  # B x seq_len

        for sp_id in special_tokens_id:
            # sep_index_tensor = sep_index_tensor + ( past_ids_tensor == sp_id ) # B x seq_len
            sep_index_tensor = sep_index_tensor | ( past_ids_tensor == sp_id ) # B x seq_len

        ## Set the attentions for seps to positive
        # causal_mask2.permute(1,2,0,3)[:, : ,sep_index_tensor  ]  = True # 1 x 1 x B x seq_len (col)

        sep_index_tensor_exp = sep_index_tensor[:, None, None, :] #  B x  1 x 1 x seq_len (col)
        causal_mask2[sep_index_tensor_exp] = True

        ## Attention sink
        if DYNAMIC_ATT_SINK  and ( not (att_sink_idx_tensor is None) ):
            assert causal_mask2.shape == att_sink_idx_tensor.shape
            assert att_sink_idx_tensor.dtype == torch.bool
            assert causal_mask2.dtype == torch.bool
            causal_mask2[att_sink_idx_tensor] = True
        else:
            causal_mask2[:, :, : , :att_sink_size] = True

        res = [] # results

        ## Local window
        if self.USE_DYNAMIC_DECODE_WINDOW_SIZE:
            assert isinstance(window_sizeS, list), "window_sizeS must be a list when self.USE_DYNAMIC_PREFILL_WINDOW_SIZE: is True"
            for ly in range(self.Layer_num):
                if (ly > 0) and ( window_sizeS[ly] == window_sizeS[ly-1]  ):
                    # res.append( copy.deepcopy( res[-1] ) )
                    res.append( res[-1]  )
                    continue

                w_size = - window_sizeS[ly] 
                # win_mask = torch.triu(torch.ones_like(causal_mask2),diagonal=w_size).bool() 
                win_mask = torch.zeros_like(causal_mask2).bool() ##  B x  1 x 1 x seq_len (col)
                win_mask[:,:,:,  w_size:] = True

                # res.append(copy.deepcopy( causal_mask2 + win_mask   ))
                res.append(( causal_mask2 | win_mask   ).clone().detach() )

        else:
            window_size = window_sizeS
            w_size = - window_size
            win_mask = torch.zeros_like(causal_mask2).bool() ##  B x  1 x 1 x seq_len (col)
            win_mask[:,:,:,  w_size:] = True                
            # res = copy.deepcopy( causal_mask2 + win_mask  )
            res =  causal_mask2 | win_mask 

        # ###########################################################################    
        # torch.set_printoptions(profile="full")
        # print("#####################decode_mask res###########################")
        # print(res.shape)
        # print(res[:, :, [0], :])
        # torch.set_printoptions(profile="default") # reset
        # ###########################################################################            

        return res

    def build_eval_att_sink_index(self,  input_ids, causal_mask2, pre_max_len ,att_sink_size ,pad_id, prefill_sink_pos_tensor=None ):
        att_sink_idx_tensor = torch.zeros_like(causal_mask2).bool()# B x  1 x seq_len (1) x seq_len
        if  not (prefill_sink_pos_tensor is None):
            att_sink_positions = prefill_sink_pos_tensor.clone().detach()
            recyc_prefill_att_sink_position_tensor = None
            assert att_sink_positions.shape[0] == att_sink_idx_tensor.shape[0], 'prefill_sink_pos_tensor\' s shape is wrong! Its 1st dim must be batch_size'
            assert att_sink_positions.shape[1] == att_sink_size, 'prefill_sink_pos_tensor\' s shape is wrong! Its 2nd dim must be att_sink_size'
        else:
            if isinstance(input_ids, list):
                input_ids_tensor = torch.tensor(input_ids).int()  # B x seq_len
            else:
                input_ids_tensor = input_ids.clone().detach().int()  # B x seq_len
            
            padding_num =  (input_ids_tensor[:, :pre_max_len] == pad_id).int().sum(-1) # shape: B   ; Since when decoding (generation), system will put pad at the end of the short seqs in the batch but we only need to count the pads at the left side
            att_sink_positions = padding_num[:, None].expand(-1, att_sink_size).clone().detach() # B x att_sink_size            
            
            for i in range(att_sink_size):
                att_sink_positions[:, i] += i  # B x att_sink_size

            recyc_prefill_att_sink_position_tensor = att_sink_positions.clone().detach()
                        # B x  1  x  1 x att_sink_size
        
        att_sink_positions = att_sink_positions[:, None, None, :].expand(-1, att_sink_idx_tensor.shape[1], att_sink_idx_tensor.shape[2], -1  ).clone().detach() # B x  1 x seq_len (1) x att_sink_size
        src_ones =  torch.ones_like(att_sink_positions).bool()

        att_sink_idx_tensor.scatter_(dim=-1, index=att_sink_positions, src=src_ones)

        # ###############################################################################
        # print("###################att_sink_idx_tensor#########################")        
        # torch.set_printoptions(profile="full")
        # print(att_sink_idx_tensor.shape)
        # print(att_sink_idx_tensor[:, :, [0], :])
        # torch.set_printoptions(profile="default") # reset
        # ###############################################################################

        return att_sink_idx_tensor, recyc_prefill_att_sink_position_tensor

        
    def build_segmented_attention_mask(self, prefill_flag, past_ids, causal_mask2,  BATCH_DYNAMIC_ATT_SINK=False, att_sink_idx_tensor = None ):

            if prefill_flag:                
                if self.USE_DYNAMIC_PREFILL_WINDOW_SIZE:
                    window_sizeS = self.prefill_win_size_list                
                    prefill_causal_mask2_list = self.build_prefill_mask(past_ids, causal_mask2, self.special_tokens_id, self.att_sink_max_idx+1, window_sizeS, DYNAMIC_ATT_SINK=BATCH_DYNAMIC_ATT_SINK, att_sink_idx_tensor=att_sink_idx_tensor)
                    return prefill_causal_mask2_list
                else:
                    window_sizeS = self.prefill_window_size                
                    causal_mask2 = self.build_prefill_mask(past_ids, causal_mask2, self.special_tokens_id, self.att_sink_max_idx+1, window_sizeS , DYNAMIC_ATT_SINK=BATCH_DYNAMIC_ATT_SINK,  att_sink_idx_tensor=att_sink_idx_tensor)
                    return causal_mask2              
            else:                
                if self.USE_DYNAMIC_DECODE_WINDOW_SIZE:
                    window_sizeS = self.decode_win_size_list                
                    decode_causal_mask2_list = self.build_decode_mask(past_ids, causal_mask2, self.special_tokens_id, self.att_sink_max_idx+1, window_sizeS, DYNAMIC_ATT_SINK=BATCH_DYNAMIC_ATT_SINK,  att_sink_idx_tensor=att_sink_idx_tensor )
                    return decode_causal_mask2_list
                else:
                    window_sizeS = self.decode_window_size                
                    causal_mask2 = self.build_decode_mask(past_ids, causal_mask2, self.special_tokens_id, self.att_sink_max_idx+1, window_sizeS,  DYNAMIC_ATT_SINK=BATCH_DYNAMIC_ATT_SINK,  att_sink_idx_tensor=att_sink_idx_tensor)   
                    return causal_mask2


    def O1mask_2_infinite_mask(self, mask,  min_value, ASCEND_910B=True):        
        if mask.dtype != torch.bool:
            mask = mask.bool()

        if ('npu' in str(mask.device)) or ASCEND_910B:
            new_mask =  (~mask).float().to(dtype=self.FLOAT_ATT_MASK_DTYPE) * min_value
        else:
            new_mask = torch.zeros_like(mask).float().to(dtype=self.FLOAT_ATT_MASK_DTYPE)
            new_mask[ ~mask] = min_value
        # ###############################################################################
        # print("######################new_mask################################")
        # print(new_mask.shape)
        # print(new_mask)
        # ###############################################################################
        return new_mask
    
    def reverse_bool_mask(self, mask, return_tensor=True):        
        if isinstance(mask, list):
            res_mask_list = []
            for i, msk in enumerate(mask):
                assert msk.dtype == torch.bool
                # mask[i] = (~msk)
                res_mask_list.append(~msk)                
            if return_tensor:
                res_mask = torch.stack(res_mask_list, dim=0)
            else:
                res_mask = res_mask_list

            del mask[:]
            del mask

            return res_mask
        else:
            assert mask.dtype == torch.bool
            return ( ~mask)

    def get_bilevel_ids(self, ids, train_scale=None):
        # sep = torch.where(ids == 869, 1, 0)
        # sep[ids == 13] = 1
        assert len(self.BiPE_seps)>=1, f"self.BiPE_seps:{self.BiPE_seps}. You should set self.BiPE_seps."
        sep = torch.where(ids == self.BiPE_seps[0], 1, 0)
        for i in range(len(self.BiPE_seps)):
            if i == 0:
                continue
            sep[ids == self.BiPE_seps[i]] = 1

        pos1 = torch.cumsum(sep, dim=1)  # inter-pos
        pos1 = torch.cat([torch.zeros((ids.shape[0], 1), device=ids.device), pos1[:,:-1]], dim=1)
        pos2 = torch.cat([sep, torch.ones((ids.shape[0], 1), device=ids.device)], dim=1).reshape(-1)
        ones = torch.cat([torch.zeros((1), device=ids.device) - 1, torch.argwhere(pos2 == 1)[:, 0]])
        diff = (ones[1:] - ones[:-1])
        pos2[pos2 == 1] = diff
        pos2 = -torch.cumsum(torch.cat([torch.zeros((1), device=ids.device), pos2[:-1]]), dim=0)
        pos2 = pos2 + torch.arange(pos2.shape[-1], device=ids.device)
        pos2 = pos2.reshape(ids.shape[0], -1)[:, :-1]   # intra-pos
        if train_scale is not None:
            # pos1[pos1 >= train_scale] = train_scale - 1
            pos2[pos2 >= train_scale] = train_scale - 1

        return pos2.long(), pos1.long()  ##  intra-pos,  inter-pos



###########################################################################################################
