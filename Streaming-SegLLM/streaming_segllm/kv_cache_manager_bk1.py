import torch

def slice_on_1d(x, start, end):
    return x[:, start:end, ...]

def slice_on_2d(x, start, end):
    return x[:, :, start:end, ...]

def slice_on_3d(x, start, end):
    return x[:, :, :, start:end, ...]


# (k, b_id, sep_index_tensor[b_id], min_sep_num)
def sep_1bat_select_on_1d(x, Bid , sep_index, min_sep_num=None):
    print(f"Debug: #####################x.shape:{x.shape} in sep_1bat_select_on_1d###########################")
    if min_sep_num is None:
        return x[Bid, sep_index, ...].detach().clone()  # # Batch x seqlen x Head x dim --> sep_num x Head x dim    
    else:
        new_x =  x[Bid, sep_index, ...]  # # Batch x seqlen x Head x dim -->  sep_num x Head x dim    
        return new_x[:min_sep_num, ...].detach().clone() # #  min_sep_num x Head x dim 


def sep_1bat_select_on_2d(x, Bid, sep_index, min_sep_num=None):
    print(f"Debug: #####################x.shape:{x.shape} in sep_1bat_select_on_2d###########################")
    if min_sep_num is None:
        return x[Bid, :, sep_index, ...].detach().clone()  # # Batch x Head x seqlen x dim -->  Head x sep_num x dim    
    else:
        new_x =  x[Bid, :, sep_index, ...].detach().clone()   # # Batch x Head x seqlen x dim -->  Head x sep_num x dim            
        return new_x[:, :min_sep_num, ...].detach().clone() # #  Head x min_sep_num x dim      


def sep_1bat_select_on_3d(x, Bid , sep_index, min_sep_num=None):
    print(f"Debug: #####################x.shape:{x.shape} in sep_1bat_select_on_3d###########################")
    if min_sep_num is None:
        return x[Bid, :, :, sep_index, ...].detach().clone()  # # Batch x Head x dim x seqlen  -->  Head x dim x sep_num 
    else:
        new_x =  x[Bid, :, :,sep_index, ...].detach().clone()    # # Batch x Head x dim x seqlen  -->  Head x dim x sep_num         
        return new_x[:, :, :min_sep_num, ...].detach().clone() # #  Head x dim x min_sep_num 


DIM_TO_SLICE = {
    1: slice_on_1d,
    2: slice_on_2d,
    3: slice_on_3d,
}

BAT_DIM_TO_SELECT = {
    1: sep_1bat_select_on_1d,
    2: sep_1bat_select_on_2d,
    3: sep_1bat_select_on_3d,
}





class SegLLM_KVCache_Manager:
    def __init__(
        self,
        initial_size=4,
        # neighboring_size=512, ##default
        neighboring_size=256,  ##my
        k_seq_dim=2,
        v_seq_dim=2,
        cache_size=1024+64,
        sep_cache_size = 64,
        special_tokens_id = None,
        model_type = 'llama',
    ):
        print(f"Building SegLLM_KVCache_Manager: {initial_size}, {neighboring_size}")
        self.initial_size = initial_size
        self.neighboring_size = neighboring_size
        # self.cache_size = initial_size + neighboring_size
        self.cache_size = cache_size  # my
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        
        ###################my####################
        self.k_bat_dim_select = BAT_DIM_TO_SELECT[k_seq_dim]
        self.v_bat_dim_select = BAT_DIM_TO_SELECT[v_seq_dim]

        self.sep_cache_size= sep_cache_size
        
        self.past_tok_ids = None
        self.sep_exrange = 0 # right boundary for seps, excluded
        self.max_sep_exidx = sep_cache_size + initial_size # max right boundary for seps, excluded
        
        if special_tokens_id is not None:
            # self.special_tokens_id = [13, 11, 30, 0, 26, 25, 198, 220, 128000, 662, 1174, 949, 758, 2652, 551, 720, 256,262] # llama3 8b
            self.special_tokens_id = special_tokens_id
        else:            
            self.special_tokens_id = [13, 11, 30, 0, 26, 25, 198, 220, 662, 1174, 949, 758, 2652, 551, 720, 256,262] # llama3 8b
            # self.special_tokens_id = [13, 11, 30, 0, 26, 25, 198, 220, 128000, 662, 1174, 949, 758, 2652, 551, 720, 256,262] # llama3 8b, add <start-of-seq|SOS>:128000
        #########################################

    def __call__bk(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.initial_size),
                        self.k_slice(k, seq_len - self.neighboring_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.initial_size),
                        self.v_slice(v, seq_len - self.neighboring_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]



    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        # return [
        #     [
        #         torch.cat(
        #             [
        #                 self.k_slice(k, 0, self.initial_size),
        #                 self.k_slice(k, seq_len - self.neighboring_size, seq_len),
        #             ],
        #             dim=self.k_seq_dim,
        #         ),
        #         torch.cat(
        #             [
        #                 self.v_slice(v, 0, self.initial_size),
        #                 self.v_slice(v, seq_len - self.neighboring_size, seq_len),
        #             ],
        #             dim=self.v_seq_dim,
        #         ),
        #     ]
        #     for k, v in past_key_values
        # ]
        
        if self.initial_size > 0:            
            sink_kv = self.slice_kv_4_all_layers(past_key_values, 0, self.initial_size,  CHECK_IDX=True) 
                
        recent_kv = self.slice_kv_4_all_layers(past_key_values, seq_len - self.neighboring_size, seq_len,  CHECK_IDX=True) 
                
        if self.initial_size > 0:            
            past_key_values = self.cat_kv_cache_4_all_layers([sink_kv,  recent_kv])
        else:            
            past_key_values = recent_kv
        
        return past_key_values
        
        
        


    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.initial_size),
                        self.k_slice(
                            k, seq_len - self.neighboring_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.initial_size),
                        self.v_slice(
                            v, seq_len - self.neighboring_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
        
        
    def slice_kv_4_all_layers(self, past_key_values, start, end,  CHECK_IDX=False):
        if CHECK_IDX:            
            seq_len = past_key_values[0][0].size(self.k_seq_dim)
            if start <0 :
                start = start + seq_len
            if end < 0 :
                end = end + seq_len
            assert (start >=0) and (start < seq_len) , f"start:{start}, end:{end}, seq_len:{seq_len}"
            assert (end >= 0) and (end <= seq_len) , f"start:{start}, end:{end}, seq_len:{seq_len}"
            assert  start < end, f"start:{start}, end:{end}, seq_len:{seq_len}"
            
        return [
                    [ self.k_slice(k, start, end),  self.v_slice(v, start, end) ]
                    for k, v in past_key_values
              ]
        
        
    def slice_kv_cache_and_tokids(self, past_key_values, tok_ids, start, end, CHECK_IDX=True ):
        if CHECK_IDX:            
            seq_len = past_key_values[0][0].size(self.k_seq_dim)
            if start <0 :
                start = start + seq_len
            if end < 0 :
                end = end + seq_len
            assert (start >=0) and (start < seq_len), f"start:{start}, end:{end}, seq_len:{seq_len}" 
            assert (end >=0) and (end <= seq_len) , f"start:{start}, end:{end}, seq_len:{seq_len}" 
            assert  start < end, f"start:{start}, end:{end}, seq_len:{seq_len}" 
        
        
        sliced_kv = self.slice_kv_4_all_layers(past_key_values, start, end,  CHECK_IDX=False)        
        sliced_ids = tok_ids[:, start:end].detach().clone()
        
        return sliced_kv , sliced_ids

    def _cat_kv_4_all_layers(self, kv_a,  kv_b):    ## cat the KV for all layers
        assert len(kv_a) == len(kv_b)        
        return [ [ torch.cat( [kv_a[i][0], kv_b[i][0]], dim=self.k_seq_dim),  torch.cat( [kv_a[i][1], kv_b[i][1]], dim=self.v_seq_dim)    ]  for i in range(len(kv_a))  ]
                 

    def cat_kv_cache_4_all_layers(self, past_key_values_list):    
        assert len(past_key_values_list) >= 1 
        
        if len(past_key_values_list) == 1 :
            return past_key_values_list[0]
        else:
            ret = None 
            for i, past_key_values in enumerate(past_key_values_list): # enumerate all the KVs needed to be cat
                if i == 0:
                    ret = past_key_values
                else:
                    ret = self._cat_kv_4_all_layers(ret, past_key_values)
            return ret

    def cat_token_ids(self,tok_ids_list ) :
        assert len(tok_ids_list) >= 1 
        
        return torch.cat(tok_ids_list, dim=-1)        
        
    def cat_kv_cache_and_tokids(self, past_key_values_list, tok_ids_list):
        
        return self.cat_kv_cache_4_all_layers(past_key_values_list), self.cat_token_ids(tok_ids_list)
        
                                

    def compress_past_win_2_seps(self, past_win_kv, past_win_tokids, MIN_SEP_ALERT=False):

        sep_index_tensor = torch.zeros_like(past_win_tokids).bool()  # B x seq_len
        for sp_id in self.special_tokens_id:            
            sep_index_tensor = sep_index_tensor | ( past_win_tokids == sp_id ) # B x seq_len

        sep_cnt = sep_index_tensor.int().sum(-1)
        min_sep_num = sep_cnt.min()  # the min sep number for the seqs in a batch
        
        if MIN_SEP_ALERT:
            assert min_sep_num>0, f"The min sep number for each compressing time in a batch should be at least one"
        # print(f"Debug: ######################sep_index_tensor:{sep_index_tensor}###########################")
        # print(f"Debug: ######################min_sep_num:{min_sep_num}###########################")
                
        batch1_sep_ids_list = []
        batch_size = past_win_tokids.shape[0]
        for b_id in range(batch_size):            
            batch1_sep_ids = past_win_tokids[b_id, sep_index_tensor[b_id]] # #  sep_num
            batch1_sep_ids = batch1_sep_ids[..., :min_sep_num ].detach().clone()  # #  min_sep_num
            batch1_sep_ids_list.append(batch1_sep_ids)                                                           
            
        new_sep_tokids = torch.stack(batch1_sep_ids_list, dim=0) # #  B x min_sep_num
        
        
        new_sep_kv = []
        for k,v in past_win_kv: # for each layer
            '''
            The shape samples listed in the comments are just for Llama3.
            '''
            ## k,v torch.Size([1, 8, 129, 128])) # B x Head x seq x dim
            assert batch_size==k.shape[0]
            # batch_size = k.shape[0]
            batch1_sep_k_list = []
            batch1_sep_v_list = []
            batch1_sep_ids_list = []
            for b_id in range(batch_size):
                # batch1_sep_k = k[b_id,:,sep_index_tensor[b_id],:] # #  Head x sep_num x dim
                # batch1_sep_k = batch1_sep_k[..., :min_sep_num, :].detach().clone() # #  Head x min_sep_num x dim                                
                # batch1_sep_k_list.append( batch1_sep_k ) 
                
                # batch1_sep_v = v[b_id,:,sep_index_tensor[b_id],:] # #  Head x sep_num x dim
                # batch1_sep_v = batch1_sep_v[..., :min_sep_num, :].detach().clone() # #  Head x min_sep_num x dim                                
                # batch1_sep_v_list.append( batch1_sep_v )     

                batch1_sep_k = self.k_bat_dim_select(k, b_id, sep_index_tensor[b_id], min_sep_num)
                batch1_sep_k_list.append(batch1_sep_k)

                batch1_sep_v = self.v_bat_dim_select(v, b_id, sep_index_tensor[b_id], min_sep_num)
                batch1_sep_v_list.append( batch1_sep_v )   
            
            sep_k = torch.stack(batch1_sep_k_list, dim=0)  ## Bx Head x min_sep_num x dim
            sep_v = torch.stack(batch1_sep_v_list, dim=0)  ## Bx Head x min_sep_num x dim           
            new_sep_kv.append( [sep_k, sep_v] )
                
        return new_sep_kv, new_sep_tokids, min_sep_num      

        
    
    def update_past_tok_ids(self, input_ids):    
        if self.past_tok_ids is None:
            self.past_tok_ids = input_ids.detach().clone()
        else:
            self.past_tok_ids = torch.cat([self.past_tok_ids , input_ids], dim=-1)
            
    def compress_kv_cache_and_tokids(self, past_key_values,  SEP_ACCUMULATION=False, USE_MAX_SEP_CACHE=False ):
        """
        SEP_ACCUMULATION: If True, it means we will try to keep all the kv for seperators. If False, only the new_sep_kv compressed from the past_win_kv will be kept.
                                                             
        USE_MAX_SEP_CACHE: If True, it means we only keep self.sep_cache_size seperators' KV.  In the paper, the hyperparameter s is an abbreviated alias for self.sep_cache_size.


        Note: If SEP_ACCUMULATION=True and USE_MAX_SEP_CACHE=False, as the number of input tokens increases, the number of separators in the KV cache will also accumulate endlessly 
              and self.cache_size will be also infinitely expanded.
        """

        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        
        if self.sep_exrange <=0:            
            self.sep_exrange = self.initial_size
                
        assert seq_len - self.neighboring_size > self.sep_exrange
        
        
        # past_win_tokids = self.past_tok_ids[:, self.sep_exrange : seq_len - self.neighboring_size ]
        # # past_win_kv = past_key_values[]
        # past_win_kv = self.slice_4_all_layers(past_key_values, self.sep_range, seq_len - self.neighboring_size )
        # initial_kv = self.slice_4_all_layers(past_key_values,  0, self.initial_size )
        
        if self.initial_size > 0:
            initial_kv, initial_tokids =  self.slice_kv_cache_and_tokids( past_key_values, self.past_tok_ids, 0, self.initial_size, CHECK_IDX=True )
                        
        Before_First_Time_Compress_Flag = (self.sep_exrange == self.initial_size)  ## If true, it means the present timestamp is before t1: the 1st time to compress the past window, in which only seperators' kv are kept.
        
        if SEP_ACCUMULATION and not Before_First_Time_Compress_Flag: ## To get the old seg kv and seg token ids.           
            past_seg_kv, past_seg_tokids =  self.slice_kv_cache_and_tokids( past_key_values, self.past_tok_ids, self.initial_size, self.sep_exrange, CHECK_IDX=True )            
        
        past_win_kv, past_win_tokids =  self.slice_kv_cache_and_tokids( past_key_values, self.past_tok_ids, self.sep_exrange, seq_len - self.neighboring_size, CHECK_IDX=True )        
        neighboring_kv, neighboring_tokids  =  self.slice_kv_cache_and_tokids( past_key_values, self.past_tok_ids, seq_len - self.neighboring_size, seq_len, CHECK_IDX=True )
        
        new_sep_kv, new_sep_tokids, min_sep_num = self.compress_past_win_2_seps( past_win_kv, past_win_tokids) ## To get the new seg kv and seg token ids that were just compressed from the past window
        
        if SEP_ACCUMULATION and not Before_First_Time_Compress_Flag:            
            seg_kv, seg_tokids  = self.cat_kv_cache_and_tokids( [ past_seg_kv, new_sep_kv ] ,  [past_seg_tokids, new_sep_tokids ] )                
            new_seg_len = new_sep_tokids.shape[-1]
            seg_len = seg_tokids.shape[-1]  
            if USE_MAX_SEP_CACHE: ## Fixed sep cache size   
                if self.initial_size + seg_len > self.max_sep_exidx:
                    max_seg_len = self.max_sep_exidx - self.initial_size
                    seg_kv, seg_tokids =  self.slice_kv_cache_and_tokids( seg_kv, seg_tokids, seg_len-max_seg_len, seg_len, CHECK_IDX=True )
                    self.sep_exrange =  self.max_sep_exidx  
                else:
                    self.sep_exrange =  self.initial_size + seg_len                        
            else:    ## Extend the sep cache and the whole cache if USE_MAX_SEP_CACHE is not set                           
                self.sep_exrange =  self.initial_size + seg_len
                if self.sep_exrange > self.max_sep_exidx:                    
                    cache_incremental_gap = self.sep_exrange - self.max_sep_exidx
                    self.max_sep_exidx = self.sep_exrange 
                    self.cache_size = self.cache_size + cache_incremental_gap

        else:
            seg_kv, seg_tokids = new_sep_kv, new_sep_tokids
            # new_seg_len = new_sep_tokids.shape[-1]
            seg_len = seg_tokids.shape[-1]  
            
            assert min_sep_num==seg_len
            
            if USE_MAX_SEP_CACHE:    
                if self.initial_size + seg_len > self.max_sep_exidx:
                    max_seg_len = self.max_sep_exidx - self.initial_size
                    seg_kv, seg_tokids =  self.slice_kv_cache_and_tokids( seg_kv, seg_tokids, seg_len-max_seg_len, seg_len, CHECK_IDX=True )
                    self.sep_exrange =  self.max_sep_exidx  
                else:
                    self.sep_exrange = self.initial_size + seg_len
            else:
                self.sep_exrange = self.initial_size + seg_len
                if self.sep_exrange > self.max_sep_exidx:                    
                    cache_incremental_gap = self.sep_exrange - self.max_sep_exidx
                    self.max_sep_exidx = self.sep_exrange 
                    self.cache_size = self.cache_size + cache_incremental_gap



        if self.initial_size > 0:                                
            past_key_values, self.past_tok_ids  = self.cat_kv_cache_and_tokids( [initial_kv, seg_kv, neighboring_kv ] ,  [initial_tokids, seg_tokids, neighboring_tokids  ] )
        else:
            past_key_values, self.past_tok_ids  = self.cat_kv_cache_and_tokids( [seg_kv, neighboring_kv ] ,  [seg_tokids, neighboring_tokids  ] )
        
        
        return past_key_values, self.past_tok_ids
            
                        
    def evict_except_for_seg(self, past_key_values,  SEP_ACCUMULATION=False, USE_MAX_SEP_CACHE=False):
        """
        SEP_ACCUMULATION: If True, it means we will try to keep all the kv for seperators. If False, only the new_sep_kv compressed from the past_win_kv will be kept.
                                                             
        USE_MAX_SEP_CACHE: If True, it means we only keep self.sep_cache_size seperators' KV.  In the paper, the hyperparameter s is an abbreviated alias for self.sep_cache_size.


        Note: If SEP_ACCUMULATION=True and USE_MAX_SEP_CACHE=False, as the number of input tokens increases, the number of separators in the KV cache will also accumulate endlessly 
              and self.cache_size will be also infinitely expanded.
        """        

        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        
        past_key_values, _ =   self.compress_kv_cache_and_tokids(past_key_values, SEP_ACCUMULATION=SEP_ACCUMULATION, USE_MAX_SEP_CACHE=USE_MAX_SEP_CACHE)
        
        return past_key_values