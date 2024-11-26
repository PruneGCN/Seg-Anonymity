# Copyright (c) 2024 EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT-2 model."""

import math
import torch
import torch.nn as nn
from collections import defaultdict

from functools import partial
from megatron.model.utils import Lambda, SequentialWrapper, recursive_setattr
from megatron.model.norms import get_norm
from megatron.model.init_functions import get_init_methods

from megatron import mpu
from megatron.mpu import ParallelRelativePositionBias
from megatron.model.transformer import (
    ParallelTransformerLayerPipe,
    NormPipe,
    ParallelLinearPipe,
    parallel_lm_logits,
    ParallelLinear,
)
from megatron.model.gmlp import GMLPBlock
from megatron.model.rwkv.v6 import RWKVResidualLayerPipe
from megatron.model.mamba import ParallelMambaResidualLayerPipe
from megatron.model.word_embeddings import EmbeddingPipe, SoftEmbedding

# Pipeline parallelism
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from typing import Union, List

import deepspeed.runtime.utils  as ds_utils  ##my
from ..segmented_mask import TrainingMask #my

def gpt2_attention_mask_func(attention_scores, ltor_mask):
    mask_value = torch.finfo(attention_scores.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.tensor(
        mask_value, dtype=attention_scores.dtype, device=attention_scores.device
    )
    attention_scores.masked_fill_(ltor_mask, mask_value)
    return attention_scores


def cross_entropy(output, labels, _fp16=False):
    """From pretrain_gpt2:forward_step()"""
    """
    if self.fp16_lm_cross_entropy:
        assert output.dtype == torch.half
        loss = mpu.vocab_parallel_cross_entropy(output, labels)
    else:
        loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
        return loss
    """
    labels, loss_mask = labels[0], labels[1]
    if _fp16:
        assert output.dtype == torch.half and loss_mask.dtype == torch.half
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous(), labels)
    else:
        losses = mpu.vocab_parallel_cross_entropy(output.float().contiguous(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


####################################my#############################################
from functools import lru_cache, partial
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)
flex_attention = torch.compile(flex_attention, dynamic=False)
# @lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, KV_BLOCK_SIZE=128, Q_BLOCK_SIZE=128 ,   device="cuda", _compile=False):

    block_mask = create_block_mask(score_mod, B, H, M, N,  BLOCK_SIZE = (KV_BLOCK_SIZE, Q_BLOCK_SIZE) ,  device=device, _compile=_compile)
    return block_mask

#################################################################################


##########################################my###########################################################

###############################default###############################
# def _pre_transformer_block(args, USE_BiPE=False):
#     # data format change for hidden_states to avoid explicit tranposes : [b s h] --> [s b h]
#     assert len(args) == 2, "Incorrect number of arguments to _pre_transformer_block. If USE_BiPE=True, it should be 2"
#     fn = lambda _args: (_args[0].transpose(0, 1).contiguous(), *_args[1:])
#     return fn(args)


# def _post_transformer_block(args):
#     # from (hidden_states, attention_mask)
#     # to (hidden_states.T)
#     assert len(args) == 2, "Incorrect number of arguments to _post_transformer_block"
#     fn = lambda _args: (_args[0].transpose(0, 1).contiguous())
#     return fn(args)
###############################default###############################

class _pre_transformer_block(object):
    def __init__(self, neox_args):
        self.USE_BiPE = neox_args.USE_BiPE
        self.USE_FLEX = neox_args.USE_FLEX
    
    def __call__(self, args):
        if self.USE_BiPE : ##my
            # data format change for hidden_states to avoid explicit tranposes : [b s h] --> [s b h]
            assert len(args) == 3, f"Incorrect number of arguments ({len(args)}) to _pre_transformer_block. If USE_BiPE=True, it should be 3."
            fn = lambda _args: (_args[0].transpose(0, 1).contiguous(), *_args[1:])
        elif self.USE_FLEX:
            assert len(args) == 3, f"Incorrect number of arguments ({len(args)}) to _pre_transformer_block. If USE_FLEX=True, it should be 3."
            fn = lambda _args: (_args[0].transpose(0, 1).contiguous(), *_args[1:])        
        else: ##Default
            # data format change for hidden_states to avoid explicit tranposes : [b s h] --> [s b h]
            assert len(args) == 2, f"Incorrect number of arguments ({len(args)}) to _pre_transformer_block. If USE_BiPE=False, it should be 2"
            fn = lambda _args: (_args[0].transpose(0, 1).contiguous(), *_args[1:])
        return fn(args)

class _post_transformer_block(object):
    def __init__(self, neox_args):
        self.USE_BiPE = neox_args.USE_BiPE
        self.USE_FLEX = neox_args.USE_FLEX
    
    def __call__(self, args):
        if self.USE_BiPE : ##my
            assert len(args) == 3, f"Incorrect number of arguments ({len(args)}) to _post_transformer_block. If USE_BiPE=True, it should be 3."
            fn = lambda _args: (_args[0].transpose(0, 1).contiguous())
        
        elif self.USE_FLEX:
            assert len(args) == 3, f"Incorrect number of arguments ({len(args)}) to _pre_transformer_block. If USE_FLEX=True, it should be 3."
            fn = lambda _args: (_args[0].transpose(0, 1).contiguous())            
                
        else: ##Default
            # from (hidden_states, attention_mask)
            # to (hidden_states.T)
            assert len(args) == 2, f"Incorrect number of arguments ({len(args)})  to _post_transformer_block. If USE_BiPE=False, it should be 2"
            fn = lambda _args: (_args[0].transpose(0, 1).contiguous())
        return fn(args)


# def _post_transformer_block(args):
#     # from (hidden_states, attention_mask)
#     # to (hidden_states.T)
#     assert len(args) == 2, "Incorrect number of arguments to _post_transformer_block"
#     fn = lambda _args: (_args[0].transpose(0, 1).contiguous())
#     return fn(args)


class GPT2ModelPipe(PipelineModule, torch.nn.Module):
    """GPT2Model adapted for pipeline parallelism.

    The largest change is flattening the GPTModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.

    :param neox_args: NeoX arguments object (configuration)
    :param num_tokentypes: number of token types (TODO: deprecated, remove)
    :param parallel_output: if true, don't gather the output logits, and calculate loss in parallel. Set to true by default in training for efficiency, but set to false for inference.
    :param topology: deepspeed topology object specifying pipe / model parallelism topology.
    :param use_cache: if true, cache key/value pairs for each layer in inference.
    """

    def __init__(
        self,
        neox_args,
        num_tokentypes=0,
        parallel_output=True,
        topology=None,
        use_cache=False,
    ):
        self.neox_args = neox_args

        self.use_cache = use_cache
        self.parallel_output = parallel_output
        self.hidden_size = self.neox_args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method, self.output_layer_init_method = get_init_methods(
            self.neox_args
        )
        self.__topology__ = topology

        self.specs = []
        self.init_specs()  # initializes the layer specs (basically a fancy nn.Sequential)

        super().__init__(
            layers=self.specs,
            loss_fn=partial(cross_entropy, _fp16=self.neox_args.fp16_lm_cross_entropy),
            topology=topology,
            activation_checkpoint_interval=self.neox_args.checkpoint_num_layers
            if self.neox_args.checkpoint_activations
            else 0,
            partition_method=neox_args.pipe_partition_method,
            checkpointable_layers=[
                "GMLPBlock",
                "ParallelTransformerLayerPipe",
                "ParallelMambaResidualLayerPipe",
            ],
        )
        
        # ##########################my###########################   
        #### Will be initialized for GPU_number times
        # print("################in GPT2ModelPipe init######################") 
        # if not self.neox_args.original_flag:                
        #     self.neox_args.trnMask = TrainingMask(neox_args)
        # else:
        #     self.neox_args.trnMask = None    
        # #######################################################

    def insert_layers(
        self, layers: Union[nn.Module, nn.ModuleList, nn.Sequential, List], idx
    ):
        """
        inserts the layers in `layers` into the pipe model at `idx`.
        """
        if isinstance(layers, nn.Module):
            self.specs.insert(idx, layers)
        elif any(
            [isinstance(layers, nn.ModuleList), isinstance(layers, nn.Sequential)]
        ):
            self.specs[idx:idx] = layers
        elif isinstance(layers, list):
            assert all(
                [hasattr(l, "__call__") for l in layers]
            ), "all items in `layers` must be Callables"
            self.specs[idx:idx] = layers
        else:
            raise ValueError(
                f"layer passed into {self.__class__.__name__}.insert_layer() should be either an nn.Module, an nn.ModuleList, an nn.Sequential object, or a list of callables not a {type(layers)}"
            )

        # re-initialize parent class
        super().__init__(
            layers=self.specs,
            loss_fn=self.loss_fn,
            topology=self.__topology__,
            activation_checkpoint_interval=self.activation_checkpoint_interval,
            partition_method=self.neox_args.pipe_partition_method,
            checkpointable_layers=[
                "GMLPBlock",
                "ParallelTransformerLayerPipe",
                "ParallelMambaResidualLayerPipe",
                "RWKVResidualLayerPipe",
            ],
        )

    def init_specs(self):

        weight_tying = not self.neox_args.no_weight_tying   #my: for pythia, no_weight_tying is True.
        self.specs = []

        # Embedding layer
        # input will be (input_ids, position_ids, attention_mask)

        if weight_tying: #my: False
            self.specs.append(
                TiedLayerSpec(
                    "embed",
                    EmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                    tied_weight_attr="word_embeddings_weight",
                )
            )
        else:
            self.specs.append(
                LayerSpec(
                    EmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                )
            )

        # NB: the attention mask always needs to be the *last* item in the args when being passed from
        # one stage to the next, because deepspeed is hacks on top of hacks.
        #
        # outputs are now (hidden_states,  attention_mask)

        # self.specs.append(_pre_transformer_block(self.neox_args)) ##default
        self.specs.append(_pre_transformer_block(self.neox_args)) ##my

        # T5 RPE positional embedding
        if self.neox_args.pos_emb == "rpe":
            hidden_size_per_attention_head = mpu.divide(
                self.neox_args.hidden_size, self.neox_args.num_attention_heads
            )
            rpe_scale = math.sqrt(hidden_size_per_attention_head)
            rpe_emb = ParallelRelativePositionBias(
                neox_args=self.neox_args,
                scale=rpe_scale,
                causal=True,
                num_buckets=self.neox_args.rpe_num_buckets,
                max_distance=self.neox_args.rpe_max_distance,
                heads=self.neox_args.num_attention_heads,
            )

        # Transformer layers
        for i in range(self.neox_args.num_layers):
            layer_type = self.neox_args.attention_config[i]
            if layer_type in ["gmlp", "amlp"]:
                self.specs.append(
                    LayerSpec(
                        GMLPBlock,
                        init_method=self.init_method,
                        layer_number=i,
                        output_layer_init_method=self.output_layer_init_method,
                        neox_args=self.neox_args,
                        mask_fn=gpt2_attention_mask_func,
                    )
                )
            elif layer_type == "rwkv":
                self.specs.append(
                    LayerSpec(
                        RWKVResidualLayerPipe,
                        neox_args=self.neox_args,
                        layer_number=i,
                    )
                )
            elif layer_type in ["mamba"]:
                self.specs.append(
                    LayerSpec(
                        ParallelMambaResidualLayerPipe,
                        neox_args=self.neox_args,
                        init_method=self.init_method,
                        output_layer_init_method=self.output_layer_init_method,
                        layer_number=i,
                    )
                )
            else:
                self.specs.append(
                    LayerSpec(
                        ParallelTransformerLayerPipe,
                        neox_args=self.neox_args,
                        attention_mask_func=gpt2_attention_mask_func,
                        init_method=self.init_method,
                        output_layer_init_method=self.output_layer_init_method,
                        layer_number=i,
                        rpe=rpe_emb if self.neox_args.pos_emb == "rpe" else None,
                        rotary=self.neox_args.pos_emb == "rotary",
                        use_cache=self.use_cache,
                    )
                )

        # used to drop attention mask + reshape hidden states
        # self.specs.append(_post_transformer_block) ##default
        self.specs.append(_post_transformer_block(self.neox_args))##my

        # NormPipe is a (deprecated) helper class that used to be used to pass presents along the pipeline - since presents are now cached to the `TransformerLayer` class this is no longer needed
        norm, eps = get_norm(self.neox_args)
        self.specs.append(
            LayerSpec(NormPipe, norm, self.neox_args.hidden_size, eps=eps)
        )

        # outputs are now a single tensor: hidden_states

        def _logits_helper(embedding, lm_output):
            """Just a wrapper to massage inputs/outputs from pipeline."""
            if self.neox_args.use_mup:
                # Since we're using pipeline parallelism, we can't directly use MuReadout. Instead, use this workaround that does the same thing as MuReadout.
                # https://github.com/microsoft/mup/issues/6#issuecomment-1082156274
                lm_output = (
                    lm_output
                    / self.tied_modules.embed.word_embeddings.weight.infshape.width_mult()
                )

            logits = parallel_lm_logits(
                lm_output,
                embedding.word_embeddings_weight,
                self.parallel_output,
                seq_parallel=self.neox_args.sequence_parallel,
            )
            return logits

        if weight_tying: ##my: False
            self.specs.append(
                TiedLayerSpec(
                    "embed",
                    EmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                    forward_fn=_logits_helper,
                    tied_weight_attr="word_embeddings_weight",
                )
            )
        else:
            self.specs.append(
                LayerSpec(
                    ParallelLinearPipe,
                    neox_args=self.neox_args,
                    init_method=self.init_method,
                    parallel_output=self.parallel_output,
                    is_last_layer=True,
                )
            )

    def _set_parallel_output(self, value):
        # sets the parallel output value of the final layer to value
        final_layer = list(self.forward_funcs)[-1]
        if isinstance(final_layer, (ParallelLinearPipe, ParallelLinear)):
            final_layer.final_linear.set_parallel_output(value)

    def inference_mode(self, use_cache=True):
        """
        Sets up the model for inference by turning on k/v caching (if specified) and setting `parallel output` of the final layer to false,
        so logits are gathered across model parallel ranks.

        :param cache: (bool) True if you want to use caching during inference, False otherwise
        """
        # first set caching to true if specified
        recursive_setattr(self.forward_funcs, "use_cache", use_cache, assert_type=bool)
        # then set parallel output of the final layer to false so we don't have to gather the output manually
        self._set_parallel_output(False)
        recursive_setattr(self.forward_funcs, "training", False)

    def train_mode(self):
        """
        Sets up the model for training by turning off k/v caching and setting `parallel output` of the final layer to True,
        so logits are not gathered across model parallel ranks, and loss is computed in parallel (more efficient).
        """
        # set caching to false
        recursive_setattr(self.forward_funcs, "use_cache", False)
        # then set parallel output to true (more efficient training)
        self._set_parallel_output(True)
        recursive_setattr(self.forward_funcs, "training", True)

    def clear_cache(self):
        """
        Recursively clears the kv cache on all layers
        """
        recursive_setattr(self.forward_funcs, "layer_past", None)

    def to_sequential(self):
        """
        Transforms the PipelineModule to a plain nn.Sequential module
        :return:
        """

        ##my: This func will not be run since For Pythia: neox_args.is_pipe_parallel=True. Please check the training.py file to understand. 

        layers = []
        tied_layers = defaultdict(list)
        for n, spec in enumerate(self.specs):
            if isinstance(spec, TiedLayerSpec):
                if spec.key in tied_layers:
                    # receiver
                    layers.append(
                        Lambda(lambda x: spec.forward_fn(tied_layers[spec.key][0], x))
                    )
                else:
                    # owner
                    module = spec.build(log=False)
                    layers.append(module)
                    tied_layers[spec.key].append(module)
            elif isinstance(spec, LayerSpec):
                layers.append(spec.build(log=False))
            elif hasattr(spec, "__call__"):
                # check that it's a callable function
                layers.append(Lambda(spec))
            else:
                raise ValueError(f"Layer number {n} ({spec}) Not recognized")
        model = SequentialWrapper(
            layers,
            self.activation_checkpoint_interval,
            self.activation_checkpoint_func,
            parent_class_name=self.__class__.__name__,
        )
        return model
    def forward(self, forward_input):   ##my adding: I add this forward funtion to override the original forward function of PipelineModule. 
        """
        I add this forward funtion to override the original forward function of PipelineModule class. 
        Note: when 'pipe-parallel-size' is set to 1 (which is the default setting for Pythia models), the training process will run the 'train_step_pipe' branch in the
        'pretrain()->train()->train_step()->train_step_pipe() function' in 'megatron.training file'  instead of 'forward_step() function'.  
        In 'train_step_pipe()', it will run the line 'loss = model.train_batch(data_iter=data_iterator)' in which 'train_batch() function' is defined in 'deepspeed.runtime.pipe.engine' file.
        In 'train_batch()' function, it will finally run the 'forward() function of PipelineModule' (defined in 'deepspeed.runtime.pipe.module'), which will be finally override by this 'forward()' function.
        
        """
        # We need to offset the seed by the microbatch ID. Save it in a local var to
        # ensure it is preserved in the closure. Otherwise checkpointed forward funcs
        # will see a different offset.

        #########################my#############################
        # print(f"########################in PipelineModule(nn.Module): forward_input { type(forward_input),forward_input} ###############################")
        
        # print(f"########################in PipelineModule(nn.Module): attention_mask {forward_input[2]} ###############################")
        # print(f"######################## my ###############################")
        
        # ########################in PipelineModule(nn.Module): forward_input (<class 'tuple'>, (tensor([[  273,  3307,   273,  ...,  3781,  1587, 14477],
        # [ 1012,  3141,   187,  ...,   326,   310,    13]], device='cuda:0'), tensor([[   0,    1,    2,  ..., 2045, 2046, 2047],
        # [   0,    1,    2,  ..., 2045, 2046, 2047]], device='cuda:0'), tensor([[[[False,  True,  True,  ...,  True,  True,  True],
        #   [False, False,  True,  ...,  True,  True,  True],
        #   [False, False, False,  ...,  True,  True,  True],
        #   ...,
        #   [False, False, False,  ..., False,  True,  True],
        #   [False, False, False,  ..., False, False,  True],
        #   [False, False, False,  ..., False, False, False]]]], device='cuda:0'))) ###############################

        
        
        ########################################################


        ################################################################################################my##################################################################################################
        if not self.neox_args.original_flag:
            trnMask = self.neox_args.trnMask
            # trnMask = TrainingMask(neox_args)
            assert not (trnMask is None), "trnMask must not be None"
        else:
            pass

        # print("######################tokens########################")
        # print(forward_input[0].shape)  # B x seq_len. Actully, it's the input_ids for HuggingFace transformer format. No padding tokens in it when pretraining
        # print(forward_input[0])
        
        # print("######################labels ########################")
        # print(labels.shape)  # B x seq_len. No padding in the labels (i.e., normally -100) when pretraining
        # print(labels)
            
        # print("######################attention mask ori########################")
        # print(forward_input[-1].shape) # 1 x 1 x seq_len x seq_len. By default, the 1st dim is not B.  torch.Size([1, 1, 2048, 2048])
        # print(forward_input[-1])

        # print(f"######################trnMask.print_KV_count: {trnMask.print_KV_count}########################")

        if not self.neox_args.original_flag:  
            attention_mask = forward_input[-1]
            input_ids = forward_input[0]
            if input_ids.shape[-1] > 1: # for prefilling
                del trnMask.past_ids[:]
                del trnMask.past_ids
                trnMask.past_ids = [] 
                trnMask.past_considered_seps_idx = [-1]
                trnMask.past_kept_tok_idx = []
                trnMask.batch_prefill_max_seq_len = input_ids.shape[-1]
                k  = trnMask.prefill_k            
                if  trnMask.PRINT_KV_RATIO: # when evaluating, print results
                    trnMask.kept_tokens_count_total = ( trnMask.kept_tokens_count_seq[0] + trnMask.kept_tokens_count_total[0], trnMask.kept_tokens_count_seq[1] + trnMask.kept_tokens_count_total[1])
                    trnMask.kept_attmap_count_total = ( trnMask.kept_attmap_count_seq[0] + trnMask.kept_attmap_count_total[0], trnMask.kept_attmap_count_seq[1] + trnMask.kept_attmap_count_total[1])
                    trnMask.print_KV_count += 1                    


                    if trnMask.print_KV_count % trnMask.print_KV_intervals == 0:
                        # print("#############################print_KV_count in forward of GPT2ModelPipe###############################")                        
                        # print(f"trnMask.print_KV_count:     {trnMask.print_KV_count}")
                        # print(f"trnMask.print_KV_intervals: {trnMask.print_KV_intervals}")
                        
                        
                        print("###############################trnMask.kept_tokens_count_seq#####################################")
                        print(f"trnMask.kept_tokens_count_seq (kept, total) : {trnMask.kept_tokens_count_seq}, ratio: {(trnMask.kept_tokens_count_seq[0]+1e-6) / (trnMask.kept_tokens_count_seq[1]+1e-6)} ", flush=True)
                        print()
                        print("###############################trnMask.kept_tokens_count_total for now#####################################")
                        print(f"trnMask.kept_tokens_count_total (kept, total) : {trnMask.kept_tokens_count_total}, ratio: { (trnMask.kept_tokens_count_total[0]+1e-6) / (trnMask.kept_tokens_count_total[1] +1e-6) } ", flush=True)
                        print()


                        print("###############################trnMask.kept_attmap_count_seq#####################################")
                        print(f"trnMask.kept_attmap_count_seq (kept, total) : {trnMask.kept_attmap_count_seq}, ratio: {(trnMask.kept_attmap_count_seq[0]+1e-6) / (trnMask.kept_attmap_count_seq[1]+1e-6)} ", flush=True)
                        print()
                        print("###############################trnMask.kept_attmap_count_total for now#####################################")
                        print(f"trnMask.kept_attmap_count_total (kept, total) : {trnMask.kept_attmap_count_total}, ratio: { (trnMask.kept_attmap_count_total[0]+1e-6) / (trnMask.kept_attmap_count_total[1] +1e-6) } ", flush=True)
                        print()

                trnMask.kept_tokens_count_seq = (0,0)                        
                trnMask.kept_attmap_count_seq = (0,0)         
                prefill_flag = True
            else:
                k = trnMask.decode_k
                prefill_flag = False            


            trnMask.past_ids.append(input_ids)        
            past_ids = torch.cat(trnMask.past_ids, dim=-1)

            # past_ids = input_ids


            # min_val_dtype = min(torch.finfo( trnMask.dtype).min,  trnMask.NPU_MIN )
            # print("##########min_val_dtype#################")
            # print(min_val_dtype)

            
            # causal_mask1 = attention_mask  # B  x 1 x seqlen x seqlen
            # causal_mask2 = attention_mask.clone().detach().expand(input_ids.shape[0], attention_mask.shape[-3], attention_mask.shape[-2], attention_mask.shape[-1] )  # B x 1 x seqlen x seqlen

            ##################### used ##########################
            # causal_mask1 = attention_mask  # B  x 1 x seqlen x seqlen ## deprecated
            # causal_mask2 = attention_mask.expand(input_ids.shape[0], attention_mask.shape[-3], attention_mask.shape[-2], attention_mask.shape[-1] )  # B x 1 x seqlen x seqlen
            causal_mask2 = attention_mask.expand(input_ids.shape[0], attention_mask.shape[-3], attention_mask.shape[-2], attention_mask.shape[-1] ).clone().detach()  # B x 1 x seqlen x seqlen
            causal_mask2 = trnMask.reverse_bool_mask(causal_mask2)
            
            att_sink_idx_tensor = None
            # del attention_mask ## pythia        

            if prefill_flag:
                if trnMask.BATCH_DYNAMIC_ATT_SINK and (not trnMask.original_flag):
                    att_sink_idx_tensor, trnMask.recyc_sink_pos = trnMask.build_eval_att_sink_index(input_ids, causal_mask2, trnMask.batch_prefill_max_seq_len  ,trnMask.att_sink_max_idx + 1 , trnMask.PADDING_ID )
            else:
                if trnMask.BATCH_DYNAMIC_ATT_SINK and (not trnMask.original_flag): # when evaluating, print results. Actually no need to add this IF statement since no shift-right generation for sft
                    att_sink_idx_tensor, _ = trnMask.build_eval_att_sink_index(trnMask.past_ids[0], causal_mask2, trnMask.batch_prefill_max_seq_len ,trnMask.att_sink_max_idx + 1 , trnMask.PADDING_ID, prefill_sink_pos_tensor=trnMask.recyc_sink_pos)

            causal_mask2 = trnMask.build_segmented_attention_mask(prefill_flag, past_ids, causal_mask2,  BATCH_DYNAMIC_ATT_SINK=trnMask.BATCH_DYNAMIC_ATT_SINK, att_sink_idx_tensor = att_sink_idx_tensor )
            
            # print("####################att_sink_idx_tensor#########################")
            # print(att_sink_idx_tensor.shape)
            # print(att_sink_idx_tensor)
            # print(att_sink_idx_tensor.int().sum(-1))
            if  trnMask.PRINT_KV_RATIO:
                trnMask.count_prefill_kept_kv_all_layers(causal_mask2)
                trnMask.count_prefill_kept_attmap_all_layers(causal_mask2)

            if self.neox_args.USE_FLEX:
                attention_mask = causal_mask2
            else:
                attention_mask = trnMask.reverse_bool_mask(causal_mask2)

            del causal_mask2 ##pythia 
            del att_sink_idx_tensor ##pythia 
            ##################### used ##########################
            
            forward_input = (forward_input[0], forward_input[1], attention_mask)
        else:
            pass





        if self.neox_args.USE_FLEX:        
            # B, H, Sq, D = query.shape

            B,  Sq  = input_ids.shape[0], input_ids.shape[-1], 
            Sk = Sq 
            idx0 = torch.zeros(1, dtype=torch.int, device=input_ids.device)
            # print(f"######################{[B, H, Sq, D, Sk]}")
            def mask_mod(b, h, q_idx, kv_idx):
                aa = attention_mask[b, idx0, q_idx, kv_idx]
                # aa = aa > torch.zeros_like(aa)
                # aa = aa > q_idx
                # aa = q_idx > kv_idx
                # print(f"############ aa: {aa.shape}")
                return aa.view([]).detach().clone()
    
            block_mask = create_block_mask_cached(mask_mod, B, 1, Sq, Sk, KV_BLOCK_SIZE=256, Q_BLOCK_SIZE=256, device=input_ids.device, _compile=False)
            
            forward_input = (input_ids, forward_input[1],  block_mask, attention_mask)
                        
        else:
            if self.neox_args.USE_BiPE:
                
                input_ids = forward_input[0]
                position_ids = forward_input[1]
                attention_mask = forward_input[-1]
                intra_position_ids,  inter_position_ids = trnMask.get_bilevel_ids(input_ids)
                forward_input = (input_ids, intra_position_ids,  inter_position_ids, attention_mask)
            else:
                pass
        # ################################################my#################################################
        # import os

        # torch.set_printoptions(profile="full")
        # print(f">>>>>>>>>>>>>>>>>>>attention_mask for layer {3}:  RANK: {os.getenv('RANK')}:  {(~(forward_input[-1][3]))[:,:,100,:]}.<<<<<<<<<<<<<<<<<<<")
        # torch.set_printoptions(profile="default")
        # ###################################################################################################


        # print(f"################# for one input:  tokens: {tokens.shape},  position_ids: {position_ids.shape}, attention_mask: {attention_mask.shape}##############################################")
        ##################################################################################################################################################################################################
    


        self.micro_offset += 1

        def exec_range_func(start, end):
            ''' Helper function to be used with checkpoint()
            Adapted from torch.utils.checkpoint:checkpoint_sequential()
            '''
            local_micro_offset = self.micro_offset + 1

            def exec_func(*inputs):
                # Single tensor inputs need to be unwrapped
                if len(inputs) == 1:
                    inputs = inputs[0]
                for idx, layer in enumerate(self.forward_funcs[start:end]):
                    self.curr_layer = idx + self._local_start
                    if self.seed_layers:
                        new_seed = (self.base_seed * local_micro_offset) + self.curr_layer
                        if self.seed_fn:
                            self.seed_fn(new_seed)
                        else:
                            ds_utils.set_random_seed(new_seed)

                    inputs = layer(inputs)
                return inputs

            return exec_func

        if self.activation_checkpoint_interval == 0:
            func = exec_range_func(0, len(self.forward_funcs))
            x = func(forward_input)
        else:
            num_layers = len(self.forward_funcs)
            x = forward_input
            for start_idx in range(0, num_layers, self.activation_checkpoint_interval):
                end_idx = min(start_idx + self.activation_checkpoint_interval, num_layers)

                funcs = self.forward_funcs[start_idx:end_idx]
                # Since we either pass tensors or tuples of tensors without unpacking, we
                # need to be careful not to double-wrap tensors with tuple.
                if not isinstance(x, tuple):
                    x = (x, )

                if self._is_checkpointable(funcs):
                    x = self.activation_checkpoint_func(exec_range_func(start_idx, end_idx), *x)
                else:
                    x = exec_range_func(start_idx, end_idx)(*x)
        return x
    
