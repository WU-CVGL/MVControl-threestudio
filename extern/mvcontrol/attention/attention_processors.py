import torch
from typing import Optional, Callable
from einops import rearrange
from diffusers.models.attention_processor import (
    Attention,
    LoRAAttnProcessor, 
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor, 
    AttnProcessor
)
from diffusers.utils import is_xformers_available

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class CrossViewAttnProcessor:
    def __init__(self, num_views: int = 1, batch_size: int = 1):
        self.num_views = num_views

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if not is_cross_attention and self.num_views > 1:
            query = rearrange(query, "(b n) l d -> b (n l) d", n=self.num_views)
            key = rearrange(key, "(b n) l d -> b (n l) d", n=self.num_views)
            value = rearrange(value, "(b n) l d -> b (n l) d", n=self.num_views)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if not is_cross_attention and self.num_views > 1:
            hidden_states = rearrange(hidden_states, "b (n l) d -> (b n) l d", n=self.num_views)

        return hidden_states
    
class XFormersCrossViewAttnProcessor:
    def __init__(
        self, 
        num_views: int = 1, 
        attention_op: Optional[Callable] = None,
    ):
        self.num_views = num_views
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if not is_cross_attention and self.num_views > 1:
            query = rearrange(query, "(b n) l d -> b (n l) d", n=self.num_views)
            key = rearrange(key, "(b n) l d -> b (n l) d", n=self.num_views)
            value = rearrange(value, "(b n) l d -> b (n l) d", n=self.num_views)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if not is_cross_attention and self.num_views > 1:
            hidden_states = rearrange(hidden_states, "b (n l) d -> (b n) l d", n=self.num_views)

        return hidden_states
