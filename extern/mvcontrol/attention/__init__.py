import torch
from diffusers.models import Transformer2DModel
from .attention_processors import (
    CrossViewAttnProcessor,
    XFormersCrossViewAttnProcessor,
)

def set_self_attn_processor(model, processor):
    r"""
    Parameters:
        `processor (`dict` of `AttentionProcessor` or `AttentionProcessor`):
            The instantiated processor class or a dictionary of processor classes that will be set as the processor
            of **all** `Attention` layers.
        In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainable attention processors.:

    """

    def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
        if hasattr(module, "set_processor") and 'attn1' in name:
            if not isinstance(processor, dict):
                module.set_processor(processor)
            else:
                module.set_processor(processor.pop(f"{name}.processor"))

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

    for name, module in model.named_children():
        fn_recursive_attn_processor(name, module, processor)

# def set_self_attn_trainable(model, params_list=None):
    
#     def make_trainable(module, params_list):
#         module.to(dtype=torch.float32)
#         module.requires_grad_(True)
#         params_list += list(module.parameters())

#     def fn_recursive_attn_trainable(module: torch.nn.Module, params_list):
#         if hasattr(module, "transformer_blocks"):
#             for sub_module in module.transformer_blocks:
#                 make_trainable(sub_module.attn1, params_list)
#         for name, sub_module in module.named_children():
#             fn_recursive_attn_trainable(sub_module, params_list)
    
#     for name, module in model.named_children():
#         fn_recursive_attn_trainable(module, params_list)

def set_self_attn_trainable(unet):
    def make_trainable(module):
        module.to(dtype=torch.float32)
        module.requires_grad_(True)

    def fn_recursive_attn_trainable(module: torch.nn.Module):
        if isinstance(module, Transformer2DModel) and hasattr(module, "transformer_blocks"):
            for sub_module in module.transformer_blocks:
                make_trainable(sub_module.attn1)    # self-attention trainable
        else:
            for name, sub_module in module.named_children():
                fn_recursive_attn_trainable(sub_module)
                
    for name, module in unet.named_children():
        fn_recursive_attn_trainable(module)
        
        
def get_self_attn_params(unet, lr=None):
    params_list = []
    
    def fn_recursive_get_self_attn_params(module: torch.nn.Module, params_list, lr=None):
        if isinstance(module, Transformer2DModel) and hasattr(module, "transformer_blocks"):
            for sub_module in module.transformer_blocks:
                if lr is not None:
                    params_list += [{'params': sub_module.attn1.parameters(), 'lr': lr}]
                else:
                    params_list += list(sub_module.attn1.parameters())
        else:
            for name, sub_module in module.named_children():
                fn_recursive_get_self_attn_params(sub_module, params_list, lr)
    
    for name, module in unet.named_children():
        fn_recursive_get_self_attn_params(module, params_list, lr)
        
    return params_list