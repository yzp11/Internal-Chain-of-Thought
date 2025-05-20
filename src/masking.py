from jaxtyping import Float
import functools

from transformer_lens import  HookedTransformer, utils
import torch as t
from torch import Tensor
import einops
from src.hook import resid_masking_hook, attention_masking_hook, pattern_masking_hook



def sentence_masking(
    model: HookedTransformer,
    prompt: list[str],
    layer: int,
    start: list[int],
    position: list[int],
    masking_type: str = 'resid' or 'attn' or 'pattern',
)-> Float[Tensor, "batch vocab"]:
    model.reset_hooks()
    if masking_type == 'resid':
        model.add_hook(f'blocks.{layer}.hook_resid_post',
                functools.partial(resid_masking_hook, starts= start, positions= position))
    elif masking_type == 'attn':
        for attn_l in range(layer, model.cfg.n_layers):
            model.add_hook(f'blocks.{attn_l}.attn.hook_attn_scores',
                functools.partial(attention_masking_hook, starts= start, positions= position))
    elif masking_type == 'pattern':
        for attn_l in range(layer, model.cfg.n_layers):
            model.add_hook(f'blocks.{attn_l}.attn.hook_pattern',
                functools.partial(pattern_masking_hook, starts= start, positions= position))

    prob = model(prompt)[:, -1, :].softmax(dim = -1)
    
    return prob
    

def mask_and_generate(
    model: HookedTransformer,
    prompt: list[str],
    max_tokens: int,
    layer: int,
    start: list[int],
    end: list[int],
    masking_type: str = 'resid' or 'attn',
    temperature: float = 1.0
) -> list[str]:
    tokens = model.to_tokens(prompt, prepend_bos=False)
    model.reset_hooks()
    if masking_type == 'resid':
        model.add_hook(f'blocks.{layer}.hook_resid_post',
                functools.partial(resid_masking_hook, starts= start, positions= end))
    elif masking_type == 'attn':
        for attn_l in range(layer, model.cfg.n_layers):
            model.add_hook(f'blocks.{attn_l}.attn.hook_attn_scores',
                functools.partial(attention_masking_hook, starts= start, positions= end))
    response = model.generate(tokens, max_new_tokens= max_tokens, temperature= temperature)
    return model.to_string(response)