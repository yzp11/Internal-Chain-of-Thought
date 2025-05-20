from jaxtyping import Float
import functools

from transformer_lens import  HookedTransformer, utils
import torch as t
from torch import Tensor
import einops

from src.hook import save_one_pos_hook


def activation_stack_to_prob(
    model: HookedTransformer,
    activation_stack: Float[Tensor, "... batch d_model"],
    scale: Float[Tensor, "batch 1"],
) -> Float[Tensor, "... batch d_vocab"]:
    
    if model.cfg.normalization_type in ["LN", "LNPre"]:
        scaled_activation_stack = activation_stack - activation_stack.mean(dim=-1, keepdim=True)
    else:
        scaled_activation_stack = activation_stack.clone()
    for i in range(len(activation_stack.shape) - len(scale.shape)):
        scale = scale.unsqueeze(0)
    scaled_activation_stack = scaled_activation_stack / scale

    return einops.einsum(
        scaled_activation_stack, model.W_U,
        "... batch d_model, d_model d_vocab -> ... batch d_vocab"
    ).softmax(dim= -1)


def save_activation_and_scale(
    model: HookedTransformer,
    prompt: list[str],
    saving_mode: str = 'attn_out' or 'mlp_out' or 'resid_post' or 'both',
    pos_ids = -1,
) -> tuple[Float[Tensor, 'n_layer batch d_model'], Float[Tensor, 'batch 1']]:
    resid_save = []
    scale_save = []

    model.reset_hooks()
    if saving_mode == 'both':
        model.add_hook(lambda name: name.endswith(f'hook_attn_out'),
                functools.partial(save_one_pos_hook, save_list= resid_save, pos_ids = pos_ids))
        model.add_hook(lambda name: name.endswith(f'hook_mlp_out'),
                functools.partial(save_one_pos_hook, save_list= resid_save, pos_ids = pos_ids))
    else:
        model.add_hook(lambda name: name.endswith(f'hook_{saving_mode}'),
                functools.partial(save_one_pos_hook, save_list= resid_save, pos_ids = pos_ids))
    model.add_hook(f'ln_final.hook_scale',
            functools.partial(save_one_pos_hook, save_list= scale_save, pos_ids = pos_ids))
    model(prompt)

    return t.stack(resid_save), scale_save[0]
