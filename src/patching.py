from jaxtyping import Float
import functools

from transformer_lens import  HookedTransformer, utils
import torch as t
from torch import Tensor
import einops

from src.hook import save_one_pos_hook, act_patching_hook


def save_activation(
    model: HookedTransformer,
    prompt: list[str],
    saving_mode: str = 'attn_out' or 'mlp_out' or 'resid_post',
    pos_ids = -1,
) -> Float[Tensor, 'n_layer d_model']:
    resid_save = []

    model.reset_hooks()
    model.add_hook(lambda name: name.endswith(f'hook_{saving_mode}'),
            functools.partial(save_one_pos_hook, save_list= resid_save, pos_ids = pos_ids))
    model(prompt)

    return (t.stack(resid_save)).sum(dim= 1)


def patch_activation(
    model: HookedTransformer,
    prompt: list[str],
    act: Float[Tensor, "d_model"],
    layer: int,
    patching_mode: str = 'attn_out' or 'mlp_out' or 'resid_post',
    replace: str = False,
    pos_ids = -1,
):
    model.reset_hooks()
    model.add_hook(f'blocks.{layer}.hook_{patching_mode}',
            functools.partial(act_patching_hook, tar_act= act, replace= replace, pos_ids = pos_ids))
    logits = model(prompt)[:, -1, :].softmax(dim = -1)
    return logits


def base_run(
    model: HookedTransformer,
    prompt: list[str],
):
    model.reset_hooks()
    logits = model(prompt)[:, -1, :].softmax(dim = -1)
    return logits


