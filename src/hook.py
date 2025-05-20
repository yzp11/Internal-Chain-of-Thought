from jaxtyping import Float, Int
import torch as t
from torch import Tensor
import einops
from transformer_lens.hook_points import HookPoint


def save_one_pos_hook(
    act: Float[Tensor, 'batch seq ...'],
    hook: HookPoint,
    save_list: list,
    pos_ids: int,
) -> Float[Tensor, 'batch ...']:
    save_list.append(act[:, pos_ids,].detach().clone())
    return act


def act_patching_hook(
    act: Float[Tensor, 'batch seq ...'],
    hook: HookPoint,
    tar_act: Float[Tensor, '...'],
    replace: bool,
    pos_ids: int,
) -> Tensor:
    if not replace:
        act[:, pos_ids,] += tar_act.detach().clone()
    else:
        act[:, pos_ids,] = tar_act.detach().clone()
    return act



def resid_masking_hook(
    act: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    starts: list[int],
    positions: list[int],  
) -> Tensor:

    batch_size, seq_len, d_model = act.shape
    
    for i in range(batch_size):
        sta = starts[i]
        pos = positions[i]

        mean_val = act[i, sta:pos, :].mean()  
        act[i, sta:pos, :] = mean_val
    
    return act



def attention_masking_hook(
    attn_scores: Float[Tensor, "batch n_head seq seq"],
    hook: HookPoint,
    starts: list[int],
    positions: list[int],
) -> Tensor:
    block_mask = t.zeros_like(attn_scores)
    batch_size, n_head, seq_len, _ = attn_scores.shape
    
    for i in range(batch_size):
        sta = starts[i]
        pos = positions[i]
        block_mask[i, :, :, sta:pos] = float('-inf')
    
    attn_scores += block_mask
    return attn_scores


def pattern_masking_hook(
    attn_pattern: Float[Tensor, "batch n_head seq seq"],
    hook: HookPoint,
    starts: list[int],
    positions: list[int],
) -> Tensor:
    batch_size, n_head, seq_len, _ = attn_pattern.shape
    
    for i in range(batch_size):
        sta = starts[i]
        pos = positions[i]
        attn_pattern[i, :, :, sta:pos] = 0.0
    
    return attn_pattern



