from jaxtyping import Float, Int
import torch as t
from torch import Tensor
import os
import time
from transformer_lens import HookedTransformer, utils
import numpy as np
import json

def save_parameter(
    args,
):
    str_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_path = f'outputs/{args.model_name}/{args.experiment_name}/{args.task_name}_{str_time}'
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    with open(f'{log_path}/parameters.txt', 'a+') as file:
        for key, value in vars(args).items():
            file.write(f"{key}: {value}\n")
    return log_path


def batch_load_data(data_list, batch_size):
    """
    Yields batches of dicts from data_list.
    
    :param data_list: list of dictionaries
    :param batch_size: size of each batch
    """
    for i in range(0, len(data_list), batch_size):
        yield data_list[i : i + batch_size]



def eval_batch(
    prompt_seqs,
    completions,
    target_completions,
    log_path = '',
) -> float:
    num_correct:Float = 0.0
    if log_path == '':
        for i in range( len(target_completions) ):
            if completions[i] == target_completions[i]:
                num_correct +=1
    else:
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        completions_file = f'{log_path}/seq.txt'

        for i in range( len(target_completions) ):
            seq = prompt_seqs[i]

            if completions[i] == target_completions[i]:
                num_correct +=1
            else:
                with open(completions_file,"a+") as file:
                    file.write(f'##Wrong: ')
                    file.close()

            with open(completions_file,"a+") as file:
                file.write(f'{seq}{completions[i]}\n')
                file.write(f'Correct completion: {target_completions[i]}\n\n')
                file.close()

    return num_correct


def get_label_inverted_rank(prob, target):
    """
    Returns the inverted rank (1 / rank) for each item in the batch.
    
    Rank is determined by sorting each row in descending order by probability:
        - rank = 1 for the top label
        - rank = d_vocab for the lowest label

    Args:
        prob (torch.Tensor): A [batch, d_vocab] tensor of probabilities or logits.
        target (torch.Tensor): A [batch] tensor of target indices.

    Returns:
        torch.Tensor: A [batch] tensor containing 1/rank of the target label.
    """
    sorted_indices = prob.argsort(dim=1, descending=True)
    ranks = (sorted_indices == target.unsqueeze(1)).nonzero()[:, 1] + 1
    inverted_rank = 1.0 / ranks.float()
    
    return inverted_rank

def eval_subtask(
    test_model: HookedTransformer,
    prob: Float[Tensor, "batch d_vocab"],
    batch_dataset: dict,
    layer: int,
    metric: str = 'accuracy' or 'probability' or 'calibrated_prob' or 'ranking',
    log_path: str = '',
) -> Float[Tensor, "3"]:
    target_completion_ids = test_model.to_tokens(batch_dataset['completion'],
                                            prepend_bos=False, padding_side='right')[:,0]
    target_completions = test_model.to_str_tokens(target_completion_ids)
    s1_completion_ids = test_model.to_tokens(batch_dataset['completion_t1'],
                                            prepend_bos=False, padding_side='right')[:,0]
    s1_completions = test_model.to_str_tokens(s1_completion_ids)
    s2_completion_ids = test_model.to_tokens(batch_dataset['completion_t2'],
                                            prepend_bos=False, padding_side='right')[:,0]
    s2_completions = test_model.to_str_tokens(s2_completion_ids)    

    eval_result = t.zeros((3), device= test_model.cfg.device)

    if metric == 'accuracy':
        eval_result[0] = eval_batch(batch_dataset['prompt'], test_model.to_str_tokens(prob.argmax(dim = -1)), target_completions, '' if log_path=='' else f'{log_path}/base/layer{layer}')
        eval_result[1] = eval_batch(batch_dataset['prompt'], test_model.to_str_tokens(prob.argmax(dim = -1)), s1_completions, '' if log_path=='' else f'{log_path}/s1/layer{layer}')
        eval_result[2] = eval_batch(batch_dataset['prompt'], test_model.to_str_tokens(prob.argmax(dim = -1)), s2_completions, '' if log_path=='' else f'{log_path}/s2/layer{layer}')
    elif metric == 'probability':
        eval_result[0] = prob[t.arange(len(target_completions)), target_completion_ids].sum().item()
        eval_result[1] = prob[t.arange(len(target_completions)), s1_completion_ids].sum().item()
        eval_result[2] = prob[t.arange(len(target_completions)), s2_completion_ids].sum().item()
    elif metric == 'calibrated_prob':
        target_prob = prob[t.arange(len(target_completions)), target_completion_ids].clone()
        s1_prob = prob[t.arange(len(target_completions)), s1_completion_ids].clone()
        s2_prob = prob[t.arange(len(target_completions)), s2_completion_ids].clone()

        stacked_prob = t.stack([target_prob, s1_prob, s2_prob], dim=1)
        stacked_prob = t.softmax(stacked_prob, dim=1)

        eval_result[0] = stacked_prob[:, 0].sum().item()
        eval_result[1] = stacked_prob[:, 1].sum().item()
        eval_result[2] = stacked_prob[:, 2].sum().item()
    elif metric == 'ranking':
        eval_result[0] = get_label_inverted_rank(prob, target_completion_ids).sum().item()
        eval_result[1] = get_label_inverted_rank(prob, s1_completion_ids).sum().item()
        eval_result[2] = get_label_inverted_rank(prob, s2_completion_ids).sum().item()

    return eval_result


def get_masking_range(
    test_model: HookedTransformer,
    batch_dataset: dict,
) -> tuple[list, list]:
    if batch_dataset["zero_prompt"][0] == 'none':
        raise ValueError("No zero prompt!")
    
    if "Qwen" in test_model.cfg.model_name:
        starts = [0] * len(batch_dataset['prompt'])
        ends = [0] * len(batch_dataset['prompt'])
    else:
        starts = [1] * len(batch_dataset['prompt'])
        ends = [1] * len(batch_dataset['prompt'])

    for i in range(len(batch_dataset['prompt'])):
        single_prompt_token = test_model.to_tokens(batch_dataset['prompt'][i])
        zero_prompt_token = test_model.to_tokens(batch_dataset['zero_prompt'][i])
        ends[i] = ends[i] + (single_prompt_token.shape[1] - zero_prompt_token.shape[1])

    batch_prompt_tokens = test_model.to_tokens(batch_dataset['prompt'])
    for i in range(len(batch_dataset['prompt'])):
            single_prompt_token = test_model.to_tokens(batch_dataset['prompt'][i])
            bias = batch_prompt_tokens.shape[1] - single_prompt_token.shape[1]
            starts[i] = starts[i] + bias
            ends[i] = ends[i] + bias

    return starts, ends

