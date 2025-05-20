import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import json

dir = "experiments"
main_dir = Path(f"{os.getcwd().split(dir)[0]}").resolve()
if str(main_dir) not in sys.path: sys.path.append(str(main_dir))

import torch as t
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from src.data.dataset import ICLDataset
from src.data.split_data import split_and_generate
from src.patching import save_activation, patch_activation, base_run
from utils.tools import save_parameter, eval_subtask

from huggingface_hub import login
login('Your Access Tokens')

def main(args, test_model: HookedTransformer, device):
    log_path = save_parameter(args)

    s1_data, s2_data, composite_data = split_and_generate(args.subtask1_name, args.subtask2_name, args.seed)

    dataset = ICLDataset(composite_data, size=args.train_data_num, n_prepended=args.n_prepended, seed=args.seed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    task_act = t.zeros((test_model.cfg.n_layers, test_model.cfg.d_model), device= device)
    for batch in tqdm(dataloader):
        task_act += save_activation(test_model, batch['prompt'], args.patching_mode, args.pos_ids)
    task_act = task_act/args.train_data_num

    base_result = t.zeros((2, test_model.cfg.n_layers), device= device)
    corrupted_result = t.zeros((2, test_model.cfg.n_layers), device= device)
    patching_result = t.zeros((2, test_model.cfg.n_layers), device= device)

    dataset = ICLDataset(s1_data, args.data_num, args.n_prepended, args.seed, generate_zero= True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for batch in tqdm(dataloader):   
        base_prob = base_run(test_model, batch['prompt'])
        corrupted_prob = base_run(test_model, batch['zero_prompt'])
        base_result[0, :] += eval_subtask(test_model, base_prob, batch, -1, 'accuracy')[0].item()
        corrupted_result[0, :] += eval_subtask(test_model, corrupted_prob, batch, -1, 'accuracy')[0].item()

        pbar = tqdm(range(test_model.cfg.n_layers))
        for layer in pbar:
            patching_prob = patch_activation(test_model, batch['zero_prompt'], task_act[layer], layer, args.patching_mode, args.replace, args.pos_ids)
            patching_result[0, layer] += eval_subtask(test_model, patching_prob, batch, layer, 'accuracy')[0].item()
            max_memory = t.cuda.max_memory_allocated()
            pbar.set_postfix(layer=f"{layer+1}/{test_model.cfg.n_layers}", memory=f"{max_memory/(1024**3):.2f} GB")

    dataset = ICLDataset(s2_data, args.data_num, args.n_prepended, args.seed, generate_zero= True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for batch in tqdm(dataloader):   
        base_prob = base_run(test_model, batch['prompt'])
        corrupted_prob = base_run(test_model, batch['zero_prompt'])
        base_result[1, :] += eval_subtask(test_model, base_prob, batch, -1, 'accuracy')[0].item()
        corrupted_result[1, :] += eval_subtask(test_model, corrupted_prob, batch, -1, 'accuracy')[0].item()

        pbar = tqdm(range(test_model.cfg.n_layers))
        for layer in pbar:
            patching_prob = patch_activation(test_model, batch['zero_prompt'], task_act[layer], layer, args.patching_mode, args.replace, args.pos_ids)
            patching_result[1, layer] += eval_subtask(test_model, patching_prob, batch, layer, 'accuracy')[0].item()
            max_memory = t.cuda.max_memory_allocated()
            pbar.set_postfix(layer=f"{layer+1}/{test_model.cfg.n_layers}", memory=f"{max_memory/(1024**3):.2f} GB")

    patching_result = (patching_result-corrupted_result) / (base_result-corrupted_result)
    
    with open(f'{log_path}/strength_results.json', 'w') as outfile:
        json.dump(patching_result.tolist(), outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--train_data_num', default=100, type=int)
    parser.add_argument('--data_num', default=500, type=int)
    parser.add_argument('--n_prepended', default=5, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--replace', default=True, type=bool)
    parser.add_argument('--pos_ids', default=-1, type=int)
    parser.add_argument('--patching_mode', default='resid_post', type=str)

    parser.add_argument('--task_name', default='', type=str)
    parser.add_argument('--subtask1_name', default='', type=str)
    parser.add_argument('--subtask2_name', default='', type=str)
    parser.add_argument('--experiment_name', default='experiment_patching', type=str)
    parser.add_argument('--model_name', default='meta-llama/Llama-3.1-8B', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()
    device = t.device(args.device)

    #Load model
    t.set_grad_enabled(False)
    test_model: HookedTransformer = HookedTransformer.from_pretrained(model_name= args.model_name, device= device, default_padding_side= 'left')

    with open("data/list.json", "r") as f:
        data = json.load(f)
    for d in data:  
        args.task_name = f"{d['task1']}-{d['task2']}"
        args.subtask1_name = d['task1']
        args.subtask2_name = d['task2']

        for i in range(5):
            args.seed = i
            main(args, test_model, device)



