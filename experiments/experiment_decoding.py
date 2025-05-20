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

from src.data.dataset import load_data, ICLDataset
from src.decoding import save_activation_and_scale, activation_stack_to_prob
from utils.tools import save_parameter, eval_subtask
from utils.display import draw_line

from huggingface_hub import login
login('Your Access Tokens')

def main(args, test_model: HookedTransformer, device):
    log_path = save_parameter(args)

    # Bulid dataset
    data = load_data(f'data/composite/{args.task_name}.json')
    dataset = ICLDataset(data, size=args.data_num, n_prepended=args.n_prepended, seed=args.seed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.decoding_type == 'both':
        acc_result = t.zeros((2 * test_model.cfg.n_layers, 3), device= device)
        ranking_result = t.zeros((2 * test_model.cfg.n_layers, 3), device= device)
    else:
        acc_result = t.zeros((test_model.cfg.n_layers, 3), device= device)
        ranking_result = t.zeros((test_model.cfg.n_layers, 3), device= device)

    pbar = tqdm(dataloader)
    for batch in pbar:   
        activation, scale = save_activation_and_scale(test_model, batch['prompt'], args.decoding_type, -1)
        stacked_prob = activation_stack_to_prob(test_model, activation, scale)

        for layer in tqdm(range(acc_result.shape[0])):
            acc_result[layer, :] += eval_subtask(test_model, stacked_prob[layer,], batch, layer, 'accuracy', '')
            ranking_result[layer, :] += eval_subtask(test_model, stacked_prob[layer,], batch, layer, 'ranking', '')

        max_memory = t.cuda.max_memory_allocated()
        pbar.set_postfix(layer=f"{layer+1}/{test_model.cfg.n_layers}", memory=f"{max_memory/(1024**3):.2f} GB")

    acc_result /= args.data_num
    ranking_result /= args.data_num

    plot_data = [
        [ranking_result[:, 0].tolist(), 'Target'],
        [ranking_result[:, 1].tolist(), 's1'],
        [ranking_result[:, 2].tolist(), 's2'],
    ]
    draw_line(plot_data, 'Layer', log_path, 'Decoding_Ranking', 'Ranking', hide_ticks=True)
    with open(f'{log_path}/ranking_results.json', 'w') as outfile:
        json.dump(ranking_result.tolist(), outfile)

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data_num', default=500, type=int)
    parser.add_argument('--n_prepended', default=5, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--decoding_type', default='resid_post', type=str)

    parser.add_argument('--task_name', default='', type=str)
    parser.add_argument('--experiment_name', default='experiment_decoding', type=str)
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

        for i in range(5):
            args.seed = i
            main(args, test_model, device)



