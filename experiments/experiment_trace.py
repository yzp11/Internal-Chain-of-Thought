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
from transformer_lens import HookedTransformer

from src.data.dataset import load_data_with_instructions
from src.masking import mask_and_generate
from utils.tools import save_parameter, batch_load_data
from utils.template import generate_prompt_with_template

from huggingface_hub import login
login('Your Access Tokens')

def main(args, test_model: HookedTransformer, device, constraints_type):
    log_path = save_parameter(args)

    # Load Data
    data_list = load_data_with_instructions("data/2024_trace_evaluation.jsonl", constraints_type)

    for batch in tqdm(batch_load_data(data_list, args.batch_size)):
        prompts, starts, ends = generate_prompt_with_template(test_model, batch, args.masking_range)

        pbar = tqdm(range(test_model.cfg.n_layers))
        for layer in pbar:
            responses = mask_and_generate(test_model, prompts, args.max_tokens, layer, starts, ends, args.masking_type)
            with open(f"{log_path}/layer{layer}.jsonl", "a+", encoding="utf-8") as file:
                for re in responses:
                    json_line = json.dumps(re, ensure_ascii=False)
                    file.write(json_line + "\n")
                file.close()
            max_memory = t.cuda.max_memory_allocated()
            pbar.set_postfix(layer=f"{layer+1}/{test_model.cfg.n_layers}", memory=f"{max_memory/(1024**3):.2f} GB")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--masking_type', default='attn', type=str)
    parser.add_argument('--masking_range', default='constraints', type=str)
    parser.add_argument('--max_tokens', default=1500, type=int)
    parser.add_argument('--batch_size', default=18, type=int)


    parser.add_argument('--task_name', default='trace', type=str)
    parser.add_argument('--experiment_name', default='experiment_trace', type=str)
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-7B-Instruct', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()
    device = t.device(args.device)
    constraints_type = ["包含约束", "输出格式约束", "语气风格约束"]

    #Load model
    t.set_grad_enabled(False)
    test_model: HookedTransformer = HookedTransformer.from_pretrained(model_name= args.model_name, device= device, default_padding_side= 'left')

    main(args, test_model, device, constraints_type)


