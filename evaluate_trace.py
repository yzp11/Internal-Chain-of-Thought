from src.data.dataset import load_data_with_instructions
from utils.template import generate_eval_prompt
import json
import os
from openai import OpenAI


def extract_response(
    result: str,
    start_tag: str,
    end_tag: str
) -> str:
    result = result + end_tag
    response = result.split(start_tag)[-1].split(end_tag)[0]
    
    return response

def api_evaluation(
    prompt: str
) -> str:
    key = "Your Key"

    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    constraints_type = ["包含约束", "输出格式约束", "语气风格约束"]

    data_list = load_data_with_instructions('data/2024_trace_evaluation.jsonl', constraints_type)

    start_tag = "<|im_start|>assistant\n"
    end_tag   = "<|im_end|>"
    n_layer = 28

    if not os.path.isdir("outputs/trace_evaluation"):
        os.makedirs("outputs/trace_evaluation")

    for layer in range(n_layer):
        response_list = []
        with open(f"outputs/trace/layer{layer}.jsonl", "r", encoding="utf-8") as file:
            for line in file:
                json_obj = json.loads(line)
                response_list.append(extract_response(json_obj, start_tag, end_tag))

        for i in range(len(response_list)):
            prompt = generate_eval_prompt(data_list[i], response_list[i])
            result = api_evaluation(prompt)

            with open(f"outputs/trace_evaluation/layer{layer}.jsonl", "a+", encoding="utf-8") as file:
                json_line = json.dumps(result, ensure_ascii=False)
                file.write(json_line + "\n")
                file.close()