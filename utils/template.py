from transformer_lens import HookedTransformer

LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM = """<|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

QWEN2_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""



EVALUATION_TEMPLATE = """[System]
You are a fair judge, and please evaluate the quality of an AI assistant’s responses to user query. You need to assess the response based on the following constraints. We will provide you with the user’s query, some constraints, and the AI assistant’s response that needs your evaluation. When you commence your evaluation, you should follow the following process:
1. Evaluate each constraint: Assess how well the AI assistant’s response meets each individual constraint.
2. Assign a score (0–10) for each constraint: After explaining your assessment for each constraint, give a corresponding score from 0 (does not meet the requirement at all) to 10 (fully meets the requirement).
3. List the scores: List the Constraints Overall Score (as a list of the individual scores in their original constraint order).
4. Strict scoring policy: Be as strict as possible in assigning scores. If the response is irrelevant, contains major factual errors, or generates harmful content, the “Constraints Overall Score” must be 0.
5. Preserve constraint order: When you provide the “Fine Grained Score,” the constraints must appear in the same order as they are listed in the input context.
6. Follow the output format: After you provide explanations for each constraint, list the Fine Grained Score in JSON format and the Constraints Overall Score as a list, as shown in the example below.
Please reference and follow the format demonstrated in the /* Example */.
/* Example */
—INPUT—
#Task Description:
Create a password for this account
#Constraints:
The password must be at least 8 characters long;
It must contain 1 uppercase letter;
It must contain 1 lowercase letter;
It must include 2 numbers;
#Input:
NULL
#Response:
Ax7y4gTf
—OUTPUT—
Explanation:
Password Length: The password “Ax7y4gTf” is 8 characters long, meeting the first constraint, scoring 10 points.
Contains 1 uppercase letter: The password “Ax7y4gTf” contains two uppercase letters, “A” and “T”, which means it meets the second constraint, but the explanation incorrectly states it does not meet the constraint, scoring 0 points.
Contains 1 lowercase letter: The password “Ax7y4gTf” contains three lowercase letters, “x”, “y”, and “g”, which means it meets the third constraint, but the explanation incorrectly states it does not meet the constraint, scoring 0 points.
Includes 2 numbers: The password “Ax7y4gTf” includes two numbers, “7” and “4”, meeting the fourth constraint, scoring 10 points.
Fine Grained Score: [ { "The password must be at least 8 characters long": 10, "It must contain 1 uppercase letter": 0, "It must contain 1 lowercase letter": 0, "It must include 2 numbers": 10 } ]
Constraints Overall Score: [10, 0, 0, 10]
/* Input */
—INPUT—
#Task Description:
{task_description}
#Constraints:
{constraint}
#Input:
{input}
<response>:
{ans}
—OUTPUT—
"""



EVALUATION_EXAMPLE = """[System]
You are a fair judge, and please evaluate the quality of an AI assistant’s responses to user query. You need to assess the response based on the following constraints. We will provide you with the user’s query, some constraints, and the AI assistant’s response that needs your evaluation. When you commence your evaluation, you should follow the following process:
1. Evaluate each constraint: Assess how well the AI assistant’s response meets each individual constraint.
2. Assign a score (0–10) for each constraint: After explaining your assessment for each constraint, give a corresponding score from 0 (does not meet the requirement at all) to 10 (fully meets the requirement).
3. List the scores: List the Constraints Overall Score (as a list of the individual scores in their original constraint order).
4. Strict scoring policy: Be as strict as possible in assigning scores. If the response is irrelevant, contains major factual errors, or generates harmful content, the “Constraints Overall Score” must be 0.
5. Preserve constraint order: When you provide the “Fine Grained Score,” the constraints must appear in the same order as they are listed in the input context.
6. Follow the output format: After you provide explanations for each constraint, list the Fine Grained Score in JSON format and the Constraints Overall Score as a list, as shown in the example below.
Please reference and follow the format demonstrated in the /* Example */.
/* Example */
—INPUT—
#Task Description:
Create a password for this account
#Constraints:
The password must be at least 8 characters long;
It must contain 1 uppercase letter;
It must contain 1 lowercase letter;
It must include 2 numbers;
#Input:
NULL
#Response:
Ax7y4gTf
—OUTPUT—
Explanation:
Password Length: The password “Ax7y4gTf” is 8 characters long, meeting the first constraint, scoring 10 points.
Contains 1 uppercase letter: The password “Ax7y4gTf” contains two uppercase letters, “A” and “T”, which means it meets the second constraint, but the explanation incorrectly states it does not meet the constraint, scoring 0 points.
Contains 1 lowercase letter: The password “Ax7y4gTf” contains three lowercase letters, “x”, “y”, and “g”, which means it meets the third constraint, but the explanation incorrectly states it does not meet the constraint, scoring 0 points.
Includes 2 numbers: The password “Ax7y4gTf” includes two numbers, “7” and “4”, meeting the fourth constraint, scoring 10 points.
Fine Grained Score: [ { "The password must be at least 8 characters long": 10, "It must contain 1 uppercase letter": 0, "It must contain 1 lowercase letter": 0, "It must include 2 numbers": 10 } ]
Constraints Overall Score: [10, 0, 0, 10]"""



def generate_prompt_with_template(
    model: HookedTransformer,
    batch_data: list[dict],
    masking_range: str= 'all' or 'constraints'
) -> tuple[list[str], list[int], list[int]]:
    prompt_list = []
    start_list = []
    end_list = []
    if "Llama" in model.cfg.model_name:
        for data in batch_data:
            prompt = LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM.format(system_prompt= data["messages"][0]["content"], instruction= data["messages"][1]["content"])
            description = data["description"]
            constraints = data["constraints"]
            description_token = model.to_tokens(description, prepend_bos=False)
            constraints_token = model.to_tokens(constraints, prepend_bos=False)
            if masking_range == 'constraints':
                start = 16 + description_token.shape[1]
                end = start + constraints_token.shape[1]
            else:
                start = 16 
                end = start + description_token.shape[1] + constraints_token.shape[1]

            prompt_list.append(prompt)
            start_list.append(start)
            end_list.append(end)

    elif "Qwen" in model.cfg.model_name:
        for data in batch_data:
            prompt = QWEN2_CHAT_TEMPLATE_WITH_SYSTEM.format(system_prompt= data["messages"][0]["content"], instruction= data["messages"][1]["content"])
            description = data["description"]
            constraints = data["constraints"]
            description_token = model.to_tokens(description, prepend_bos=False)
            constraints_token = model.to_tokens(constraints, prepend_bos=False)
            if masking_range == 'constraints':
                start = 14 + description_token.shape[1]
                end = start + constraints_token.shape[1]
            else:
                start = 14
                end = start + description_token.shape[1] + constraints_token.shape[1]

            prompt_list.append(prompt)
            start_list.append(start)
            end_list.append(end)


    batch_prompt_token = model.to_tokens(prompt_list)
    for i in range(len(prompt_list)):
        prompt_token = model.to_tokens(prompt_list[i])
        bias = batch_prompt_token.shape[1] - prompt_token.shape[1]
        start_list[i] = start_list[i] + bias
        end_list[i] = end_list[i] + bias

    return prompt_list, start_list, end_list


def generate_eval_prompt(
    data,
    response,
) -> str:
    prompt = EVALUATION_EXAMPLE+"\n/* Input */\n—INPUT—\n#Task Description:\n"+data["description"]+"\n#Constraints:\n"+data["constraints"]+"\n#Input:\n"+data["input"]+"\n<response>:\n"+response+"\n—OUTPUT—\n"

    return prompt