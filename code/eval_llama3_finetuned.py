from unsloth import FastLanguageModel
import torch
import ast
import json
import jsonlines
import os
import re
import glob
from python_tool import PythonInterpreter
python = PythonInterpreter(globals=globals(), locals=None)


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = torch.bfloat16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
model_name_path = 'gee-llama3-8B'

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="gee-llama3-8B",  # YOUR MODEL YOU USED FOR TRAINING
    cache_dir=model_name_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
# alpaca_prompt = You MUST copy from above!
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]



def extract_code_full(text):
    #print(repr(text))
    if "```python\n" in text:
        pattern = r"(?:%s(?:.*?)%s)" % (r'```python\n', r'\n```')  # included     r"%s(.?)%s"%('# ---\n# jupyter','# ---') # excluded
        cut_content = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
    elif "```Python\n" in text:
        pattern = r"(?:%s(?:.*?)%s)" % (r'```Python\n', r'\n```')  # included     r"%s(.?)%s"%('# ---\n# jupyter','# ---') # excluded
        cut_content = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
    elif "``` corrected.\n" in text:
        pattern = r"(?:%s(?:.*?)%s)" % ('``` corrected.\n', '\n```')  # included     r"%s(.?)%s"%('# ---\n# jupyter','# ---') # excluded
        cut_content = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
    elif "```\n" in text:
        pattern = r"(?:%s(?:.*?)%s)" % ('```\n', '\n```')  # included     r"%s(.?)%s"%('# ---\n# jupyter','# ---') # excluded
        cut_content = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
    else:
        cut_content = []
    if len(cut_content) > 0:
        #print('-------------extracted here------------------')
        #print(cut_content[0])
        lines = cut_content[0].split('\n')
        lines.pop(0)
        lines.pop(-1)
        text = '\n'.join(lines)
    return text


def extract_code_split(text):
    new_text = []
    add_flag = False
    if '```' in text:
        for text_i in text.splitlines(keepends=True):
            if add_flag and '```' not in text_i:
                new_text.append(text_i)
            if '```' in text_i:
                add_flag = not add_flag
        text = ''.join(new_text)

    return text


# get data input
in_path = '/workspace/comparison_bench/cibench'
out_path = '/workspace/comparison_bench/cibench_out_llama31'
path = in_path
output = out_path
filelist = glob.glob(os.path.join(path, '*.json'))
output_list = glob.glob(os.path.join(output, '*.json'))
output_nm_list = [nm.split('/')[-1].split('.')[0] for nm in output_list]
print('There are ' + str(len(filelist)) + ' python scripts!')
print('There are ' + str(len(output_list)) + ' python scripts!')


# processing data
for pyscript in filelist:
    each_nm = pyscript.split('/')[-1].split('.')[0]
    if each_nm in output_nm_list:
        print('skip !!!!!!!!!!!!!!!!!', each_nm)
        continue
    with open(pyscript, "r") as file:
        new_code_dict = {}
        content = json.load(file)

    pre_context = ''
    num = len(content) - 1
    dict_keys = list(content.keys())
    dict_keys = sorted(dict_keys, key=int)
    for ii in dict_keys:
        comment_i = content[str(ii)]['prompt']
        library_i = content[str(ii)]['exlib']


        if library_i:
            messages = [
            {"role": "system",
                "content": f"I want you to become my Expert programmer. Your goal is to help me write python code for the given task using python library{library_i}"},
                {"role": "user",
                 "content": f"I first give you the previous task with code: {pre_context}" + "Then, you need to write code according to the detailed prompt this step:" + f'{comment_i}' + "\n Give the corresponding code:"}, ]
        else:
            messages = [
                {"role": "system",
                "content": "I want you to become my Expert programmer. Your goal is to help me write python code for the given task"},
                {"role": "user",
                 "content": f"I first give you the previous task with code: {pre_context}" + "Then, you need to write code according to the detailed prompt this step:" + f'{comment_i}' + "\n Give the corresponding code:"}, ]


        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            model.device)
        outputs = model.generate(input_ids, max_new_tokens=256, eos_token_id=terminators, do_sample=True,
                                 temperature=0.6, top_p=0.9, )
        response = outputs[0][input_ids.shape[-1]:]
        code_i = tokenizer.decode(response, skip_special_tokens=True)
        #code_i = extract_code_full(code_i)
        code_i = extract_code_split(code_i)
        print(code_i)
        # update previous context with code
        pre_context = pre_context + '\n#' + comment_i.replace('\n', '\n#')
        pre_context = pre_context + '\n' + code_i


        new_code_dict[str(ii)] = {'task': comment_i, 'code': code_i}


    # Convert and write JSON object to file
    with open(pyscript.replace(in_path,out_path), "w") as outfile:
      json.dump(new_code_dict, outfile)

