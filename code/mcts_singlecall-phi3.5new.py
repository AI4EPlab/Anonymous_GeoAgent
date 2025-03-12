# RAG
import nest_asyncio
'''
nest_asyncio.apply()
import logging
import sys
from llama_index.core import SimpleDirectoryReader
from llama_index.core import SummaryIndex

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import IndexNode
from llama_index.core.retrievers import QueryFusionRetriever

# llm
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context
from llama_index.llms.huggingface import HuggingFaceLLM
import torch

system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# loads BAAI/bge-small-en-v1.5
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

service_context = ServiceContext.from_defaults(
    chunk_size=2048,
    llm=None,  # llm,
    embed_model=embed_model
)
set_global_service_context(service_context)

# this is very omportant and avoid the error in graph index
from llama_index.core import Settings

Settings.llm = None  # llm
Settings.chunk_size = 2048
# maximum input size to the LLM
Settings.context_window = 4096
# number of tokens reserved for text generation.
Settings.num_output = 256
'''
import random
from typing import Tuple, Any
import numpy as np
from visualise import render_graphviz_tree
from math import exp, log, inf, sqrt
import time, sys
import itertools, copy
from unsloth import FastLanguageModel
import torch
import torch.nn.functional as F
import ast
import json
import jsonlines
import os
import re
import glob
import jedi
from multi_process import monitor_process, make_serializable, is_serializable
# 设置监控的最大执行时间和内存使用百分比
MAX_EXECUTION_TIME = 10  # 秒
MAX_MEMORY_USAGE = 90  # 百分比
from humaneval import stats_execute, get_prompts_with_ids, STOP_SEQUENCES
from human_eval.data import write_jsonl

os.environ["TOKENIZERS_PARALLELISM"] = "false"
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = torch.bfloat16  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.
#model_name_path = '/workspace/unsloth-llama-3.1-8b-Instruct'
#model_name = '/workspace/unsloth-llama-3.1-8b-Instruct'
#model_name = 'unsloth/llama-3-8b-Instruct'  # 'unsloth/mistral-7b-v0.2' #
#model_name_path = '/workspace/unsloth-llama-3-8b-Instruct'

#model_name_path = '/workspace/codegemma-7b'
#model_name = 'unsloth/codegemma-7b'

#model_name_path = '/workspace/Qwen2-7B-Instruct'
#model_name = 'unsloth/Qwen2-7B-Instruct'

# uninstall unsloth and reinstall everythings
model_name_path = '/workspace/Phi-3.5-mini-instruct'
model_name = 'unsloth/Phi-3.5-mini-instruct'

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    cache_dir=model_name_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    rope_scaling='null',
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|end|>"),
]

tokenizer = get_chat_template(
    tokenizer,
    chat_template="phi-3",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping={"role": "role", "content": "content", "user": "human", "assistant": "gpt"},  # ShareGPT style
    map_eos_token=True,  # Maps <|im_end|> to </s> instead
)


def extract_code_split(text, pre_code):
    if '```' in text:
        new_text = []
        add_flag = False
        for text_i in text.splitlines(keepends=True):
            if add_flag and '```' not in text_i:
                new_text.append(text_i)
            if '```' in text_i:
                add_flag = not add_flag
        text = ''.join(new_text)
        # get ast code block
        code_block = ''
        # split code with '.' and '\n'
        multi_text = text.splitlines()
        multi_text = [text for text in multi_text if not text.startswith('#')]
        if multi_text:
            for i in range(len(multi_text)):
                try:
                    code_block = '\n'.join(multi_text)
                    ast.parse(code_block)
                    break
                except:
                    multi_text.pop()
                    code_block = '\n'.join(multi_text)
                    continue
    else:
        # get ast code block
        code_block = []
        one_block = []
        start_id = 0
        # split code with '.' and '\n'
        multi_text = text.splitlines()
        if '' in multi_text:
            multi_text.remove('')
        multi_text.append('end')

        if multi_text:
            for i, text_i in enumerate(multi_text):
                start_id += 1
                if start_id > 1:
                    if text_i.startswith(' '):
                        one_block.append(text_i)
                    else:
                        try:
                            one_block_ = '\n'.join(one_block) + '\n' + text_i
                            code_block.append('\n'.join(one_block))
                            one_block = []
                            ast.parse('\n'.join(code_block))
                            pop_one_block = False
                        except:
                            code_block.pop()
                            try:
                                code_block.append(one_block_)
                                ast.parse('\n'.join(code_block))
                                pop_one_block = True
                            except:
                                code_block.pop()
                                pop_one_block = False
                                pass
                        # test current line
                        try:
                            code_block.append(text_i)
                            one_block = []
                            ast.parse('\n'.join(code_block))
                            start_id = 0
                        except:
                            code_block.pop()
                            if not pop_one_block:
                                one_block.append(text_i)
                            continue
                else:
                    one_block.append(text_i)

        if 'end' in code_block:
            code_block.remove('end')
        code_block = '\n'.join(code_block)

    #print('----------------------------------------------------------------')
    #print(code_block)
    #print('----------------------------------------------------------------')
    if pre_code not in code_block:
        code_block = pre_code + '\n' + code_block

    if code_block:
        new_context = ''
        unique_lines = []
        tree = ast.parse(code_block)
        line_to_nodes = sort_nodes_by_line(tree)
        for linen_key, line_values in line_to_nodes.items():
            line = max([ast.unparse(line_value) for line_value in line_values], key=len)
            if not is_string_contained(line, new_context):
                new_context = new_context + line
                unique_lines.append(line + '\n')

        selected_code_block = ''.join(unique_lines)

        error = None
    else:
        selected_code_block = ''
        error = 'Write right python code please!'
    return selected_code_block, error


def extract_code_again(text):
    if '```' in text:
        new_text = []
        add_flag = False
        for text_i in text.splitlines(keepends=True):
            if add_flag and '```' not in text_i:
                new_text.append(text_i)
            if '```' in text_i:
                add_flag = not add_flag
        text = ''.join(new_text)
        # get ast code block
        code_block = ''
        # split code with '.' and '\n'
        multi_text = text.splitlines()
        multi_text = [text for text in multi_text if not text.startswith('#')]
        if multi_text:
            for i in range(len(multi_text)):
                try:
                    code_block = '\n'.join(multi_text)
                    ast.parse(code_block)
                    break
                except:
                    multi_text.pop()
                    code_block = '\n'.join(multi_text)
                    continue
    else:
        # get ast code block
        code_block = []
        one_block = []
        start_id = 0
        # split code with '.' and '\n'
        multi_text = text.splitlines()
        if '' in multi_text:
            multi_text.remove('')
        multi_text.append('end')

        if multi_text:
            for i, text_i in enumerate(multi_text):
                start_id += 1
                if start_id > 1:
                    if text_i.startswith(' '):
                        one_block.append(text_i)
                    else:
                        try:
                            one_block_ = '\n'.join(one_block) + '\n' + text_i
                            code_block.append('\n'.join(one_block))
                            one_block = []
                            ast.parse('\n'.join(code_block))
                            pop_one_block = False
                        except:
                            code_block.pop()
                            try:
                                code_block.append(one_block_)
                                ast.parse('\n'.join(code_block))
                                pop_one_block = True
                            except:
                                code_block.pop()
                                pop_one_block = False
                                pass
                        # test current line
                        try:
                            code_block.append(text_i)
                            one_block = []
                            ast.parse('\n'.join(code_block))
                            start_id = 0
                        except:
                            code_block.pop()
                            if not pop_one_block:
                                one_block.append(text_i)
                            continue
                else:
                    one_block.append(text_i)

        if 'end' in code_block:
            code_block.remove('end')
        code_block = '\n'.join(code_block)

    if code_block:
        new_context = ''
        unique_lines = []
        tree = ast.parse(code_block)
        line_to_nodes = sort_nodes_by_line(tree)
        for linen_key, line_values in line_to_nodes.items():
            line = max([ast.unparse(line_value) for line_value in line_values], key=len)
            if not is_string_contained(line, new_context):
                new_context = new_context + line
                unique_lines.append(line + '\n')

        selected_code_block = ''.join(unique_lines)
    else:
        selected_code_block = ''
    return selected_code_block


def sort_nodes_by_line(node, line_to_nodes=None):
    """
    按照代码行的顺序对AST节点进行排列
    """
    if line_to_nodes is None:
        line_to_nodes = {}

    if hasattr(node, 'lineno'):
        line = node.lineno
        if line not in line_to_nodes:
            line_to_nodes[line] = []
        line_to_nodes[line].append(node)

    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    sort_nodes_by_line(item, line_to_nodes)
        elif isinstance(value, ast.AST):
            sort_nodes_by_line(value, line_to_nodes)

    return line_to_nodes


def is_string_contained(substring, string):
    # 删除两个字符串中的空格、换行符和单双引号
    cleaned_substring = re.sub(r'[\s\'"]', '', substring.strip()).replace('\\', '')
    cleaned_string = re.sub(r'[\s\'"]', '', string).replace('\\', '')
    return cleaned_substring in cleaned_string


def is_string_equal(substring, string):
    # 删除两个字符串中的空格、换行符和单双引号
    cleaned_substring = re.sub(r'[\s\'"]', '', substring.strip()).replace('\\', '')
    cleaned_string = re.sub(r'[\s\'"]', '', string).replace('\\', '')
    #print('------------------------------------------------------')
    #print(cleaned_substring)
    #print(cleaned_string)
    #print('------------------------------------------------------')
    # 检查清理后的子字符串是否在清理后的字符串中
    return (cleaned_substring == cleaned_string)

def refactor_context(context, pre_code):
    import autopep8
    comment_out = []
    for line in pre_code.splitlines(keepends=True):
        if line.lstrip().startswith('#') and not line.rstrip('\n').endswith('"') and not line.rstrip('\n').endswith("'") and line.count('#') == 1:
            new_line = line.replace('#', '"""#')
            if new_line.endswith('\n'):
                new_line = new_line.replace('\n', '"""\n')
            else:
                new_line = new_line + '"""'
            comment_out.append(new_line)
        else:
            comment_out.append(line)
    pre_code = '\n'.join(comment_out)

    # construct comment and code paris
    comment2code = {}
    tree = ast.parse(pre_code)
    new_context = ''
    curr_task = []
    line_to_nodes = sort_nodes_by_line(tree)
    for idx, (linen_key, line_values) in enumerate(line_to_nodes.items()):
        comment2code[str(idx)] = {}
        line = max([ast.unparse(line_value) for line_value in line_values], key=len)
        if not is_string_contained(line, new_context):
            new_context = new_context + line
            if line.startswith("'#"):
                curr_task.append(line)
            elif not line.strip('\n').startswith(' '):
                comment2code[str(idx)]['task'] = curr_task
                comment2code[str(idx)]['code'] = line
                curr_task = []


    #print('///////////////////////////comment2code////////////////////////////////////////')
    #print(comment2code)
    #print('///////////////////////////context////////////////////////////////////////')
    #print(context)
    #print('//////////////////////////////////////////////////////////////////////////')
    output = []
    dict_keys = list(comment2code.keys())
    dict_keys = sorted(dict_keys, key=int)

    if len(dict_keys) > 0:
        # stacking each code
        for idx, ii in enumerate(dict_keys):
            if not comment2code[str(ii)]:
                continue
            code = comment2code[str(ii)]['code']
            task = comment2code[str(ii)]['task']
            for line_code in context:
                try:
                    line_code = ast.literal_eval(line_code)
                    line_code = line_code.replace('("', "('").replace('")', "')")
                except:
                    pass
                #print('---------------------------------line-code--------------------------------------')
                #print(repr(line_code))
                #print('-------------------------------------------code----------------------------------------')
                #print(repr(code))
                if is_string_equal(str(line_code), str(code)) and line_code not in output:
                    # print('line_code:', line_code, 'collect_code:', ast.unparse(key), 'value:', value)
                    if task:
                        output.append(task[-1])
                    output.append(line_code)
                    break

    #print(output)
    fixed_code = '\n'.join(output)

    final_out = []
    for line in fixed_code.splitlines(keepends=True):
        if line.lstrip().startswith('"""#'):
            line = line.replace('"""#', '#').replace('"""\n', '\n')
        if line.lstrip().startswith("'#"):
            line = line.replace("'#", '#').replace("'\n", '\n')
        final_out.append(line)
    fixed_code = ''.join(final_out)

    fixed_code = autopep8.fix_code(fixed_code)
    return fixed_code


class ImportFinder(ast.NodeVisitor):
    def __init__(self, code):
        self.import_statements = []
        self.code = code

    def visit_Import(self, node):
        # 将 import 语句的源代码添加到列表中
        self.import_statements.append(ast.get_source_segment(self.code, node))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        # 将 from ... import ... 语句的源代码添加到列表中
        self.import_statements.append(ast.get_source_segment(self.code, node))
        self.generic_visit(node)


def update_code(code, prompt, solution):
    improved_code = ''
    improved_prompt = ''
    if code:
        message_prompt = [
            {"role": "system",
             "content": f"You goal is to fix the current code from proposed solutions."},
            {"role": "user",
             "content": f"I first give you the Python code: \n{code}"
                        + f"\nFix it from the solution: {solution}; \nor by yourself."
                        + f"\nFix directly on \n{code}", }, ]
        input_prompt_ids = tokenizer.apply_chat_template(message_prompt, add_generation_prompt=True, return_tensors="pt").to(model.device)  # llama3.1
        prompt_outputs = model.generate(input_prompt_ids, max_new_tokens=512, eos_token_id=terminators, temperature=0.9)
        improved_code = tokenizer.decode(prompt_outputs[0][input_prompt_ids.shape[-1]:], skip_special_tokens=True)
        print('----------------------fixed code--------------------------------')
        print(improved_code)
    else:
        message_prompt = [
            {"role": "system",
             "content": f"You goal is to improve the current prompt for code generation from proposed suggestion."},
            {"role": "user",
             "content": f"I first give you the prompt: {prompt}"
                        + f"\nImprove it from the suggestion: {solution}; \nor by yourself."}, ]
        input_prompt_ids = tokenizer.apply_chat_template(message_prompt, add_generation_prompt=True, return_tensors="pt").to(model.device)  # llama3.1
        prompt_outputs = model.generate(input_prompt_ids, max_new_tokens=512, eos_token_id=terminators, temperature=0.9)
        improved_prompt = tokenizer.decode(prompt_outputs[0][input_prompt_ids.shape[-1]:], skip_special_tokens=True)
        print('----------------------improved prompt--------------------------------')
        print(improved_prompt)

    return improved_code, improved_prompt


def find_functions_in_code(code):
    # 解析代码为 AST
    tree = ast.parse(code)

    # 找到所有函数定义
    functions = [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    return functions

def remove_duplicate_lines(context):
    new_context = ''
    unique_lines = []
    tree = ast.parse(context)
    line_to_nodes = sort_nodes_by_line(tree)
    for key, values in line_to_nodes.items():
        line = max([ast.unparse(value) for value in values], key=len)
        if not is_string_contained(line, new_context):
            new_context = new_context + line
            unique_lines.append(line)
    return unique_lines

def arrange_code_block(code_block):
    # extract imports
    tree = ast.parse(code_block)
    finder = ImportFinder(code_block)
    finder.visit(tree)
    imports = finder.import_statements
    #for imp in imports:
    #    code_block = code_block.replace(imp + '\n', '')
    code_block_ = '\n'.join(imports)
    funcs = find_functions_in_code(code_block)
    if len(funcs) > 0:
        for func in funcs:
            code_block = code_block.replace(func, '')
        code_block_ = code_block_ + '\n' + '\n'.join(funcs)
    code_block = code_block_ + '\n' + code_block

    try:
        new_context = ''
        unique_lines = []
        tree = ast.parse(code_block)
        line_to_nodes = sort_nodes_by_line(tree)
        for linen_key, line_values in line_to_nodes.items():
            line = max([ast.unparse(line_value) for line_value in line_values], key=len)
            if not is_string_contained(line, new_context):
                new_context = new_context + line
                unique_lines.append(line + '\n')

        selected_code_block = ''.join(unique_lines)
        error = None
    except:
        selected_code_block = ''
        error = 'avoid the IndentationError: unexpected indent or SyntaxError: invalid syntax'
    return selected_code_block, error

def clean_code(code, pre_code):
    code_list = remove_duplicate_lines(code)
    clean_code_ = refactor_context(code, pre_code)
    for code_ in code_list:
        if code_ not in clean_code_:
            clean_code_ += code_ + '\n'

    print('----------------------fixed code--------------------------------')
    print(clean_code_)
    print('----------------------end fixed code--------------------------------')
    return clean_code_


def enrich_prompt(curr_node):
    no_name, no_attribute, others, token = curr_node.error
    if no_name:
        source = curr_node.failed_code
        message_prompt = [
            {"role": "system",
             "content": f"Your goal is to fix the missing variables in the code."},
            {"role": "user",
             "content": f"I first give you the Python code: \n{source}" + f"\nGive your solution to fix {no_name[0]} error or define {no_name[0]}: ", }, ]
        error = f'define {no_name[0]}'
    elif no_attribute:
        source = curr_node.failed_code.replace(list(no_attribute.values())[0], '<MISSING>')
        message_prompt = [
            {"role": "system",
             "content": f"Your goal is to fix the missing function in the code."},
            {"role": "user",
             "content": f"I first give you the Python code: \n{source}" + f"\nGive your solution in <MISSING> place: ", }, ]
        error = f'replace {list(no_attribute.values())[0]}'
    elif others:
        source = curr_node.failed_code
        message_prompt = [
            {"role": "system",
             "content": f"Your goal is to fix the error in the code."},
            {"role": "user",
             "content": f"I first give you the Python code: \n{source}" + f"\nGive your solution for fixing the error: {others[0]}", }, ]
        error = f'fix {others[0]}'
    else:
        error = None
        message_prompt = None

    if message_prompt:
        input_prompt_ids = tokenizer.apply_chat_template(message_prompt, add_generation_prompt=True, return_tensors="pt").to(model.device)  # llama3.1
        prompt_outputs = model.generate(input_prompt_ids, max_new_tokens=512, eos_token_id=terminators, temperature=0.9)
        prompt_response = tokenizer.decode(prompt_outputs[0][input_prompt_ids.shape[-1]:], skip_special_tokens=True)
        print('----------------------solution--------------------------------')
        print(prompt_response)
    else:
        prompt_response = None

    return error, prompt_response


def model_generate_long_context(curr_node, completions, max_tokens, temperature, num_return_sequences, exec_path):
    work_path = f"import os\nos.chdir('{exec_path}')\nprint('the current work path is:', os.getcwd())\n"
    pre_code = curr_node.pre_code
    prompt = curr_node.prompt
    assert num_return_sequences > 1
    if completions:
        completions = [str(idx) + ':' + completion if completion is not None else 'None' for idx, completion in
                       enumerate(completions)]
        completions = '\n'.join(completions)
        messages = [
            {"role": "system",
             "content": f"I want you to become my Expert Python programmer. Your goal is to turn the prompt into Python code."},
            {"role": "user",
             "content": f"\nHere are three suggested code: \n{completions}."
                        + "\n{" + f"'prompt':{prompt}, " + f"'code':'''Python\n{pre_code}" + "\n#YC# write your answer below\n" + "'''\n}", }, ]
    else:
        messages = [
            {"role": "system",
             "content": f"I want you to become my Expert programmer. Your goal is to turn the prompt into Python code."},
            {"role": "user",
             "content": "\n{" + f"'prompt':{prompt}, " + f"'code':'''Python\n{pre_code}" + "\n#YC# write your answer below\n" + "'''\n}", }, ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids, max_new_tokens=max_tokens, eos_token_id=terminators, temperature=temperature)
    code_i_initial = tokenizer.decode(outputs[0][input_ids.shape[-1]:])
    print('-----------------------initial code------------------------')
    print(code_i_initial)

    # code_i, error = extract_code_split(code_i_initial, pre_code)
    code_i = extract_code_again(code_i_initial)
    print('------------------------extracted code-----------------------')
    print('extrated code is:', code_i)
    if not code_i.strip():
        # code_i = pre_code + completions[0].split(':')[1]
        # code_i = extract_code_again(code_i) + '\n#YC# complete!'
        code_i = pre_code + '\n#YC# complete!'
    else:
        #code_i = clean_code(pre_code + '\n' + code_i, pre_code)
        code_i, error = arrange_code_block(pre_code + '\n' + code_i)
    print(code_i)
    print('------------------------end code--------------------------')

    if code_i:
        cal_reward = 1
        #return_back = monitor_process(work_path + code_i, MAX_EXECUTION_TIME, MAX_MEMORY_USAGE)
        #if return_back:
        #    print('-------------------some error in long context--------------------')
        #    print(return_back)
    else:
        code_i = code_i_initial
        cal_reward = 0

    return code_i, 0, cal_reward


def model_generate_topk(curr_node, max_tokens, temperature, num_return_sequences):
    assert num_return_sequences > 1
    pre_code = curr_node.pre_code
    prompt = curr_node.prompt
    messages = [
        {"role": "system",
         "content": f"I want you to become my Expert Python programmer. Your goal is to turn the prompt into Python code."},
        {"role": "user",
         "content": "\n{" + f"'prompt':{prompt}, " + f"'code':'''Python\n{pre_code}" + "\n#YC# write your answer below\n" + "'''\n}", }, ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids, max_new_tokens=max_tokens, eos_token_id=terminators, temperature=temperature,
                             num_return_sequences=num_return_sequences, do_sample=True,
                             output_scores=True, return_dict_in_generate=True)
    code = []
    errors = []
    # compute the scores using compute_transition_scores()
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    transition_scores = torch.where(torch.isinf(transition_scores), torch.tensor(0.0), transition_scores)
    length_penalty = 1.0
    output_length = input_ids.shape[1] + np.sum(transition_scores.cpu().float().numpy() < 0, axis=1)
    reconstructed_scores = np.exp(transition_scores.sum(axis=1).cpu().float().numpy() / (output_length ** length_penalty))
    print('topk scores:', reconstructed_scores)
    for i in range(num_return_sequences):
        response = outputs['sequences'][i]
        response = response[input_ids.shape[-1]:]  # if add generation prompt = true
        code_i = tokenizer.decode(response, skip_special_tokens=True)
        print('pre topk coding:', code_i)
        # extract_code_i, error = extract_code_split(code_i, pre_code)
        extract_code_i = extract_code_again(code_i)
        #extract_code_i = clean_code(pre_code + '\n' + extract_code_i, pre_code)
        extract_code_i, error = arrange_code_block(pre_code + '\n' + extract_code_i)
        print('topk error:', error)
        print('topk coding:', extract_code_i)
        if extract_code_i:
            code.append(extract_code_i)
        else:
            errors.append(error)
    return code, reconstructed_scores, errors


def main():
    max_rollouts = 2
    top_k = 2
    beam_width = 2
    # hyperparameters for P-UCB function
    c_base = 10
    c = 4

    class Node:
        id_iter = itertools.count()

        def __init__(self, logprob, label, prompt, failed_code, solution, pre_code, parent, value=0, error=[]):
            self.value = value  # total reward obtainable from node
            self.prob = exp(logprob)  # necessary for P-UCB calculation
            self.prompt = prompt  # prompt for the whole task
            self._children = []
            self._parent = parent
            self.visits = 0
            self.runs = 0
            # attributes for graph visualisation
            self.id = next(self.id_iter)
            self.label = label  # the label for current task
            self.failed_code = failed_code  # the failed child code for the current task
            #self.succes_code = succes_code  # the success child code for the current task
            self.solution = solution  # for fixing failed code or prompt
            self.pre_code = pre_code  # the stacking of previous code
            self.p_ucb = 0  # last calculated p_ucb value
            self.error = error  # use to record error in childNode

        def backprop(self, value):
            # only propagate if new reward is greater than current max
            if value > self.value:
                self.value = value
                if self._parent is not None:
                    self._parent.backprop(value)

    class NodeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Node):
                cpy = copy.copy(obj)
                del cpy._parent
                del cpy._children
                return vars(cpy)
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, obj)

    def p_ucb_select(parent_node, child_nodes):
        s_visits = parent_node.visits
        beta = log((s_visits + c_base + 1) / c_base) + c

        max_p_ucb = -inf
        max_node = None
        for i in range(len(child_nodes)):
            node = child_nodes[i]
            p_ucb = node.value + beta * node.prob * sqrt(log(s_visits)) / (
                    1 + node.visits)
            print('-----------------------------------------selecting node---------------------------------')
            if node.pre_code:
                print(node.pre_code)
            elif node.failed_code:
                print(node.failed_code)
            print(p_ucb)
            node.p_ucb = p_ucb  # store most recent p_ucb for visualisation
            if p_ucb > max_p_ucb:
                max_node = node
                max_p_ucb = p_ucb
        return max_node

    def get_top_k_tokens(curr_node, k=2):
        if curr_node.solution:
            improved_code, improved_prompt = update_code(curr_node.failed_code, curr_node.prompt, curr_node.solution)
            return improved_code, improved_prompt
        else:
            temperature = 0.8 #random.uniform(0.7, 1)
            max_tokens = 400 #random.randint(512, 1024)
            tokens, probs, errors = model_generate_topk(curr_node=curr_node, max_tokens=max_tokens, temperature=temperature,
                                                    num_return_sequences=k)
        return tokens, probs, errors

    def beam_search(curr_node, tokens, exec_path):
        print('-----------------------------beam search------------------------')
        temperature = 0.8 #random.uniform(0.7, 1)
        max_tokens = 700 #random.randint(512, 1536)
        tokens, probs, cal_reward = model_generate_long_context(curr_node, completions=tokens, max_tokens=max_tokens,
                                                    temperature=temperature, num_return_sequences=beam_width, exec_path=exec_path)
        print(tokens)
        return tokens, cal_reward

    def calculate_reward(precode, completion, exec_path):
        work_path = f"import os\nos.chdir('{exec_path}')\nprint('the current work path is:', os.getcwd())\n"
        new_context = ''
        unique_lines = []
        # unique line
        tree = ast.parse(completion)
        line_to_nodes = sort_nodes_by_line(tree)
        for linen_key, line_values in line_to_nodes.items():
            line = max([ast.unparse(line_value) for line_value in line_values], key=len)
            if not is_string_contained(line, new_context):
                new_context = new_context + line
                unique_lines.append(line + '\n')

        code_len = len(unique_lines)
        test_code = ''
        if code_len == 0:
            return 0
        elif all(code_i in precode for code_i in unique_lines):
            print('**************************task completed!!*******************************888')
            return 1
        else:
            success_len = 0
            for code_i in unique_lines:
                test_code = test_code + '\n' + code_i
                return_back = monitor_process(work_path + test_code, MAX_EXECUTION_TIME, MAX_MEMORY_USAGE)
                if return_back is None:
                    success_len += 1
                elif ('ERROR' not in return_back) and ('Error' not in return_back):
                    success_len += 1
                else:
                    break
            return success_len / code_len

    def get_best_program(program_dict):
        max_reward = -inf
        best_program = None
        for program, reward in program_dict.items():
            print('-----------------------select best completion-----------------------')
            print(program, reward)
            if reward > max_reward:
                best_program = program
                max_reward = reward
        return best_program, max_reward

    def check_child_nodes(code, exec_path):
        work_path = f"import os\nos.chdir('{exec_path}')\nprint('the current work path is:', os.getcwd())\n"
        no_name = []
        no_attribute = {}
        others = []
        print('-------------------------------checking child code -------------------------')
        print(code)
        # code pass from interpreter
        if not code.endswith(':\n'):
            try:
                if code.endswith('.'):
                    code = code[:-1]
                elif code.endswith(' \\\n'):
                    code = code.replace(' \\\n', '')
                ast.parse(code)
                if 'exit()' in code:
                    code.replace('exit()', 'pass')

                return_back = monitor_process(work_path + code, MAX_EXECUTION_TIME, MAX_MEMORY_USAGE)

                if return_back:
                    print('------------------------some error in checked child node----------------------------')
                    print(return_back)

                    if 'NameError' in return_back:
                        var = return_back.split("name '")[1].split("'")[0]
                        no_name.append(var)
                    elif 'AttributeError' in return_back:
                        obj = return_back.split("module '")[1].split("'")[0]
                        att = return_back.split("attribute '")[1].split("'")[0]
                        # complete using jedi
                        script = jedi.Script(source)
                        cliped_source = source.split('at')[0] + '\n'
                        multilines = cliped_source.splitlines()
                        jedi_completion = script.complete(line=len(multilines) - 1, column=len(multilines[-1]))
                        jedi_completion = [com.name for com in jedi_completion]
                        #print(jedi_completion)
                        no_attribute[obj] = att + jedi_completion
                    elif ('ERROR' in return_back) or ('Error' in return_back):
                        others.append(return_back)
            except Exception as e:
                others.append(e)

        return no_name, no_attribute, others

    def count_occurrences(lst, target_str):
        return lst.count(target_str)


    task_path = '/workspace/comparison_bench/GeoCode_GEE_selected.jsonl'
    # this is for cibench
    exec_path_root = '/workspace/comparison_bench/GeoSpatial2task/'
    base_dir = '/workspace/RAG_bench/ds1000/'
    task_name = os.path.basename(task_path).split('.')[0]
    output_doc = '/workspace/comparison_bench/sequential_task/GeoCode_GEE_selected_mcts.jsonl'

    tasks = []
    ori_task_id = 0
    with jsonlines.open(task_path, mode='r') as eval_reader:
        samples = [sample for sample in eval_reader][0:100]
        for sample in samples:
            if 'task' in sample:
                task = sample['task']
            elif 'prompt' in sample:
                task = sample['prompt']
            else:
                task = None
            if 'context' in sample:
                pre_task = sample['context']
            elif 'pre_code' in sample:
                pre_task = sample['pre_code']
            else:
                pre_task = None

            if 'exec_path' in sample:
                exec_path_ = sample['exec_path']
            else:
                exec_path_ = None

            if 'exlib' in sample:
                library = sample['exlib']
            elif 'libraries' in sample:
                library = sample['libraries']
            else:
                library = None
            new_library = []
            if library is not None:
                for lib in library:
                    if 'import' not in lib:
                        new_library.append(lib)
                    elif 'from' in lib:
                        lib = lib.split('import')[0].split('from')[1].strip()
                        if '.' in lib:
                            lib = lib.split('.')[0].strip()
                        new_library.append(lib)
                    elif 'import' in lib:
                        lib = lib.split('import')[1].strip()
                        if 'as' in lib:
                            lib = lib.split('as')[0].strip()
                        if '.' in lib:
                            lib = lib.split('.')[0].strip()
                        new_library.append(lib)
                new_library = list(set(new_library))

            if pre_task is not None:
                tasks.append({'task': task, 'code': sample['code'], 'library': new_library, 'pre_task': pre_task,
                              'ori_task_id': ori_task_id, 'exec_path': exec_path_})
            else:
                _pre_task = extract_code_again(sample['prompt'])
                tasks.append({'task': task, 'code': sample['code'], 'library': new_library, 'pre_task': _pre_task,
                              'ori_task_id': ori_task_id, 'exec_path': exec_path_})
            ori_task_id += 1

    # cut tasks according to the output
    len_tasks = len(tasks)
    if os.path.exists(output_doc):
        with open(output_doc, 'r', encoding='utf-8') as file:
            # Read all lines
            lines = file.readlines()
            while True:
                # Get the last line
                last_line = lines.pop()
                # Parse the JSON object
                try:
                    last_json = json.loads(last_line)
                    current_line = last_json['ori_task_id']
                except:
                    current_line = len_tasks
                    time.sleep(100)

                if isinstance(current_line, int):
                    break
    else:
        current_line = len_tasks

    tasks = tasks[0:current_line]

    start = time.perf_counter()
    num_iter = 1
    # processing and add item dynamically
    task_id = current_line
    repeated_task_num = 0
    best_completion = None
    max_reward = None
    task_log = []
    while tasks:
        reward = None
        restart_flag = False
        # print tasks
        print('----------------------start task---------------------------------')
        current_task = tasks.pop()  # select the last one
        if 'ori_task_id' in current_task:
            ori_task_id = current_task['ori_task_id']

        if (max_reward == 1) and best_completion:
            pre_tasks = best_completion
        else:
            pre_tasks = current_task['pre_task']

        # save for next step replacing failed task in multi-step tasks
        if 'code' in current_task:
            code = current_task['code']
        else:
            code = None

        if 'runs' in current_task:
            runs = current_task['runs']
        else:
            runs = 0

        if 'solution' in current_task:
            solution = current_task['solution']
        else:
            solution = None

        if 'library' in current_task:
            library = current_task['library']
            print('here is the library!', library)
        else:
            library = None

        if 'exec_path' in current_task and current_task['exec_path'] is not None:
            exec_path = os.path.join(exec_path_root, current_task['exec_path'])
        elif current_task['ori_task_id'] == 'insert':
            pass
        else:
            exec_path = exec_path_root

        if not pre_tasks.endswith('\n'): pre_tasks = pre_tasks + '\n'
        prompt = current_task['task'].replace('prompt', '').replace('{', '').replace('}', '').replace(':', '').replace(
            "''", '')
        task_log.append(prompt)
        '''
        # --------------------------------------------RAG----------------------------
        if 'exlib' in current_task:
            library = current_task['exlib']
            rag_dir_list = [base_dir + lib for lib in library]
            rag_dir_list_exist = all(os.path.exists(rag_dir) for rag_dir in rag_dir_list)
            rag_dir_list_exist = False
        else:
            rag_dir_list_exist = False
        rag_context = []
        if rag_dir_list_exist:
            for lib in library:
                rag_dir = base_dir + lib
                filelist = glob.glob(os.path.join(rag_dir, '*.json'))
                file_nm = [os.path.basename(nm) for nm in filelist]
                library_list = [nm.split('.')[0] for nm in file_nm]
                library_metadatas = {}
                for library in library_list:
                    library_metadatas[library] = {'library': library}  # it may can change to description of the library
                # load all documents
                docs_dict = []
                for doc_nm in file_nm:
                    library = doc_nm.split('.')[0]
                    doc = SimpleDirectoryReader(
                        input_files=[f"{rag_dir}/{doc_nm}"]
                    ).load_data()[0]
                    doc.metadata.update(library_metadatas[library])
                    docs_dict.append(doc)
                # build vector
                # simple vector store
                vector_store = SimpleVectorStore()
                vector_storage_context = StorageContext.from_defaults(vector_store=vector_store)
                indexes = []
                for doc in docs_dict:
                    vector_index = VectorStoreIndex.from_documents(
                        [doc], service_context=service_context, storage_context=vector_storage_context
                    )
                    indexes.append(vector_index)
                # build retriver
                indexes_retriever = [idx.as_retriever() for idx in indexes]
                if len(indexes_retriever) < 1:
                    continue
                try:
                    retriever = QueryFusionRetriever(
                        indexes_retriever,
                        similarity_top_k=3,
                        num_queries=1,  # set this to 1 to disable query generation
                        use_async=True,
                        verbose=True,
                    )
                except:
                    continue
                # run recursive retriever
                rag_outputs = retriever.retrieve(prompt)
                for rag_output in rag_outputs:
                    rag_output_text = rag_output.node.get_content()
                    if len(rag_output_text) > 4000:
                        rag_output_text = rag_output_text[0:4000]
                    rag_context.append(rag_output_text)
                print('------------------------------------rag start-------------------------------------------------')
                print('rag_context', rag_context)
                print('------------------------------------rag end-------------------------------------------------')
                del docs_dict, vector_store, vector_storage_context, indexes, vector_index, indexes_retriever, retriever

            prompt = prompt + f"here is the possible functions: {rag_context}"

        else:
            pass
        '''
        task_id = task_id + 1
        prompt_start = time.perf_counter()
        print(f"---- STARTING MCTS FOR {str(task_id)} ({num_iter}/{len(tasks)}) ----")
        print(prompt)
        # cache of generated programs => rewards
        program_dict = {}
        num_rollouts = max_rollouts
        root = Node(logprob=log(1), label=code, prompt=prompt, failed_code='', solution=solution, pre_code=pre_tasks,
                    parent=None)
        root.runs = runs
        test_times = [0, ]
        # graph snapshots for web visualisation
        nodes, edges = {root.id: root}, {}
        graph_dict = {}
        for i in range(max_rollouts):
            if i > 0:
                print('!!!!!!!!!!!!!!!!!!!!!!!continue steps!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            graph_dict[i] = {
                "selectedNodes": [root.id],
                "state": "",
                "completion": "",
                "reward": 0.0,
                "task_id": task_id,
                "childNodes": {},
            }
            curr_node = root
            curr_node.visits += 1
            curr_node.runs += 1
            # selection
            while len(curr_node._children) > 0:
                for child in curr_node._children:
                    nodes[child.id] = child
                    edges[(curr_node.id, child.id)] = True
                curr_node = p_ucb_select(curr_node, curr_node._children)
                graph_dict[i]["selectedNodes"].append(curr_node.id)
                curr_node.visits += 1
                curr_node.runs += 1

            print('------------------------curr_node---------------------')
            print('-----------------curr_node.pre_code---------------')
            print(curr_node.pre_code)
            print('--------------curr_node.prompt-----------------')
            print(curr_node.prompt)

            if curr_node.value == -1 and count_occurrences(task_log, curr_node.prompt) < 3:
                print('-----------------appending new task-----------------')
                _error, _updated_prompt = enrich_prompt(curr_node)
                # insert new task
                current_task = {'task': curr_node.prompt, 'code': curr_node.label, 'exlib': 'ee',
                                'pre_task': curr_node.pre_code, 'ori_task_id': ori_task_id, 'runs': curr_node.runs}
                tasks.append(current_task)
                new_task = {'task': curr_node.prompt, 'code': curr_node.label, 'exlib': 'ee',
                            'failed_code': curr_node.failed_code, 'pre_task': curr_node.pre_code,
                            'solution': _updated_prompt, 'ori_task_id': 'insert', 'runs': curr_node.runs}
                tasks.append(new_task)
                restart_flag = True
                break

            print('---start to expand using this information------')
            if curr_node.solution:
                improved_code, improved_prompt = get_top_k_tokens(curr_node, top_k)
                if improved_code:
                    curr_node.prompt = improved_prompt
                if improved_code:
                    curr_node.pre_code = improved_code
                curr_node.solution = None

            tokens, probs, top_k_errors = get_top_k_tokens(curr_node, top_k)
            child_nodes = []
            if tokens:
                for (token, prob) in zip(tokens, probs):
                    no_name, no_attribute, others = check_child_nodes(token, exec_path)
                    error_list = [no_name, no_attribute, others, token]
                    print(error_list)
                    if not no_name and not no_attribute and not others:
                        child_nodes.append(Node(logprob=prob, label=curr_node.label, prompt=curr_node.prompt,
                                                failed_code=None, solution=None,
                                                pre_code=token, parent=curr_node, value=0))
                    else:
                        child_nodes.append(Node(logprob=prob, label=curr_node.label, prompt=curr_node.prompt,
                                                failed_code=token, solution=None,
                                                pre_code=curr_node.pre_code, parent=curr_node, value=-1,
                                                error=error_list))

            elif (curr_node.runs > 3) or (
                    count_occurrences(task_log, curr_node.prompt) > 3):  # cannot continue since no child node
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~ FAILD in UPDATING~~~~~~~~~~~~~~~~~~~~~~~~~~~~', repeated_task_num)
                # failed on current task
                completion, cal_reward = beam_search(curr_node, '', exec_path)
                test_start = time.perf_counter()
                if cal_reward:
                    reward = calculate_reward(curr_node.pre_code, completion, exec_path)
                else:
                    reward = -1
                test_end = time.perf_counter()
                test_times.append(test_end - test_start)
                program_dict[completion] = reward
                graph_dict[i]["state"] = curr_node.pre_code
                graph_dict[i]["completion"] = completion
                graph_dict[i]["reward"] = reward
                break

            else:
                print(
                    '++++++++++++++++++++++++++++++++++++++stuck in updating+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')  # nothing generated
                current_task['solution'] = 'No functional code, please generate better Python code!'
                current_task['runs'] = curr_node.runs
                tasks.append(current_task)
                restart_flag = True
                break

            curr_node._children = child_nodes
            for child in child_nodes:
                nodes[child.id] = child
                edges[(curr_node.id, child.id)] = True

            # evaluation
            if not reward:
                print('final reward evaluation!!!')
                completion, cal_reward = beam_search(curr_node, tokens, exec_path)
                test_start = time.perf_counter()
                if cal_reward:
                    reward = calculate_reward(curr_node.pre_code, completion, exec_path)
                else:
                    reward = -1
                test_end = time.perf_counter()
                test_times.append(test_end - test_start)
                program_dict[completion] = reward
                graph_dict[i]["state"] = curr_node.pre_code
                graph_dict[i]["completion"] = completion
                graph_dict[i]["reward"] = reward

            graph_dict[i]["nodes"] = list(nodes.values())
            graph_dict[i]["edges"] = list(edges.keys())

            # backprop
            curr_node.backprop(reward)

            if reward == 1:
                num_rollouts = i + 1
                break

        if restart_flag:
            continue

        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@come to the results@@@@@@@@@@@@@@@@@@@@@@@@@:', ori_task_id)

        repeated_task_num = 0
        best_completion, max_reward = get_best_program(program_dict)

        if reward != 1:
            print('$$$$$$$$$$$$$$$$$$$$$$$$$ The task is failed $$$$$$$$$$$$$$$$$$$$$$$$$')
            print(max_reward, best_completion)

        end = time.perf_counter()
        item = dict(
            task_id=task_id,
            ori_task_id=ori_task_id,
            reward=reward,
            gen_code=best_completion,
            prompt=prompt,
            code=code,
            pre_tasks=curr_node.pre_code,
            stats=dict(
                num_rollouts=num_rollouts,
                num_generations=len(program_dict.keys()),
                eval_time=f"{(end - prompt_start):.4f}s",
                mean_test_time=f"{(sum(test_times) / len(test_times)):.4f}s",
            ),
        )

        write_jsonl(f"{task_name}_mcts.jsonl", [item], append=True)
        print(f"---- COMPLETED MCTS FOR {str(task_id)} ({num_iter}/{len(tasks)}) ----")
        print(f"Eval time: {(end - prompt_start):.4f}s")
        print(f"Mean test time: {(sum(test_times) / len(test_times)):.4f}s")
        print(f"Stats: {item['stats']}")
        num_iter += 1
        render_graphviz_tree(
            root, filename=f"svgviz/tree_{task_name}_{str(task_id)}", view=False
        )
        with open(f"graph_{task_name}_{str(task_id)}.json", "w") as f:
            # print('--------------------------------graph-----------------------------------')
            # print(graph_dict)
            try:
                json.dump(graph_dict, f, cls=NodeEncoder)
            except:
                # 处理字典
                graph_dict = make_serializable(graph_dict)
                json.dump(graph_dict, f, cls=NodeEncoder)

        if ori_task_id != 'insert':
            best_completion = None

    end = time.perf_counter()
    print(f"Total elapsed time: {(end - start):.4f}s\n")


# necessary to prevent multiple executions of main() within stats_execute threads
if __name__ == "__main__":
    main()
