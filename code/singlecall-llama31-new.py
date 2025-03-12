# RAG
import nest_asyncio
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
model_name_path = '/workspace/unsloth-llama-3.1-8b-Instruct'
model_name = '/workspace/unsloth-llama-3.1-8b-Instruct'

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
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]


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


def model_generate_long_context(prompt, pre_tasks, max_tokens, temperature, exec_path):
    work_path = f"import os\nos.chdir('{exec_path}')\nprint('the current work path is:', os.getcwd())\n"
    pre_code = pre_tasks.strip()
    prompt = prompt
    if pre_code:
        messages = [
            {"role": "system",
             "content": f"I want you to become my Expert Python programmer. Your goal is to turn the prompt into Python code."},
            {"role": "user",
             "content": "\n{" + f"'prompt':{prompt}, " + f"'code':'''Python\n{pre_code}" + "\n#YC# write your answer below\n" + "'''\n}", }, ]
    else:
        messages = [
            {"role": "system",
             "content": f"I want you to become my Expert programmer. Your goal is to turn the prompt into Python code."},
            {"role": "user",
             "content": "\n{" + f"'prompt':{prompt}, " + f"'code':'''Python\n#YC# write your answer below\n" + "'''\n}", }, ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids, max_new_tokens=max_tokens, eos_token_id=terminators, temperature=temperature)
    code_i_initial = tokenizer.decode(outputs[0][input_ids.shape[-1]:])
    print('-----------------------initial code------------------------')
    print(code_i_initial)
    #code_i, error = extract_code_split(code_i_initial, pre_code)
    code_i = extract_code_again(code_i_initial)
    print('------------------------extracted code-----------------------')
    print('extrated code is:', code_i)
    if not code_i.strip():
        #code_i = pre_code + completions[0].split(':')[1]
        #code_i = extract_code_again(code_i) + '\n#YC# complete!'
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


def main():
    def beam_search(prompt, pre_tasks, exec_path):
        print('-----------------------------beam search------------------------')
        temperature = 0.9
        max_tokens = 700
        tokens, probs, cal_reward = model_generate_long_context(prompt, pre_tasks, max_tokens=max_tokens,
                                                                temperature=temperature, exec_path=exec_path)
        print(tokens)
        return tokens, cal_reward

    def calculate_reward(precode, completion, exec_path):
        work_path = f"import os\nos.chdir('{exec_path}')\nprint('the current work path is:', os.getcwd())\n"
        new_context = ''
        unique_lines = []
        # test if no output in completion
        # if '#YC' in completion:
        #    completion = completion.split('#YC')[0]
        # gen_code_ = extract_code_again(completion)
        # if not gen_code_:
        #    return 1
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
                elif ('ERROR' not in return_back) and ('Error' not in return_back) and (
                        'EEException' not in return_back):
                    success_len += 1
                else:
                    break
            return success_len / code_len

    task_path = '/workspace/comparison_bench/GeoCode_GEE_selected.jsonl'
    # this is for cibench
    #exec_path_root = '/workspace/comparison_bench/sequential_task/'
    exec_path_root = '/workspace/comparison_bench/GeoSpatial2task'
    output_doc = '/workspace/comparison_bench/sequential_task/GeoCode_GEE_selected_single.jsonl'
    task_name = os.path.basename(task_path).split('.')[0]

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

    num_iter = 1
    # processing and add item dynamically
    task_id = current_line
    while tasks:
        # print tasks
        print('----------------------start task---------------------------------')
        current_task = tasks.pop()  # select the last one
        if 'ori_task_id' in current_task:
            ori_task_id = current_task['ori_task_id']

        if 'pre_task' in current_task:
            pre_tasks = current_task['pre_task']

        # save for next step replacing failed task in multi-step tasks
        if 'code' in current_task:
            code = current_task['code']
        else:
            code = None

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

        task_id = task_id + 1
        print(f"---- STARTING Coding FOR {str(task_id)} ({num_iter}/{len(tasks)}) ----")
        print(prompt)

        print('final reward evaluation!!!')
        completion, cal_reward = beam_search(prompt, pre_tasks, exec_path)
        if cal_reward:
            reward = calculate_reward(pre_tasks, completion, exec_path)
            completion = completion.replace('\n#YC# complete!', '')
        else:
            reward = -1

        item = dict(
            task_id=task_id,
            ori_task_id=ori_task_id,
            reward=reward,
            gen_code=completion,
            prompt=prompt,
            code=code,
            pre_tasks=pre_tasks,
            stats=dict(
            ),
        )

        write_jsonl(f"{task_name}_single.jsonl", [item], append=True)
        print(f"---- COMPLETED MCTS FOR {str(task_id)} ({num_iter}/{len(tasks)}) ----")


# necessary to prevent multiple executions of main() within stats_execute threads
if __name__ == "__main__":
    main()
