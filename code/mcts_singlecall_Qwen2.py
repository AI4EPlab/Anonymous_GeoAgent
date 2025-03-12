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

model_name_path = '/workspace/Qwen2-7B-Instruct'
model_name = 'unsloth/Qwen2-7B-Instruct'


from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    cache_dir=model_name_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|endoftext|>"),
]



def extract_code_split(text, pre_code):
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
                        code_block.append('\n'.join(one_block))
                        one_block = []
                        ast.parse('\n'.join(code_block))
                    except:
                        code_block.pop()
                        pass

                    # test current line
                    try:
                        code_block.append(text_i)
                        one_block = []
                        ast.parse('\n'.join(code_block))
                        start_id = 0
                    except:
                        code_block.pop()
                        one_block.append(text_i)
                        continue
            else:
                one_block.append(text_i)

    if 'end' in code_block:
        code_block.remove('end')
    code_block = '\n'.join(code_block)
    #print('-------------------------------initial prompt-------------------------------------')
    #print(text)
    #print('---------------------------extracted code-----------------------------------------')
    #print(code_block)

    if code_block:
        new_context = ''
        unique_lines = []
        # unique linexvcx
        tree = ast.parse(code_block)
        line_to_nodes = sort_nodes_by_line(tree)
        for linen_key, line_values in line_to_nodes.items():
            line = max([ast.unparse(line_value) for line_value in line_values], key=len)
            if not is_string_contained(line, new_context):
                new_context = new_context + line
                unique_lines.append(line + '\n')

        selected_code_block = ''.join(unique_lines)

        ## remove the precode from text
        #if (pre_code not in ['\n', ' ']) and (selected_code_block != pre_code):
        #    selected_code_block = selected_code_block.replace(pre_code, '')

        error = None
    else:
        selected_code_block = ''
        error = 'Write right python code please!'
    return selected_code_block, error


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


def enrich_prompt(curr_node):
    no_name, no_attribute, others, token = curr_node.error
    if no_name:
        source = curr_node.task + curr_node.label + f'\n{no_name[0]} = <MISSING>' + token
        message_prompt = [
            {"role": "system",
             "content": f"You gola is to fix the missing variables in the initial prompt."},
            {"role": "user",
             "content": f"I first give you the Python code: {source}" + f"\nComplete the <MISSING> place with a exact {no_name[0]}: ", }, ]
        error = f'define {no_name[0]}'
    elif no_attribute:
        source = curr_node.task + curr_node.label + token.replace(list(no_attribute.values())[0], '<MISSING>')
        message_prompt = [
            {"role": "system",
             "content": f"You gola is to fix the missing function in the initial prompt."},
            {"role": "user",
             "content": f"I first give you the Python code: {source}" + f"\nGive your solution in <MISSING> place: ", }, ]
        error = f'replace {list(no_attribute.values())[0]}'
    elif others:
        source = curr_node.task + curr_node.label + token
        message_prompt = [
            {"role": "system",
             "content": f"You gola is to fix the error in the initial prompt."},
            {"role": "user",
             "content": f"I first give you the Python code: {source}" + f"\nGive your solution for fixing the error {others[0]}: ", }, ]
        error = f'fix {others[0]}'
    else:
        error = None
        message_prompt = None

    if message_prompt:
        input_prompt_ids = tokenizer.apply_chat_template(message_prompt, add_generation_prompt=True, return_tensors="pt").to(model.device)  # llama3.1
        #input_prompt_ids = tokenizer(message_prompt, return_tensors='pt').to(model.device)  # gemma2
        #prompt_outputs = model.generate(**input_prompt_ids, max_new_tokens=128, eos_token_id=terminators, temperature=0.9) # gemma2
        #prompt_outputs = model.generate(input_prompt_ids, max_new_tokens=128, eos_token_id=terminators, temperature=0.9) #llama3.1
        prompt_outputs = model.generate(input_prompt_ids, max_new_tokens=128, eos_token_id=terminators, temperature=0.9)
        prompt_response = tokenizer.decode(prompt_outputs[0], skip_special_tokens=True)
        print('----------------------enriched prompt--------------------------------')
        print(prompt_response)
    else:
        prompt_response = None

    return error, prompt_response


def model_generate_long_context(curr_node, completions, max_tokens, temperature, num_return_sequences):
    pre_code = curr_node.task + curr_node.label
    prompt = curr_node.prompt
    print(pre_code, prompt)
    assert num_return_sequences > 1
    if completions:
        completions = [str(idx) + ':' + completion if completion is not None else 'None' for idx, completion in enumerate(completions)]
        completions = ','.join(completions)
        messages = [
            {"role": "system",
             "content": f"I want you to become my Expert Python programmer. Your goal is to complete the code."},
            {"role": "user",
             "content": f"I first give you the previous code: {pre_code}" + f'The task is {prompt}' + f"\nhere are two suggested code: {completions}" + f"\nif the task is completed by previous code, just give the same output as the previous code otherwise give your answer:"}, ]
    else:
        messages = [
            {"role": "system",
             "content": f"I want you to become my Expert programmer. Your goal is to complete the code."},
            {"role": "user",
             "content": f"I first give you the previous code: {pre_code}" + f'The task is {prompt}' + f"\nif the task is completed by previous code, just give the same output as the previous code otherwise give your answer:"}, ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    #input_ids = tokenizer(messages, return_tensors='pt').to(model.device)  # gemma2
    #outputs = model.generate(**input_ids, max_new_tokens=max_tokens, eos_token_id=terminators, temperature=temperature)
    outputs = model.generate(input_ids, max_new_tokens=max_tokens, eos_token_id=terminators, temperature=temperature)
    code_i_initial = tokenizer.decode(outputs[0])
    code_i, error = extract_code_split(code_i_initial, curr_node.label)
    if code_i:
        cal_reward = 1
        code_snippet = pre_code + code_i
        return_back = monitor_process(code_snippet, MAX_EXECUTION_TIME, MAX_MEMORY_USAGE)
        if return_back:
            print('-------------------some error in long context--------------------')
            print(return_back)
    else:
        code_i = code_i_initial
        cal_reward = 0

    return code_i, 0, cal_reward


def model_generate_topk(curr_node, max_tokens, temperature, num_return_sequences):
    assert num_return_sequences > 1
    pre_code = curr_node.task + curr_node.label
    prompt = curr_node.prompt
    messages = [
        {"role": "system",
         "content": f"I want you to become my Expert Python programmer. Your goal is to complete the code."},
        {"role": "user",
         "content": f"I first give you the previous code: {pre_code}" + f"you need to complete the code:" + f'{prompt}'}, ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    #input_ids = tokenizer(messages, return_tensors='pt').to(model.device)  # gemma2
    #outputs = model.generate(**input_ids, max_new_tokens=max_tokens, eos_token_id=terminators, temperature=temperature,
    #                         num_return_sequences=num_return_sequences, do_sample=True,
    #                         output_scores=True, return_dict_in_generate=True)
    outputs = model.generate(input_ids, max_new_tokens=max_tokens, eos_token_id=terminators, temperature=temperature,
                             num_return_sequences=num_return_sequences, do_sample=True,
                             output_scores=True, return_dict_in_generate=True)

    code = []
    errors = []

    # compute the scores using compute_transition_scores()
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    # -inf is the PAD
    # Replace inf and -inf with 0 or any other value
    transition_scores = torch.where(torch.isinf(transition_scores), torch.tensor(0.0), transition_scores)
    length_penalty = 1.0
    #output_length = input_ids.input_ids.shape[1] + np.sum(transition_scores.cpu().float().numpy() < 0, axis=1)
    output_length = input_ids.shape[1] + np.sum(transition_scores.cpu().float().numpy() < 0, axis=1)
    reconstructed_scores = np.exp(transition_scores.sum(axis=1).cpu().float().numpy() / (output_length ** length_penalty))
    print('topk scores:', reconstructed_scores)
    for i in range(num_return_sequences):
        response = outputs['sequences'][i]
        #response = response[input_ids.input_ids.shape[-1]:]
        response = response[input_ids.shape[-1]:]
        code_i = tokenizer.decode(response, skip_special_tokens=True)
        print('pre topk coding:', code_i)
        extract_code_i, error = extract_code_split(code_i, curr_node.label)
        print('topk error:', error)
        print('topk coding:', extract_code_i)

        if extract_code_i:
            code.append(extract_code_i)
        else:
            errors.append(error)

    return code, reconstructed_scores, errors


def main():
    max_rollouts = 2
    top_k = 3
    beam_width = 3
    # hyperparameters for P-UCB function
    c_base = 10
    c = 4

    class Node:
        id_iter = itertools.count()

        def __init__(self, logprob, label, prompt, task, parent, value=0, error=[]):
            self.value = value  # total reward obtainable from node
            self.prob = exp(logprob)  # necessary for P-UCB calculation
            self.prompt = prompt
            self._children = []
            self._parent = parent
            self.visits = 0
            # attributes for graph visualisation
            self.id = next(self.id_iter)
            self.label = label  # for current subtask
            self.task = task  # for whole task
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

    # Implements P-UCB heuristic as defined in https://arxiv.org/pdf/2303.05510.pdf#subsection.D.1
    # P-UCB-SELECT(s, c) = argmax_a P-UCB(s, a)
    # -> where P-UCB(s, a) = Q(s, a) + ß(s) * P(a|s) * √log(s.visits) / (1 + s'.visits)
    # -> where ß(s) = log((s.visits + c_base + 1) / c_base) + c
    # -> c_base & c are hyperparameters, set to values c_base = 10 & c = 4
    def p_ucb_select(parent_node, child_nodes):
        s_visits = parent_node.visits
        beta = log((s_visits + c_base + 1) / c_base) + c

        max_p_ucb = -inf
        max_node = None
        for i in range(len(child_nodes)):
            node = child_nodes[i]
            p_ucb = node.value + beta * node.prob * sqrt(log(s_visits)) / (
                    1 + node.visits
            )
            print('-----------------------------------------selecting node---------------------------------')
            print(node.label)
            print(p_ucb)
            node.p_ucb = p_ucb  # store most recent p_ucb for visualisation
            if p_ucb > max_p_ucb:
                max_node = node
                max_p_ucb = p_ucb
        return max_node

    def get_top_k_tokens(curr_node, k):
        temperature = random.uniform(0.7, 1)
        max_tokens = random.randint(128, 256)
        tokens, probs, errors = model_generate_topk(curr_node=curr_node, max_tokens=max_tokens, temperature=temperature,
                                                    num_return_sequences=2)
        return tokens, probs, errors

    def beam_search(curr_node, tokens):
        """
        Returns the full generation with both prompt + completion concatenated.
        Original prompt needs to be indexed out to get the actual generated program.
        """
        print('-----------------------------beam search------------------------')
        temperature = random.uniform(0.7, 1)
        max_tokens = random.randint(256, 512)
        tokens, probs, cal_reward = model_generate_long_context(curr_node, completions=tokens, max_tokens=max_tokens,
                                                    temperature=temperature, num_return_sequences=beam_width)
        return tokens, cal_reward

    def calculate_reward(precode, completion):
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
        if code_len == 0:
            return 0
        elif all(code_i in precode for code_i in unique_lines):
            print('**************************task completed!!*******************************888')
            return 1
        else:
            success_len = 0
            for code_i in unique_lines:
                precode = precode + code_i
                return_back = monitor_process(precode, MAX_EXECUTION_TIME, MAX_MEMORY_USAGE)
                if not return_back:
                    success_len += 1
            return success_len / code_len

    # check if a generated program exists for a given node state and return reward if found
    def match_cached_programs(prefix, program_dict):
        for program, reward in program_dict.items():
            if program.startswith(prefix):
                return reward
        return -1

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

    def check_child_nodes(task_code, subtask_code, extract_code):
        source = [code for code in [task_code, subtask_code, extract_code] if code != '']
        source = ''.join(source)
        no_name = []
        no_attribute = {}
        others = []
        print('-------------------------------checking child nodes -------------------------')
        print('-------------curr_node.task------------------')
        print(task_code)
        print('---------------curr_node.label----------------')
        print(subtask_code)
        print('-----------------generated token------------------')
        print(extract_code)

        # code pass from interpreter
        if not source.endswith(':\n'):
            try:
                if source.endswith('.'):
                    source = source[:-1]
                elif source.endswith(' \\\n'):
                    source = source.replace(' \\\n', '')
                ast.parse(source)
                if 'exit()' in source:
                    source.replace('exit()', 'pass')

                return_back = monitor_process(source, MAX_EXECUTION_TIME, MAX_MEMORY_USAGE)

                if return_back:
                    print('------------------------some error in checked child node----------------------------')
                    print(return_back)

                    if 'NameError' in return_back:
                        var = return_back.split("name '")[1].split("'")[0]
                        no_name.append(var)
                    elif 'AttributeError' in return_back:
                        obj = return_back.split("module '")[1].split("'")[0]
                        att = return_back.split("attribute '")[1].split("'")[0]
                        #print(python.run('dir(obj)'))
                        # complete using jedi
                        script = jedi.Script(source)
                        cliped_source = source.split('at')[0] + '\n'
                        multilines = cliped_source.splitlines()
                        jedi_completion = script.complete(line=len(multilines) - 1, column=len(multilines[-1]))
                        jedi_completion = [com.name for com in jedi_completion]
                        #print(jedi_completion)
                        no_attribute[obj] = att + jedi_completion
                    elif ('error' in return_back) or ('Error' in return_back):
                        others.append(return_back)
            except Exception as e:
                others.append(e)

        return no_name, no_attribute, others


    task_path = '/workspace/comparison_bench/cibench-exec.jsonl'
    # this is for cibench
    exec_path = os.path.dirname(task_path)
    base_dir = '/workspace/RAG_bench/ds1000/'
    task_name = os.path.basename(task_path).split('.')[0]
    output_doc = '/workspace/comparison_bench/sequential_task/cibench-exec_mcts.jsonl'

    tasks = []
    ori_task_id = 0
    with jsonlines.open(task_path, mode='r') as eval_reader:
        for sample in eval_reader:
            if 'pre_code' in sample:
                tasks.append({'task': sample['prompt'], 'code': sample['code'], 'exlib': sample['exlib'], 'pre_task': sample['pre_code'], 'ori_task_id': ori_task_id})
            else:
                _pre_task, _ = extract_code_split(sample['prompt'], '')
                tasks.append({'task': sample['prompt'], 'code': sample['code'], 'exlib': sample['exlib'], 'pre_task': _pre_task, 'ori_task_id': ori_task_id})
            ori_task_id += 1

    # cut tasks according to the output
    len_tasks = len(tasks)
    if os.path.exists(output_doc):
        with open(output_doc, 'r', encoding='utf-8') as file:
            # Read all lines
            lines = file.readlines()
            if lines:
                # Get the last line
                last_line = lines[-1]
                # Parse the JSON object
                try:
                    last_json = json.loads(last_line)
                    current_line = last_json['ori_task_id']
                except:
                    current_line = len_tasks
                    time.sleep(100)
    else:
        current_line = len_tasks

    tasks = tasks[0:current_line]

    start = time.perf_counter()
    num_iter = 1
    # processing and add item dynamically
    task_id = current_line
    repeated_task_num = 0
    while tasks:
        reward = None
        restart_flag = False
        # print tasks
        print('----------------------start task---------------------------------')
        current_task = tasks.pop()  # select the last one
        if 'code' in current_task:
            code = current_task['code']
        else:
            code = None
        if 'pre_task' in current_task:
            pre_tasks = current_task['pre_task']
        else:
            pre_tasks = ''
        if 'ori_task_id' in current_task:
            ori_task_id = current_task['ori_task_id']

        prompt = current_task['task'].replace('prompt', '').replace('{', '').replace('}', '').replace(':', '').replace("''", '')
        if ('data/' in prompt) and (exec_path + '/data/' not in prompt):
            prompt = prompt.replace('data/', exec_path + '/data/')
        '''
        # --------------------------------------------RAG----------------------------
        if 'exlib' in current_task:
            library = current_task['exlib']
            rag_dir_list = [base_dir + lib for lib in library]
            rag_dir_list_exist = all(os.path.exists(rag_dir) for rag_dir in rag_dir_list)
            rag_dir_list_exist = False
        else:
            rag_dir_list_exist = False
        if rag_dir_list_exist:
            rag_context = []
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
        print(repr(prompt))
        # cache of generated programs => rewards
        program_dict = {}
        num_rollouts = max_rollouts
        root = Node(log(1), '', prompt, pre_tasks, None)
        test_times = [0, ]
        # graph snapshots for web visualisation
        nodes, edges = {root.id: root}, {}
        graph_dict = {}
        origin_node = root
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
            # selection
            while len(curr_node._children) > 0:
                for child in curr_node._children:
                    nodes[child.id] = child
                    edges[(curr_node.id, child.id)] = True
                curr_node = p_ucb_select(curr_node, curr_node._children)
                graph_dict[i]["selectedNodes"].append(curr_node.id)
                curr_node.visits += 1

            print('------------------------curr_node---------------------')
            print('-----------------curr_node.task---------------')
            print(curr_node.task)
            print('--------------curr_node.prompt-----------------')
            print(curr_node.prompt)

            if curr_node.value == -1:
                _error, _updated_prompt = enrich_prompt(curr_node)
                _added_prompt = f'Now we do first step: {_error} using prompt {_updated_prompt}'
                _new_prompt = _added_prompt[0:4000] + f'\n Then second step is  {curr_node.prompt[0:4000]}'
                # update the prompt
                curr_node.prompt = _new_prompt

            # expansion
            print('---start to expand using this information------')
            # print(repr(curr_node.task), repr(curr_node.prompt))
            tokens, probs, top_k_errors = get_top_k_tokens(curr_node, top_k)
            # check if there is undefined variable and function otherwise filter the child nodes or cut it and start again
            child_nodes = []
            if tokens:
                for (token, prob) in zip(tokens, probs):
                    no_name, no_attribute, others = check_child_nodes(curr_node.task, curr_node.label, token)
                    error_list = [no_name, no_attribute, others, token]
                    print(error_list)
                    if not no_name and not no_attribute and not others:
                        child_nodes.append(Node(prob, curr_node.label + token[0:4000], curr_node.prompt, curr_node.task, curr_node, value=0))
                    else:
                        child_nodes.append(Node(prob, curr_node.label + token[0:4000], curr_node.prompt, curr_node.task, curr_node, value=-1, error=error_list))

            elif repeated_task_num > 1:  # cannot continue since no child node
                # if too many times repeated, give up this task
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~ FAILD in UPDATING~~~~~~~~~~~~~~~~~~~~~~~~~~~~', repeated_task_num)
                # failed on current task
                completion, cal_reward = beam_search(origin_node, '')
                test_start = time.perf_counter()
                if cal_reward:
                    reward = calculate_reward(curr_node.task + curr_node.label, completion)
                else:
                    reward = -1
                test_end = time.perf_counter()
                test_times.append(test_end - test_start)
                program_dict[completion] = reward
                graph_dict[i]["state"] = curr_node.task + curr_node.label
                graph_dict[i]["completion"] = completion
                graph_dict[i]["reward"] = reward
                break

            else:
                print('++++++++++++++++++++++++++++++++++++++stuck in updating+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                completions = [str(idx) + ':' + completion[0:500] if completion is not None else 'None' for idx, completion in enumerate(top_k_errors)]
                completions = ','.join(completions)
                if 'None' in completions and '0' in completions:
                    fix_error = f'\navoiding these errors: {completions}'
                    current_task['task'] = curr_node.prompt[0:4000] + fix_error[0:4000]
                else:
                    current_task['task'] = curr_node.prompt[0:4000]
                tasks.append(current_task)
                restart_flag = True
                repeated_task_num = repeated_task_num + 1
                break

            curr_node._children = child_nodes
            for child in child_nodes:
                nodes[child.id] = child
                edges[(curr_node.id, child.id)] = True

            # evaluation
            if not reward:
                reward = match_cached_programs(curr_node.label, program_dict)
            # only run generation if node state not found in cached programs
            if reward == -1:
                print('final reward evaluation!!!')
                completion, cal_reward = beam_search(curr_node, tokens)
                test_start = time.perf_counter()
                if cal_reward:
                    reward = calculate_reward(curr_node.task + curr_node.label, completion)
                test_end = time.perf_counter()
                test_times.append(test_end - test_start)
                program_dict[completion] = reward
                graph_dict[i]["state"] = curr_node.task + curr_node.label
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
            try:
                json.dump(graph_dict, f, cls=NodeEncoder)
            except:
                # 处理字典
                graph_dict = make_serializable(graph_dict)
                json.dump(graph_dict, f, cls=NodeEncoder)


    end = time.perf_counter()
    print(f"Total elapsed time: {(end - start):.4f}s\n")


# necessary to prevent multiple executions of main() within stats_execute threads
if __name__ == "__main__":
    main()
