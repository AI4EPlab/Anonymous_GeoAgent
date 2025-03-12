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
#################################################################################################
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
#from python_tool import PythonInterpreter
from multi_process import monitor_process, make_serializable, is_serializable
# 设置监控的最大执行时间和内存使用百分比
MAX_EXECUTION_TIME = 10  # 秒
MAX_MEMORY_USAGE = 90  # 百分比
####################################
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

model_name_path = '/workspace/codegemma-7b'
model_name = 'unsloth/codegemma-7b'

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    cache_dir=model_name_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    #device_map='cuda',
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
# alpaca_prompt = You MUST copy from above!
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]


def extract_code_split(text, pre_code):
    new_text = []
    add_flag = False
    if '```' in text:
        for text_i in text.splitlines(keepends=True):
            if add_flag and '```' not in text_i:
                new_text.append(text_i)
            if '```' in text_i:
                add_flag = not add_flag
        text = ''.join(new_text)

    #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55')
    #print('text', text)
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

    #print('generated text!!!!!!!!!!!!!!!!!', code_block)
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
        # remove the precode from text
        if (pre_code not in ['\n', ' ']) and (selected_code_block != pre_code):
            selected_code_block = selected_code_block.replace(pre_code, '')
        error = None
    else:
        selected_code_block = ''
        error = text
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
        message_prompt = '''### Instruction: {}  ### Input: {}  ### Response: {}'''.format("You gola is to fix the missing variables in the initial prompt.",
                                                                                           f"I first give you the Python code: {source}" + f"\nComplete the <MISSING> place with a exact {no_name[0]}: ",
                                                                                           " ")
        error = f'define {no_name[0]}'
    elif no_attribute:
        source = curr_node.task + curr_node.label + token.replace(list(no_attribute.values())[0], '<MISSING>')
        message_prompt = '''### Instruction: {}  ### Input: {}  ### Response: {}'''.format("You gola is to fix the missing function in the initial prompt",
                                                                                           f"I first give you the Python code: {source}" + f"\nGive your solution in <MISSING> place: ",
                                                                                           " ")
        error = f'replace {list(no_attribute.values())[0]}'
    elif others:
        source = curr_node.task + curr_node.label + token
        message_prompt = '''### Instruction: {}  ### Input: {}  ### Response: {}'''.format("You gola is to fix the error in the initial prompt.",
                                                                                           f"I first give you the Python code: {source}" + f"\nGive your solution for fixing the error {others[0]}: ",
                                                                                           " ")
        error = f'fix {others[0]}'
    else:
        error = None
        message_prompt = None

    if message_prompt:
        message_prompt = message_prompt[0:2000]
        #input_prompt_ids = tokenizer.apply_chat_template(message_prompt, add_generation_prompt=True, return_tensors="pt").to(model.device)  # llama3.1
        input_prompt_ids = tokenizer(message_prompt, return_tensors='pt').to(model.device)  # gemma2
        prompt_outputs = model.generate(**input_prompt_ids, max_new_tokens=256, eos_token_id=terminators, do_sample=True,
                                        temperature=0.9, top_p=0.9, repetition_penalty=1.1)
        prompt_response = prompt_outputs[0][input_prompt_ids.input_ids.shape[-1]:]
        prompt_response = tokenizer.decode(prompt_response, skip_special_tokens=True)
        print('----------------------enriched prompt--------------------------------')
        # print(token)
        # print('---------------------response-------------------')
        print(prompt_response)
    else:
        prompt_response = None

    return error, prompt_response


# https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores.example
def model_generate_long_context(curr_node, completions, max_tokens, temperature, num_return_sequences):
    pre_code = curr_node.task + curr_node.label
    prompt = curr_node.prompt
    assert num_return_sequences > 1
    completions = [str(idx) + ':' + completion if completion is not None else 'None' for idx, completion in enumerate(completions)]
    completions = ','.join(completions)
    messages =  '''### Instruction: {}  ### Input: {}  ### Response: {}'''.format("I want you to become my Expert Python programmer. Your goal is to complete the code.",
                                                                                           f"I first give you the previous code: {pre_code}" + f'The task is {prompt}' + f"\nhere are two suggested code: {completions}" + f"\nif the task is completed by previous code, just give the same output as the previous code otherwise give your answer:",
                                                                                           " ")
    messages = messages[0:2000]
    #input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    input_ids = tokenizer(messages, return_tensors='pt').to(model.device)  # gemma2
    outputs = model.generate(**input_ids, max_new_tokens=max_tokens, eos_token_id=terminators, temperature=temperature,
                             output_scores=True, return_dict_in_generate=True)
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    # -inf is the PAD
    # Replace inf and -inf with 0 or any other value
    transition_scores = torch.where(torch.isinf(transition_scores), torch.tensor(0.0), transition_scores)
    # output_length = np.sum(transition_scores.cpu().float().numpy() < 0, axis=1)
    # length_penalty = model.generation_config.length_penalty
    length_penalty = 1.0
    # assert length_penalty == 1.0
    output_length = input_ids.input_ids.shape[1] + np.sum(transition_scores.cpu().float().numpy() < 0, axis=1)
    reconstructed_scores = np.exp(
        transition_scores.sum(axis=1).cpu().float().numpy() / (output_length ** length_penalty))

    response = outputs['sequences'][0]
    response = response[input_ids.input_ids.shape[-1]:]
    code_i = tokenizer.decode(response, skip_special_tokens=True)
    code_i, error = extract_code_split(code_i, curr_node.label)
    if code_i:
        code_snippet = pre_code + code_i
        return_back = monitor_process(code_snippet, MAX_EXECUTION_TIME, MAX_MEMORY_USAGE)
        if return_back:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!some error inside!!!---long context----')
            print(return_back)

    return code_i, reconstructed_scores


def model_generate_topk(curr_node, max_tokens, temperature, num_return_sequences):
    assert num_return_sequences > 1
    print('------------------topkkkk----------------------------------')
    pre_code = curr_node.task + curr_node.label
    prompt = curr_node.prompt
    messages = '''### Instruction: {}  ### Input: {}  ### Response: {}'''.format("I want you to become my Expert Python programmer. Your goal is to complete the code.",
                                                                                 f"I first give you the previous code: {pre_code}" + f"you need to complete the code:" + f'{prompt}',
                                                                                 " ")
    messages = messages[0:2000]
    #input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    input_ids = tokenizer(messages, return_tensors='pt').to(model.device)  # gemma2
    outputs = model.generate(**input_ids, max_new_tokens=max_tokens, eos_token_id=terminators, temperature=temperature,
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
    #output_length = np.sum(transition_scores.cpu().float().numpy() < 0, axis=1)
    #length_penalty = model.generation_config.length_penalty
    length_penalty = 1.0
    #assert length_penalty == 1.0
    output_length = input_ids.input_ids.shape[1] + np.sum(transition_scores.cpu().float().numpy() < 0, axis=1)
    reconstructed_scores = np.exp(transition_scores.sum(axis=1).cpu().float().numpy() / (output_length ** length_penalty))
    print('topk scores:', reconstructed_scores)
    #time.sleep(100)
    for i in range(num_return_sequences):
        response = outputs['sequences'][i]
        response = response[input_ids.input_ids.shape[-1]:]
        code_i = tokenizer.decode(response, skip_special_tokens=True)
        print('pre topk coding:', code_i)
        extract_code_i, error = extract_code_split(code_i, curr_node.label)
        print('topk error:', error)
        print('topk coding:', extract_code_i)

        #time.sleep(50)
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
            node.p_ucb = p_ucb  # store most recent p_ucb for visualisation
            if p_ucb > max_p_ucb:
                max_node = node
                max_p_ucb = p_ucb
        return max_node

    def get_top_k_tokens(curr_node, k):
        temperature = random.uniform(0.7, 1)
        max_tokens = random.randint(512, 600)
        tokens, probs, errors = model_generate_topk(curr_node=curr_node, max_tokens=max_tokens, temperature=temperature,
                                                    num_return_sequences=2)
        return tokens, probs, errors

    def beam_search(curr_node, tokens):
        """
        Returns the full generation with both prompt + completion concatenated.
        Original prompt needs to be indexed out to get the actual generated program.
        """
        print('-----------------------------beam search------------------------')
        temperature = random.uniform(0.5, 1)
        max_tokens = random.randint(512, 600)
        tokens, probs = model_generate_long_context(curr_node, completions=tokens, max_tokens=max_tokens,
                                                    temperature=temperature, num_return_sequences=beam_width)
        return tokens

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
            print('***************************************************task completed!!********************************************************888')
            return 1
        else:
            success_len = 0
            for code_i in unique_lines:
                precode = precode + code_i
                return_back = monitor_process(precode, MAX_EXECUTION_TIME, MAX_MEMORY_USAGE)
                if not return_back:
                    success_len += 1
            return success_len / code_len, success_len

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
            if reward > max_reward:
                best_program = program
                reward = max_reward
        return best_program

    def check_child_nodes(task_code, subtask_code, extract_code):
        #python = PythonInterpreter(globals=globals(), locals=None)
        source = [code for code in [task_code, subtask_code, extract_code] if code != '']
        source = ''.join(source)
        no_name = []
        no_attribute = {}
        others = []

        # code pass from interpreter
        if not source.endswith(':\n'):
            try:
                if source.endswith('.'):
                    source = source[:-1]
                elif source.endswith(' \\\n'):
                    source = source.replace(' \\\n', '')
                    print(source)
                ast.parse(source)
                print('i am stucked 1')
                print(source)
                if 'exit()' in source:
                    source.replace('exit()', 'pass')

                #return_back = python.run(source)
                return_back = monitor_process(source, MAX_EXECUTION_TIME, MAX_MEMORY_USAGE)

                print('i am stucked 2')
                if return_back:
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!some error inside!!!----check child node')
                    print(return_back)
                    #time.sleep(10)
                    if 'NameError' in return_back:
                        var = return_back.split("name '")[1].split("'")[0]
                        no_name.append(var)
                    elif 'AttributeError' in return_back:
                        obj = return_back.split("module '")[1].split("'")[0]
                        att = return_back.split("attribute '")[1].split("'")[0]
                        print(python.run('dir(obj)'))
                        # complete using jedi
                        script = jedi.Script(source)
                        cliped_source = source.split('at')[0] + '\n'
                        multilines = cliped_source.splitlines()
                        jedi_completion = script.complete(line=len(multilines) - 1, column=len(multilines[-1]))
                        jedi_completion = [com.name for com in jedi_completion]
                        print(jedi_completion)
                        no_attribute[obj] = att
                    elif ('error' in return_back) or ('Error' in return_back):
                        others.append(return_back)
            except Exception as e:
                others.append(e)

        return no_name, no_attribute, others


    task_path = '/workspace/comparison_bench/ds1000-exec.jsonl'
    # this is for cibench
    exec_path = os.path.dirname(task_path)
    base_dir = '/workspace/RAG_bench/ds1000/'
    task_name = os.path.basename(task_path).split('.')[0]
    output_doc = '/workspace/comparison_bench/sequential_task/ds1000-exec_mcts.jsonl'

    tasks = []
    ori_task_id = 0
    with jsonlines.open(task_path, mode='r') as eval_reader:
        for sample in eval_reader:
            if 'pre_code' in sample:
                tasks.append({'task': sample['prompt'], 'code': sample['code'], 'exlib': sample['exlib'], 'pre_task': sample['pre_code'], 'ori_task_id': ori_task_id})
            else:
                tasks.append({'task': sample['prompt'], 'code': sample['code'], 'exlib': sample['exlib'], 'pre_task': '', 'ori_task_id': ori_task_id})
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
        print('----------------------start check all tasks---------------------------------')
        print(tasks[-1])
        current_task = tasks.pop()  # select the last one
        if 'code' in current_task:
            code = current_task['code']
            #if 'tensorflow' in code:
            #    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!skip tensorflow')
            #    continue
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
                    if len(rag_output_text) > 2000:
                        rag_output_text = rag_output_text[0:2000]
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
        for i in range(max_rollouts):
            graph_dict[i] = {
                "selectedNodes": [root.id],
                "state": "",
                "completion": "",
                "reward": 0.0,
                "task_id": task_id,
                "childNodes": {},
                "success_len": 0,
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

            if curr_node.value == -1:
                _error, _updated_prompt = enrich_prompt(curr_node)
                _new_prompt = f'First step: {_error} using prompt {_updated_prompt}' + f'\n then {curr_node.prompt}'
                # update the prompt
                curr_node.prompt = _new_prompt

            # expansion
            print('---start to expand using this information------')
            # print(repr(curr_node.task), repr(curr_node.prompt))
            tokens, probs, top_k_errors = get_top_k_tokens(curr_node, top_k)
            # check if there is undefined variable and function otherwise filter the child nodes or cut it and start again
            child_nodes = []
            if tokens:
                for idx, (token, prob) in enumerate(zip(tokens, probs)):
                    # calculate reward of childNode
                    reward, success_len = calculate_reward(curr_node.task + curr_node.label, token)
                    graph_dict[i]["childNodes"][idx] = {
                        "id": idx,
                        "state": curr_node.task + curr_node.label,
                        "completion": token,
                        "success_len": success_len,
                    }
                    no_name, no_attribute, others = check_child_nodes(curr_node.task, curr_node.label, token)
                    error_list = [no_name, no_attribute, others, token]
                    print(error_list)
                    if not no_name and not no_attribute and not others:
                        child_nodes.append(Node(prob, curr_node.label + token, curr_node.prompt, curr_node.task, curr_node, value=0))
                    else:
                        child_nodes.append(Node(prob, curr_node.label, curr_node.prompt, curr_node.task, curr_node, value=-1, error=error_list))

            elif repeated_task_num > 1:  # cannot continue since no child node
                # if too many times repeated, give up this task
                print('repeated_task_num!!!!!!!!!', repeated_task_num)
                # failed on current task
                completion = beam_search(curr_node, top_k_errors)
                test_start = time.perf_counter()
                reward, success_len = calculate_reward(curr_node.task + curr_node.label, completion)
                test_end = time.perf_counter()
                test_times.append(test_end - test_start)
                program_dict[completion] = reward
                graph_dict[i]["state"] = curr_node.task + curr_node.label
                graph_dict[i]["completion"] = completion
                graph_dict[i]["reward"] = reward
                graph_dict[i]["success_len"] = success_len
                break

            else:
                completions = [str(idx) + ':' + completion[0:1000] if completion is not None else 'None' for idx, completion in enumerate(top_k_errors)]
                completions = ','.join(completions)
                current_task['task'] = prompt +  f'\navoiding these errors: {completions}'
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
                completion = beam_search(curr_node, tokens)
                print(completion)
                if completion:
                    test_start = time.perf_counter()
                    reward, success_len = calculate_reward(curr_node.task + curr_node.label, completion)
                    test_end = time.perf_counter()
                    test_times.append(test_end - test_start)
                    program_dict[completion] = reward
                    graph_dict[i]["state"] = curr_node.task + curr_node.label
                    graph_dict[i]["completion"] = completion
                    graph_dict[i]["reward"] = reward
                    graph_dict[i]["success_len"] = success_len
                else:
                    test_start = time.perf_counter()
                    time.sleep(1)
                    test_end = time.perf_counter()
                    test_times.append(test_end - test_start)
                    program_dict[completion] = reward
                    graph_dict[i]["state"] = curr_node.task + curr_node.label
                    graph_dict[i]["completion"] = completion


            graph_dict[i]["nodes"] = list(nodes.values())
            graph_dict[i]["edges"] = list(edges.keys())

            # backprop
            curr_node.backprop(reward)

            if reward == 1:
                num_rollouts = i + 1
                break

        if restart_flag:
            continue

        print('------------------------------come to the results--------------------------:', ori_task_id)
        if reward != 1:
            print('!!!!!!!!!!!!!!!!!!!The task is failed!!!!!!!!!!!!!!!!!!!!!')
            # 初始化最大值和对应的图 ID
            max_success_len = -1
            max_graph_id = None
            max_completion = ""

            # 遍历 graph_dict
            for graph_id, graph_data in graph_dict.items():
                # 检查当前图自身的 success_len
                graph_success_len = graph_data["success_len"]
                if graph_success_len > max_success_len:
                    max_success_len = graph_success_len
                    max_graph_id = graph_id
                    max_completion = graph_data["completion"]

                # 遍历子图
                for child_id, child_data in graph_data["childNodes"].items():
                    # 获取子图的 success_len
                    success_len = child_data["success_len"]
                    # 更新最大值和对应的图 ID
                    if success_len > max_success_len:
                        max_success_len = success_len
                        max_graph_id = graph_id
                        max_completion = child_data["completion"]

            # 输出结果
            if max_graph_id is not None:
                print(
                    f"包含子图中 success_len 最大的图 ID: {max_graph_id}, success_len: {max_success_len}, completion: {max_completion}")
            else:
                print("没有找到子图。")

        repeated_task_num = 0
        best_completion = get_best_program(program_dict)
        end = time.perf_counter()
        item = dict(
            task_id=task_id,
            ori_task_id=ori_task_id,
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
