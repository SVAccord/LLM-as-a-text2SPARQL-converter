#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 

@author: Sergeev

code for using finetuned llm as text to sparql converter

place where the model is:
base_model = Path.cwd() + "//model"

question - it is text to convert
model.generate() - llm usage
answer - contains the result of llm work - sparql query 
"""

import torch
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
gpu_name = pynvml.nvmlDeviceGetName(handle)
gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**2)
print("torch ", torch.__version__)
print(f"GPU: {gpu_name}, {gpu_mem} MiB")
print(f"{torch.cuda.is_available() = }")
#if torch.cuda.is_available():
#torch.cuda.empty_cache()
print(torch.version.cuda)

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import sentence_transformers 
from sentence_transformers import SentenceTransformer, util

from pathlib import Path
import re
import pandas as pd
from magic_timer import MagicTimer
from pynvml import *
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    
    
def process_note(note):
    """
    Обрабатывает строку note. 
    1) извлекает текст перед скобкой '{'
    2) извлекает текст между скобками '{' и '}' верхнего уровня, включая сами скобки.
    3) а также текст за скобками, если он соответствует одному из заданных шаблонов.
    Нужна для очистки результата генерации модели
    """

    patterns = [
        r"GROUP BY \?factor\s*ORDER BY DESC\(\?count\)\s*LIMIT\s*\d+\s*",
        r"GROUP BY \?factor\s*ORDER BY DESC\(\?count\)",
        r"GROUP BY \?factor\s*",
        r"GROUP BY \?factor\s*\?year\s*ORDER BY \?year",
        r"GROUP BY \?pub\s*HAVING\(COUNT\(DISTINCT \?factor\)\s*>\s*\d+\s*AND COUNT\(DISTINCT \?org\)\s*>\s*\d+\s*\)",
        r"GROUP BY \?pub\s*HAVING\(COUNT\(DISTINCT \?factor\)\s*>\s*\d+\)",
        r"GROUP BY \?pub\s*ORDER BY DESC\(\?citationCount\)\s*LIMIT\s*\d+\s*",
        r"GROUP BY \?pub\s*ORDER BY DESC\(\?citationCount\)",
        r"GROUP BY \?pub\s*",
        r"GROUP BY \?author\s*ORDER BY DESC\(\?count\)\s*LIMIT\s*\d+\s*",
        r"GROUP BY \?author\s*ORDER BY DESC\(\?count\)",
        r"GROUP BY \?author\s*",
        r"GROUP BY \?journal\s*ORDER BY DESC\(\?count\)\s*LIMIT\s*\d+\s*",
        r"GROUP BY \?journal\s*ORDER BY DESC\(\?count\)",
        r"GROUP BY \?journal\s*",
        r"GROUP BY \?conference\s*ORDER BY DESC\(\?count\)\s*LIMIT\s*\d+\s*",
        r"GROUP BY \?conference\s*ORDER BY DESC\(\?count\)",
        r"GROUP BY \?conference\s*",
        r"GROUP BY \?org\s*ORDER BY DESC\(\?count\)\s*LIMIT\s*\d+\s*",
        r"GROUP BY \?org\s*ORDER BY DESC\(\?count\)",
        r"GROUP BY \?org\s*",
        r"GROUP BY \?term\s*ORDER BY DESC\(\?count\)\s*LIMIT\s*\d+\s*",
        r"GROUP BY \?term\s*ORDER BY DESC\(\?count\)",
        r"GROUP BY \?term\s*",
        r"GROUP BY \?subdivision\s*ORDER BY DESC\(\?count\)\s*LIMIT\s*\d+\s*",
        r"GROUP BY \?subdivision\s*ORDER BY DESC\(\?count\)",
        r"GROUP BY \?subdivision\s*",
        r"GROUP BY \?person\s*ORDER BY DESC\(\?pubcount\)\s*LIMIT\s*\d+\s*",
        r"GROUP BY \?person\s*ORDER BY DESC\(\?pubcount\)",
        r"GROUP BY \?person\s*",
        r"ORDER BY DESC\(\?pubcount\)" 
    ]

    result = ""
    brace_level = 0
    start_index = -1
    first_brace_found = False
    i = 0

    while i < len(note):
        char = note[i]
        if char == '{':
            if brace_level == 0:
                if not first_brace_found:
                    result += note[:i]
                    first_brace_found = True
                start_index = i  
            brace_level += 1
            i += 1
        elif char == '}':
            brace_level -= 1
            if brace_level == 0 and start_index != -1:
                result += note[start_index:i+1]  
                start_index = -1  # Сбрасываем индекс
            i += 1

        elif brace_level == 0 and start_index == -1:
            # Проверяем, соответствует ли текст за скобками одному из шаблонов
            matched = False
            for pattern in patterns:
                match = re.match(pattern, note[i:])
                if match:
                    result += match.group(0)  
                    i += len(match.group(0))  
                    matched = True
                    break
            if not matched:
              i+=1
        else:
            i += 1

    return result


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True, # ?
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 # getattr(torch, "float16")
)

base_model = Path("model")

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    #load_in_4bit=True,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print_gpu_utilization()
  
term_use_4_ans = "Answer" #Answer Sparql
question = ""
question = "Покажи список сотрудников из организации :nio_5886 с числом публикаций более  39."
question = re.sub(r'\s+', ' ', question.replace("\n", " ") ).strip() 

prompt =f"""
      Task: Generate SPARQL queries to query the knowledge graph based on the provided schema definition.
      ### Question: {question}
      ### {term_use_4_ans}:    
    """

model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
print("model output is:")
model.eval()

with torch.no_grad():
    answer = tokenizer.decode(model.generate(**model_input, max_new_tokens=500, pad_token_id=2)[0], skip_special_tokens=True)
    answer = re.sub(r'\s+', ' ', answer.replace("\n", " ") ).strip()
    print(process_note(answer))
    
print_gpu_utilization()
