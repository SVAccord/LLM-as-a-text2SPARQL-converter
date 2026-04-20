#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 

@author: sergeev
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

from vllm import LLM, SamplingParams

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def save_list_to_txt(my_list, filename):
  """
  Сохраняет содержимое списка в текстовый файл.
  """
  try:
    with open(filename, 'w', encoding='utf-8') as file:
      for item in my_list:
        file.write(str(item) + '\n')  
  except Exception as e:
    print(f"Ошибка при сохранении списка в файл: {e}")

# tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5-16k")
# model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5-16k")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True, # ?
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 # getattr(torch, "float16")
)

RUN_DIR = Path("/home/sergeev/PythonScripts/tests/qwen2.5_7b-i/250425_16ep")
DATA_DIR = str(RUN_DIR)
CHECKPOINT_DIR = RUN_DIR / "training_checkpoints"
MODEL_DIR = str(RUN_DIR) + "//model"
base_model = MODEL_DIR

header = "autoTrue4btruebnbBf16"
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
model = LLM(model=base_model)
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
#model = AutoModelForCausalLM.from_pretrained(
#    base_model,
#    quantization_config=bnb_config,
#    load_in_4bit=True,
#    device_map={"": 0}
#)
#model.config.use_cache = False
#model.config.pretraining_tp = 1

#tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
#tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"
print_gpu_utilization()


max_seq_length = 500          
test_dataset = load_dataset("csv", data_files= str(DATA_DIR) + '//QA-SPARQL_v19_test_test20.csv')
test_dataset = test_dataset["train"]
print("len(test_dataset) ", len(test_dataset))
print(test_dataset)


with MagicTimer() as timer:

    #test on train data    
    term_use_4_ans = "Answer" #Answer Sparql
    num_of_example = 0
    questions = []
    answers = []
    answers_b = []
    start_index = 0
    for num_of_example in range(0+start_index,len(test_dataset)):
        print("num_of_example", num_of_example)
        test_dataset[num_of_example]
        question = test_dataset['user request'][num_of_example]
        question = re.sub(r'\s+', ' ', question.replace("\n", " ") ).strip() 
    
        prompt =f"""
              Task: Generate SPARQL queries to query the knowledge graph based on the provided schema definition.
              ### Question: {question}
              ### {term_use_4_ans}:    
            """
        pr_len = len(re.sub(r'\s+', ' ',prompt.replace("\n", " ") ).strip()) #before insert the question or after?
        print("promt is:", re.sub(r'\s+', ' ',prompt.replace("\n", " ") + test_dataset['query'][num_of_example].replace("\n", " ") ).strip()) 
        questions.append(re.sub(r'\s+', ' ',prompt.replace("\n", " ") + test_dataset['query'][num_of_example].replace("\n", " ") ).strip()) 
        phrase_len = len(re.sub(r'\s+', ' ', prompt.replace("\n", " ") + test_dataset['query'][num_of_example].replace("\n", " ") ).strip()) 
        #ans_len = phrase_len-pr_len
        print("phrase_len", phrase_len)

        #model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
model_input = prompt
outputs = llm.generate(model_input, sampling_params)
        print("model output is:")
        #model.eval()
prompt_ = outputs.prompt
        generated_text = outputs.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        
        #with torch.no_grad():
            #answer = tokenizer.decode(model.generate(**model_input, max_new_tokens=max_seq_length, pad_token_id=2)[0], skip_special_tokens=True)
           answer = re.sub(r'\s+', ' ', generated_text.replace("\n", " ") ).strip()
           answer = answer[0:phrase_len+1]
           print(answer)
           answers.append(answer[pr_len:phrase_len+1])
            
        answers_b.append(answers[num_of_example-start_index].strip() == re.sub(r'\s+', ' ', test_dataset['query'][num_of_example].replace("\n", " ") ).strip())
        print(answers_b[num_of_example-start_index])
        save_list_to_txt(answers_b, "answers_b.txt")
    print_gpu_utilization()
    
    table = pd.DataFrame(list(zip(questions, answers, answers_b)))
    quality = table[2].values.sum()
    table.loc['quality'] = [quality, quality*100/20,0]
    table.to_excel('250501_test20_'+header +'.xlsx', engine="xlsxwriter", index=False)
    print("quality ", quality)

print(f"evaluated in {timer}.") # 30 tasks in evaluated in 29 minutes

nvmlInit()
handle_0 = nvmlDeviceGetHandleByIndex(0)
handle_1 = nvmlDeviceGetHandleByIndex(1)
info_0 = nvmlDeviceGetMemoryInfo(handle_0)
info_1 = nvmlDeviceGetMemoryInfo(handle_1)
print(f"GPU0 memory occupied: {info_0.used//1024**2} MB.")
print(f"GPU1 memory occupied: {info_1.used//1024**2} MB.")

save_list_to_txt([info_0.used/1024**2, info_1.used/1024**2, timer, base_model, max_seq_length],"tech_info_pipe_"+header+".txt")



