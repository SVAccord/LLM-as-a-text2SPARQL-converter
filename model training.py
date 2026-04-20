# -*- coding: utf-8 -*-

"""
test on qwen2.5_7b 
2.6-2.9 it/sec
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import wandb
run = wandb.init(project="ft_vicuna-13b-v1.5-4k_QA-SPARQL_4bit", name="13_QA-SPARQL_v19_8ep")

import torch
print(torch.__version__)
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from accelerate import Accelerator
import pynvml
from pathlib import Path
import re
import pandas as pd
from magic_timer import MagicTimer

import sentence_transformers 
from sentence_transformers import SentenceTransformer, util

from pynvml import *
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
gpu_name = pynvml.nvmlDeviceGetName(handle)
gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**2)
print(handle)
print(f"GPU: {gpu_name}, {gpu_mem} MiB")
print(f"{torch.cuda.is_available() = }")
handle = pynvml.nvmlDeviceGetHandleByIndex(1)
gpu_name = pynvml.nvmlDeviceGetName(handle)
gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**2)
print(handle)
print(f"GPU: {gpu_name}, {gpu_mem} MiB")
print(f"{torch.cuda.is_available() = }")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#local_rank = os.getenv("LOCAL_RANK")
#device_string = "cuda:" + str(local_rank)


model_name = "lmsys/vicuna-13b-v1.5-16k"
NUM_EPOCHS = 4
USE_FLASH_ATTENTION_2 = False
RUN_DIR = Path.cwd()
DATA_DIR = RUN_DIR
CHECKPOINT_DIR = RUN_DIR / "training_checkpoints"
MODEL_DIR = str(RUN_DIR) + "//model"
max_seq_length = 500
print("max_seq_length: ", max_seq_length)

accelerator = Accelerator()  # Auto-detects GPU/CPU and distributes the load
device = accelerator.device  # 

print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          padding_side="right",  
                                          use_fast=True,
                                          trust_remote_code=True) 
tokenizer.pad_token = tokenizer.eos_token # 

header = "autoTrue4bNobnbBf16"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto", 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16,
    load_in_4bit = True,
    use_flash_attention_2=USE_FLASH_ATTENTION_2, # 
)

if USE_FLASH_ATTENTION_2:
    try:
        from flash_attn import __version__ as flash_attn_version
        print(f"Flash Attention 2 is enabled (version {flash_attn_version})")
    except ImportError:
        print("Flash Attention 2 is not installed.  Install it with: pip install flash-attn --no-build-isolation")
        USE_FLASH_ATTENTION_2 = False # Disable if not installed

print(f"Memory used before LoRA: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,  
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ] 
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() #


train_dataset = load_dataset("csv", data_files= str(DATA_DIR) + '//QA-SPARQL_v19_train.csv')
validation_dataset = load_dataset("csv", data_files= str(DATA_DIR) + '//QA-SPARQL_v19_val.csv')

print("len(train_dataset) ", len(train_dataset))
print("len(validation_dataset) ", len(validation_dataset))
train_dataset = train_dataset["train"]

validation_dataset = validation_dataset["train"]


train_dataset = train_dataset.map(lambda example: {'text': f"Task: Generate SPARQL queries to query the knowledge graph based on the provided schema definition. ### Question: " + re.sub(r'\s+', ' ', example['user request'].replace("\n", " ") ).strip() + f" ### Answer: " + re.sub(r'\s+', ' ', example['query'].replace("\n", " ") ).strip() })
validation_dataset = validation_dataset.map(lambda example: {'text': f"Task: Generate SPARQL queries to query the knowledge graph based on the provided schema definition. ### Question: " + re.sub(r'\s+', ' ', example['user request'].replace("\n", " ") ).strip() + f" ### Answer: " + re.sub(r'\s+', ' ', example['query'].replace("\n", " ") ).strip() })

training_params = TrainingArguments(
    output_dir=CHECKPOINT_DIR, 
    learning_rate=2e-4,
    warmup_ratio=0.1, # 0.4,
    lr_scheduler_type="cosine",   
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit", 
    #optim="adamw_torch",
    weight_decay=0.001, 
    fp16=not torch.cuda.is_bf16_supported(),       
    bf16=torch.cuda.is_bf16_supported(),
    save_steps=4000,
    save_total_limit=10,
    logging_steps=700,
    evaluation_strategy='steps',
    report_to="wandb",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    seed=1,
)

trainer = SFTTrainer(
    args=training_params,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    dataset_text_field="text", 
    peft_config=lora_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    
    #packing=False #True
)
with MagicTimer() as timer:
    trainer.train()#
    #result=trainer.train(resume_from_checkpoint=True)
print(f"Trained model in {timer}.")

#print_summary(result) # undefined 
trainer.model.save_pretrained(MODEL_DIR)
trainer.tokenizer.save_pretrained(MODEL_DIR)

def save_list_to_txt(my_list, filename):
  """
  """
  try:
    with open(filename, 'w', encoding='utf-8') as file:
      for item in my_list:
        file.write(str(item) + '\n')  
  except Exception as e:
    print(f"Р В Р’В Р »: {e}")

print_gpu_utilization()

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)
save_list_to_txt([info.used/1024**2, timer, model_name, max_seq_length],"tech_info.txt")

