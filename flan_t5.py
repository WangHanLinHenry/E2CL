# 先尝试训练一个T5的模型来看看效果
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda
device = 'cuda:0' if cuda.is_available() else 'cpu'
import json
import random

from rich.table import Column, Table
from rich import box
from rich.console import Console

from tqdm import tqdm

console=Console(record=True)

def display_df(df):
  """display dataframe in ASCII format"""

  console=Console()
  table = Table(Column("source_text", justify="center" ), Column("target_text", justify="center"), title="Sample Data",pad_edge=False, box=box.ASCII)

  for i, row in enumerate(df.values.tolist()):
    table.add_row(row[0], row[1])

  console.print(table)

training_logger = Table(Column("Epoch", justify="center" ), 
                        Column("Steps", justify="center"),
                        Column("Loss", justify="center"), 
                        title="Training Status",pad_edge=False, box=box.ASCII)


class YourDataSetClass(Dataset):

  def __init__(self, data, tokenizer, source_len, target_len):
    self.tokenizer = tokenizer
    self.data = data
    self.source_len = source_len
    self.summ_len = target_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    source_text = self.data[index]['input']
    target_text = self.data[index]['output']

    #cleaning data so as to ensure data is in string type
    source_text = ' '.join(source_text.split())
    target_text = ' '.join(target_text.split())

    source = self.tokenizer.batch_encode_plus([source_text], max_length= self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
    target = self.tokenizer.batch_encode_plus([target_text], max_length= self.summ_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

    source_ids = source['input_ids'].squeeze()
    source_mask = source['attention_mask'].squeeze()
    target_ids = target['input_ids'].squeeze()
    target_mask = target['attention_mask'].squeeze()

    return {
        'source_ids': source_ids.to(dtype=torch.long), 
        'source_mask': source_mask.to(dtype=torch.long), 
        'target_ids': target_ids.to(dtype=torch.long),
        'target_ids_y': target_ids.to(dtype=torch.long)
    }

def train(epoch, tokenizer, model, device, loader, optimizer):

  model.train()
  
  progress_bar = tqdm(total=len(loader), desc="Training Progress of epoch {}".format(epoch+1))

  total_loss = 0
  for _,data in enumerate(loader, 0):
    y = data['target_ids'].to(device, dtype = torch.long)
    y[y[: ,:] == 0 ] = -100

    ids = data['source_ids'].to(device, dtype = torch.long)
    mask = data['source_mask'].to(device, dtype = torch.long)
    outputs = model(input_ids = ids, attention_mask = mask, labels = y, return_dict = True)

    loss = outputs['loss']
    total_loss = total_loss + loss

    if _%1000==0:
      training_logger.add_row(str(epoch), str(_), str(total_loss/(_+1)))
      console.print(training_logger)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    progress_bar.update(1)

  progress_bar.close()

def T5Trainer(raw_data, model_params, output_dir="./outputs/" ):
  
  torch.manual_seed(model_params["SEED"]) # pytorch random seed
  np.random.seed(model_params["SEED"]) # numpy random seed
  torch.backends.cudnn.deterministic = True

  console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

  tokenizer = T5Tokenizer.from_pretrained("flan-t5-base")
  model = T5ForConditionalGeneration.from_pretrained("flan-t5-base")
  model = model.to(device)

  console.log(f"[Data]: Reading data...\n")

  training_set = YourDataSetClass(raw_data, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"])

  train_params = {
      'batch_size': model_params["TRAIN_BATCH_SIZE"],
      'shuffle': True,
      'num_workers': 0
      }

  training_loader = DataLoader(training_set, **train_params)
  optimizer = torch.optim.Adam(params =  model.parameters(), lr=model_params["LEARNING_RATE"])

  console.log(f'[Initiating Fine Tuning]...\n')

  for epoch in range(model_params["TRAIN_EPOCHS"]):
      train(epoch, tokenizer, model, device, training_loader, optimizer)

      model.save_pretrained(output_dir + f"/epoch_{epoch}")
      tokenizer.save_pretrained(output_dir + f"/epoch_{epoch}")

model_params={
    "MODEL":"flan-t5-base",             # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE":30,          # training batch size
    "VALID_BATCH_SIZE":30,          # validation batch size
    "TRAIN_EPOCHS":6,              # number of training epochs
    "VAL_EPOCHS":1,                # number of validation epochs
    "LEARNING_RATE":1e-4,          # learning rate
    "MAX_SOURCE_TEXT_LENGTH":1000,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH":8,   # max length of target text
    "SEED": 42                     # set seed for reproducibility 
}

# total_data = []

# with open('/home/hanlin/hlwang_projects/FYP/final_exp/few_data/planning_feedback_input_output.jsonl', "r") as file:
#     for line in file:
#         # 解析 JSON 对象
#         json_object = json.loads(line)
#         # 在此处进行处理
#         total_data.append(json_object)

# with open('/home/hanlin/hlwang_projects/FYP/final_exp/few_data/free_explo/correction/filter_1.jsonl', "r") as file:
#     for line in file:
#         # 解析 JSON 对象
#         json_object = json.loads(line)
#         # 在此处进行处理
#         total_data.append(json_object)

# with open('/home/hanlin/hlwang_projects/FYP/final_exp/few_data/teacher_explo/gen_sft/corrective_filter.jsonl', "r") as file:
#     for line in file:
#         # 解析 JSON 对象
#         json_object = json.loads(line)
#         # 在此处进行处理
#         total_data.append(json_object)

total_data = []
with open('', "r") as file:
    for line in file:
        # 解析 JSON 对象
        json_object = json.loads(line)
        # 在此处进行处理
        total_data.append(json_object)

T5Trainer(total_data, model_params=model_params, output_dir="outputs/")
