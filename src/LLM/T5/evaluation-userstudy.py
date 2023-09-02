from os.path import join
import json
import os
from tqdm import tqdm
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

model_path = './flan-t5-large'
max_source_length = 512
max_target_length = 512

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(join('output', 'win64-texts-actioncodes-texts', '1', 'checkpoint-best')).cuda()

def preprocess(example):
    model_inputs = tokenizer(example['content'], max_length=max_source_length, padding='max_length', truncation=True)
    model_inputs['labels'] = example['summary']
    return model_inputs

data_dir = './data/json/userstudy'
data_list = os.listdir(data_dir)
for individual_file in tqdm(data_list):
    dataset = load_dataset('json', data_files={'test': join(data_dir, individual_file)})
    dataset = dataset.map(preprocess, batched=True, remove_columns=['content', 'summary'])

    preds, labels = [], []
    for sample in tqdm(dataset['test']):
        input_ids = torch.Tensor(sample['input_ids']).long().to(model.device).unsqueeze(0)
        attention_mask = torch.Tensor(sample['attention_mask']).long().to(model.device).unsqueeze(0)
        labels_batch = sample['labels']

        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_target_length).squeeze(0)

        preds.append(outputs)
        labels.append(labels_batch)
    
    decoded_preds = [tokenizer.decode(p, skip_special_tokens=True) for p in preds]
    output_prediction_file = os.path.join('./output', 'userstudy', os.path.splitext(individual_file)[0], "generated_predictions.txt")
    if not os.path.exists(join('./output', 'userstudy', os.path.splitext(individual_file)[0])):
        os.mkdir(join('./output', 'userstudy', os.path.splitext(individual_file)[0]))
    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        for p, l in zip(decoded_preds, labels):
            res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
            writer.write(f"{res}\n")
