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

parser = argparse.ArgumentParser()
parser.add_argument('exp_id', type=str)
parser.add_argument('ckpt_id', type=str)
args = parser.parse_args()

exp_name = 'win192-texts-actioncodes-patterns'
exp_id = args.exp_id
checkpoint_id = args.ckpt_id
#exp_id = '1'
#checkpoint_id = '5000'

data_dir = join('./data/patterneval', exp_name, exp_id)
output_dir = join('./output', exp_name, exp_id)

'''Load dataset'''
dataset = load_dataset('json', data_files={'train': join(data_dir, 'train.json'), 'test': join(data_dir, 'test.json'), 'val': join(data_dir, 'val.json')})

'''Preprocessing'''
tokenizer = AutoTokenizer.from_pretrained(model_path)
def preprocess(example):
    model_inputs = tokenizer(example['content'], max_length=max_source_length, padding='max_length', truncation=True)
    model_inputs['labels'] = example['summary']
    return model_inputs
dataset = dataset.map(preprocess, batched=True, remove_columns=['content', 'summary'])

'''Model'''
model = AutoModelForSeq2SeqLM.from_pretrained(join(output_dir, f'checkpoint-{checkpoint_id}')).cuda()
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100
)

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

output_prediction_file = os.path.join(output_dir, f"checkpoint-{checkpoint_id}", "generated_predictions.txt")
with open(output_prediction_file, "w", encoding="utf-8") as writer:
    for p, l in zip(decoded_preds, labels):
        res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
        writer.write(f"{res}\n")