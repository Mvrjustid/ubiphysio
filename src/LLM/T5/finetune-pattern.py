from os.path import join
import json

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from rouge_score import rouge_scorer

model_path = './flan-t5-large'
max_source_length = 512
max_target_length = 512
exp_name = 'win192-texts-actioncodes-patterns'
exp_id = '5'
expected_outputs = 'patterns'
data_dir = join('./data/json', exp_name, exp_id)
output_dir = join('./output', exp_name, exp_id)

'''Load dataset'''
dataset = load_dataset('json', data_files={'train': join(data_dir, 'train.json'), 'test': join(data_dir, 'test.json'), 'val': join(data_dir, 'val.json')})

'''Preprocessing'''
tokenizer = AutoTokenizer.from_pretrained(model_path)
def preprocess(example):
    model_inputs = tokenizer(example['content'], max_length=max_source_length, padding='max_length', truncation=True)
    labels = tokenizer(text_target=example['summary'], max_length=max_target_length, padding='max_length', truncation=True)
    labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs['labels'] = labels['input_ids']
    return model_inputs
dataset = dataset.map(preprocess, batched=True, remove_columns=['content', 'summary'])

'''Model'''
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100
)

'''Training Arguments'''
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    predict_with_generate=True,
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    max_steps=50000,
    logging_dir=join(output_dir, 'logs'),
    logging_strategy="steps",
    logging_steps=10,
    report_to="tensorboard",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=5000,
    save_steps=5000,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset['train'],
    eval_dataset=dataset['val'],
)

model.config.use_cache = False
trainer.train()