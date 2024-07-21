from os.path import join
import json
import os
import argparse

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from rouge_score import rouge_scorer
from peft import PeftModelForSeq2SeqLM, LoraConfig, PeftConfig, PeftModel

# Parse command line arguments
parser = argparse.ArgumentParser(description="Finetune target model.")
parser.add_argument('--exp_name', type=str, help="target experiment name")
parser.add_argument('--exp_id', type=str, help="target experiment id")
parser.add_argument('--ckpt_id', type=str, help="target checkpoint id")
args = parser.parse_args()

# Set paths and parameters based on the arguments
model_path = f'./datadir/models/flan-t5-xl'
max_source_length = 512
max_target_length = 512
exp_name = args.exp_name
exp_id = args.exp_id
expected_outputs = 'texts'
data_dir = join('./datadir/data/json', exp_name, exp_id)
output_dir = join('./datadir/output', 'xl', exp_name, exp_id)

# Load dataset from JSON files
dataset = load_dataset('json', data_files={'train': join(data_dir, 'train.json'), 'test': join(data_dir, 'test.json'), 'val': join(data_dir, 'val.json')})

# Preprocess the dataset
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

# Load label dictionary
label_dict = {}
with open(f'./datadir/data/{expected_outputs}_dict.json', 'r') as f:
    label_dict = json.load(f)

# Define a function to compute evaluation metrics
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    predicted_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Use label_dict to convert labels
    label_texts = [label_dict[label] for label in label_texts]

    # Create a ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Compute ROUGE-L score for each sample
    scores = []
    for pred, label_list in zip(predicted_texts, label_texts):
        max_score = 0
        for label in label_list:
            score = scorer.score(label, pred)['rougeL'].fmeasure
            if score > max_score:
                max_score = score
        scores.append(max_score)

    # Return the average ROUGE-L score
    return {'rougeL': np.mean(scores)}

# Set the target path for loading model
target_path = join(output_dir, f'checkpoint-{args.ckpt_id}')

# Load Peft configuration and model
peft_config = PeftConfig.from_pretrained(target_path)
model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, target_path)

# Data collator for Seq2Seq model
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100
)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    predict_with_generate=True,
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    max_steps=100000,
    logging_dir=join(output_dir, 'logs'),
    logging_strategy="steps",
    logging_steps=10,
    report_to="tensorboard",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=5000,
    save_steps=5000,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    save_total_limit=2
)

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset['train'],
    eval_dataset=dataset['val'],
    compute_metrics=compute_metrics
)

# Predict and evaluate the model on the test set
predict_results = trainer.predict(dataset['test'])
preds, labels = predict_results.predictions, predict_results.label_ids
if isinstance(preds, tuple):
    preds = preds[0]
decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
# Replace -100 in the labels as we can't decode them.
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

# Use label_dict to convert labels
decoded_labels = [label_dict[lb] for lb in decoded_labels]

# Save the predictions to a file
output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
with open(output_prediction_file, "w", encoding="utf-8") as writer:
    for p, l in zip(decoded_preds, decoded_labels):
        res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
        writer.write(f"{res}\n")
