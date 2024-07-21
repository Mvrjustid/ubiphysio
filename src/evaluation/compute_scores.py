import os
import json
import argparse
from bert_score import score as bert_score
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider

def compute_bertscore(data):
    # Split data into hypotheses and references
    hypotheses = [item['predict'] for item in data]
    references = [item['labels'] for item in data]

    # Ensure the number of hypotheses and references match
    assert len(hypotheses) == len(references)

    # Compute BERTScore
    P, R, F1 = bert_score(hypotheses, references, lang='en', verbose=True,
                          rescale_with_baseline=True, idf=True)

    # Compute the average BERTScore values
    average_precision = P.mean().item()
    average_recall = R.mean().item()
    average_f1_score = F1.mean().item()

    print('Average BERT Precision: ', average_precision * 100)
    print('Average BERT Recall: ', average_recall * 100)
    print('Average BERT F1 Score: ', average_f1_score * 100)

def compute_rouge(data):
    # Split data into hypotheses and references
    hypotheses = [item['predict'] for item in data]
    references = [item['labels'] for item in data]

    # Ensure the number of hypotheses and references match
    assert len(hypotheses) == len(references)

    # Initialize Rouge object
    rouge = Rouge()

    # Lists to store all ROUGE scores
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    # Compute ROUGE scores for each pair of hypothesis and reference(s)
    for hypothesis, reference_list in zip(hypotheses, references):
        max_rouge_1 = max_rouge_2 = max_rouge_l = 0
        for reference in reference_list:
            scores = rouge.get_scores(hypothesis, reference)[0]
            max_rouge_1 = max(max_rouge_1, scores['rouge-1']['f'])
            max_rouge_2 = max(max_rouge_2, scores['rouge-2']['f'])
            max_rouge_l = max(max_rouge_l, scores['rouge-l']['f'])
        rouge_1_scores.append(max_rouge_1)
        rouge_2_scores.append(max_rouge_2)
        rouge_l_scores.append(max_rouge_l)

    # Compute the average ROUGE scores
    average_rouge_1_score = sum(rouge_1_scores) / len(rouge_1_scores)
    average_rouge_2_score = sum(rouge_2_scores) / len(rouge_2_scores)
    average_rouge_l_score = sum(rouge_l_scores) / len(rouge_l_scores)

    print('Average ROUGE-1 score: ', average_rouge_1_score * 100)
    print('Average ROUGE-2 score: ', average_rouge_2_score * 100)
    print('Average ROUGE-L score: ', average_rouge_l_score * 100)

def compute_bleu(data):
    # Create a SmoothingFunction object for BLEU score calculation
    smoothie = SmoothingFunction().method3

    # Split data into hypotheses and references
    hypotheses = [item['predict'] for item in data]
    references = [item['labels'] for item in data]

    # Ensure the number of hypotheses and references match
    assert len(hypotheses) == len(references)

    # Lists to store all BLEU-1 and BLEU-4 scores
    bleu1_scores = []
    bleu4_scores = []

    # Compute BLEU-1 and BLEU-4 scores for each pair of hypothesis and references
    for hypothesis, reference in zip(hypotheses, references):
        reference_list = [ref.split() for ref in reference]  # Split each reference sentence and store in a list
        score1 = sentence_bleu(reference_list, hypothesis.split(), weights=(1, 0, 0, 0), smoothing_function=smoothie)
        score4 = sentence_bleu(reference_list, hypothesis.split(), smoothing_function=smoothie)
        bleu1_scores.append(score1)
        bleu4_scores.append(score4)

    # Compute the average BLEU-1 and BLEU-4 scores
    average_bleu1_score = sum(bleu1_scores) / len(bleu1_scores)
    average_bleu4_score = sum(bleu4_scores) / len(bleu4_scores)

    # Print the average scores
    print('Average BLEU-1 score: ', average_bleu1_score * 100)
    print('Average BLEU-4 score: ', average_bleu4_score * 100)

def compute_cider(data):
    # Split data into hypotheses and references
    hypotheses = [item['predict'] for item in data]
    references = [item['labels'] for item in data]

    # Ensure the number of hypotheses and references match
    assert len(hypotheses) == len(references)

    # Convert lists into the format required by the CIDEr scorer
    hypotheses_dict = {i: [text] for i, text in enumerate(hypotheses)}
    references_dict = {i: text for i, text in enumerate(references)}

    # Compute the CIDEr score
    scorer = Cider()
    score, _ = scorer.compute_score(references_dict, hypotheses_dict)

    # Print the CIDEr score
    print('CIDEr score: ', score)

def main(file_prefix):
    # Construct the filename
    filename = f'{file_prefix}/generated_predictions.txt'
    
    # Read data from the file
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Compute all scores
    compute_bertscore(data)
    compute_rouge(data)
    compute_bleu(data)
    compute_cider(data)

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Compute BERTScore, ROUGE, BLEU, and CIDEr scores for predictions.')
    
    # Add argument for file prefix
    parser.add_argument('file_prefix', type=str, help='The file prefix for the predictions file.')
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Call the main function
    main(args.file_prefix)
