import json
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu(filename):
    # 创建一个SmoothingFunction对象
    smoothie = SmoothingFunction().method3

    # 从文件中读取数据
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f]

    # 分割数据为假设句子和参考句子
    hypotheses = [item['predict'] for item in data]
    references = [item['labels'] for item in data]

    # 确保假设句子和参考句子的数量匹配
    assert len(hypotheses) == len(references)

    # 用于存储所有BLEU-1和BLEU-4分数的列表
    bleu1_scores = []
    bleu4_scores = []

    # 对每一对假设句子和参考句子计算BLEU-1和BLEU-4分数
    for hypothesis, reference in zip(hypotheses, references):
        reference_list = [ref.split() for ref in reference]  # 将每个参考句子分割并存储在一个列表中
        score1 = sentence_bleu(reference_list, hypothesis.split(), weights=(1, 0, 0, 0), smoothing_function=smoothie)
        score4 = sentence_bleu(reference_list, hypothesis.split(), smoothing_function=smoothie)
        bleu1_scores.append(score1)
        bleu4_scores.append(score4)

    # 计算BLEU-1和BLEU-4分数的平均值
    average_bleu1_score = sum(bleu1_scores) / len(bleu1_scores)
    average_bleu4_score = sum(bleu4_scores) / len(bleu4_scores)

    print('Average BLEU-1 score: ', average_bleu1_score * 100)
    print('Average BLEU-4 score: ', average_bleu4_score * 100)

if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description='Compute BLEU scores for predictions.')
    
    # 添加参数
    parser.add_argument('file_prefix', type=str, help='The file prefix for the predictions file.')
    
    # 解析参数
    args = parser.parse_args()
    
    # 构建文件名
    filename = f'{args.file_prefix}/generated_predictions.txt'
    
    # 调用函数，计算BLEU分数
    compute_bleu(filename)
