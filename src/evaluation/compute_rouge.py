import os
import json
import argparse
from rouge import Rouge

def compute_rouge(filename):
    # 从文件中读取数据
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f]

    # 分割数据为假设句子和参考句子
    hypotheses = [item['predict'] for item in data]
    references = [item['labels'] for item in data]

    # 确保假设句子和参考句子的数量匹配
    assert len(hypotheses) == len(references)

    # 初始化Rouge对象
    rouge = Rouge()

    # 用于存储所有ROUGE分数的列表
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    # 对每一对假设句子和参考句子计算ROUGE分数
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

    # 计算ROUGE分数的平均值
    average_rouge_1_score = sum(rouge_1_scores) / len(rouge_1_scores)
    average_rouge_2_score = sum(rouge_2_scores) / len(rouge_2_scores)
    average_rouge_l_score = sum(rouge_l_scores) / len(rouge_l_scores)

    print('Average ROUGE-1 score: ', average_rouge_1_score * 100)
    print('Average ROUGE-2 score: ', average_rouge_2_score * 100)
    print('Average ROUGE-L score: ', average_rouge_l_score * 100)

if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description='Compute ROUGE scores for predictions.')
    
    # 添加参数
    parser.add_argument('file_prefix', type=str, help='The file prefix for the predictions file.')
    
    # 解析参数
    args = parser.parse_args()
    
    # 构建文件名
    filename = os.path.join(args.file_prefix, 'generated_predictions.txt')
    
    # 调用函数，计算ROUGE分数
    compute_rouge(filename)
