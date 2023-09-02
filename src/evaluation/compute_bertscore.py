import json
import argparse
from bert_score import score

def compute_bertscore(filename):
    # 从文件中读取数据
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f]

    # 分割数据为假设句子和参考句子
    hypotheses = [item['predict'] for item in data]
    references = [item['labels'] for item in data]

    # 确保假设句子和参考句子的数量匹配
    assert len(hypotheses) == len(references)

    # 计算BERTScore
    P, R, F1 = score(hypotheses, references, lang='en', verbose=True,
                     rescale_with_baseline=True, idf=True)

    # 计算BERTScore的平均值
    average_precision = P.mean().item()
    average_recall = R.mean().item()
    average_f1_score = F1.mean().item()

    print('Average Precision: ', average_precision * 100)
    print('Average Recall: ', average_recall * 100)
    print('Average F1 Score: ', average_f1_score * 100)

if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description='Compute BERTScore for predictions.')
    
    # 添加参数
    parser.add_argument('file_prefix', type=str, help='The file prefix for the predictions file.')
    
    # 解析参数
    args = parser.parse_args()
    
    # 构建文件名
    filename = f'{args.file_prefix}/generated_predictions.txt'
    
    # 调用函数，计算BERTScore
    compute_bertscore(filename)
