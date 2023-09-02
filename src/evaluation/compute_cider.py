import json
import argparse
from pycocoevalcap.cider.cider import Cider

def compute_cider(filename):
    # 从文件中读取数据
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f]

    # 分割数据为假设句子和参考句子
    hypotheses = [item['predict'] for item in data]
    references = [item['labels'] for item in data]

    # 确保假设句子和参考句子的数量匹配
    assert len(hypotheses) == len(references)

    # 将列表转换为CIDEr评分器需要的格式
    hypotheses_dict = {i: [text] for i, text in enumerate(hypotheses)}
    references_dict = {i: text for i, text in enumerate(references)}

    # 计算CIDEr分数
    scorer = Cider()
    score, _ = scorer.compute_score(references_dict, hypotheses_dict)

    print('CIDEr score: ', score)

if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description='Compute CIDEr scores for predictions.')
    
    # 添加参数
    parser.add_argument('file_prefix', type=str, help='The file prefix for the predictions file.')
    
    # 解析参数
    args = parser.parse_args()
    
    # 构建文件名
    filename = f'{args.file_prefix}/generated_predictions.txt'
    
    # 调用函数，计算CIDEr分数
    compute_cider(filename)
