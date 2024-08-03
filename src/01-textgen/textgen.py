import json
import os
import glob
import re
import pandas as pd
from tqdm import tqdm
import argparse

# Expert annotated error patterns
error_patterns = ['腰背部前屈/弯腰拱背', '搬起重物后腰背部过伸', '直接从仰卧位起身', '直接仰卧位躺下', '直接从仰卧位躺下',
        '活动度降低','脊柱前屈', '躯干旋转', '髋部侧移', '屈膝代偿', '躯干稳定性不足','腰部过伸/骨盆前倾', 
        '躯干稳定性不足','腰背部前屈/弯腰弓背', '腰背部过伸/塌腰', '躯干旋转', '躯干偏离中线/躯干水平侧移', 
        '髋部过屈', '胸椎过伸代替肩关节活动', '腰背部过伸/塌腰', '头未着地', '脊柱后伸','躯干扭转','颈部侧屈代偿',
        '髋部过伸导致腰背部过伸', '躯干旋转', '躯干偏离中线/躯干水平侧移', '上胸部下沉', '髋部过屈','蹲得太浅',
        '腰部侧屈', '髋部倾斜', '腰椎前屈/弯腰拱背', '膝关节过度前移', '上身直立下蹲', '大腿未垂直地面',
        '屈髋屈膝角度过大', '双侧负重不均', '活动度降低','脊柱前屈', '躯干旋转','垫脚/骨盆倾斜','做错为正常行走','躯干前屈',
        '屈膝代偿', '髋部过度左右移动', '踝背屈不足', '躯干前倾', '腰部离开床面', '同手同脚活动', '无躯干活动',
        '小腿未平行床面/小腿下垂','核心肌力不足','协调性不足','活动度不足']

def extract_text():
    '''Extracts action categories and error patterns from xlsx files, sorts by action time, and outputs as json'''

    folders = glob.glob('expert_descriptions/[CP]*')

    for folder in tqdm(folders):
        for file in os.listdir(folder):
            if file.endswith('.xlsx'):
                filename, fileext = os.path.splitext(file)
                file_path = os.path.join(folder, file)
                
                df = pd.read_excel(file_path)
                df = df.sort_values('onset')
                df = df.reset_index(drop=True)
                df = df[['action_label', '行为或错误描述']]

                result = []
                for i in range(len(df)):
                    label = df['action_label'][i]
                    error = "" if pd.isnull(df['行为或错误描述'][i]) else df['行为或错误描述'][i]
                    
                    error = re.split(',|，|、|⭐', error)
                    error_idx = [error_patterns.index(err) for err in error if err]
                    
                    result.append((label, error_idx))
                
                assert len(result) == len(df)
                with open(os.path.join(folder, filename + '.json'), 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False)

def sumup():
    '''Sums up the error patterns for each action category and saves them into a total json file'''

    folders = glob.glob('expert_descriptions/[CP]*')

    total_pattern = {}

    for folder in tqdm(folders):
        for file in os.listdir(folder):
            if file.endswith('.json'):
                file_path = os.path.join(folder, file)

                with open(file_path, 'r', encoding='utf-8') as f:
                    annos = json.load(f)
                
                for anno in annos:
                    action_label, error_pattern = anno[0], anno[1]
                    try:
                        if error_pattern in total_pattern[action_label]:
                            continue
                        else:
                            total_pattern[action_label].append(error_pattern)
                    except Exception as e:
                        total_pattern[action_label] = [error_pattern]
                
    with open('total.json', 'w', encoding='utf-8') as f:
        json.dump(total_pattern, f, ensure_ascii=False)

def generate_texts():
    '''Generates text descriptions for each <action category, error pattern> combination'''

    openai.api_key = 'API_KEY'

    with open('total.json', 'r', encoding='utf-8') as f:
        total = json.load(f)
    
    results = {}
    
    for key, values in tqdm(total.items(), total=len(total)):
        if key not in results:
            results[key] = {}

        for value in values:
            if value:
                prompt = f'''现在我将给你提供一些运动描述对，包括运动种类和其对应的错误模式组合，组合中从左到右代表错误模式出现的频次从高到低。
请你扮演一个专业的临床康复理疗医师，用流畅的英文，为每个这样的文本对提供3种生动的描述，字数保持在15个单词上下，不要超过20个单词。每次你只需要输出三句描述，用回车隔开，不要添加数字编号。
在描述中，请用 'this individual' 称呼动作执行者，人称代词使用 they/them。
例如，对于运动种类：系鞋带, 错误模式：腰背部前屈/弯腰拱背，你可以做出这样的描述：“This individual is tying their shoe laces, with their back bended.”
现在你需要描述的运动种类是：{key}，错误模式组合是{'，'.join([error_patterns[idx] for idx in value])}'''
            else:
                prompt = f'''现在我将给你提供一些运动种类关键词。
请你扮演一个专业的临床康复理疗医师，用流畅的英文，为每个这样的关键词提供3种生动的描述，字数保持在15个单词上下，不要超过20个单词。每次你只需要输出三句描述，不需要输出任何其他内容。
在描述中，请用 'this individual' 称呼动作执行者，人称代词使用 they/them。
例如，对于运动种类：系鞋带, 你可以做出这样的描述：“This individual is tying their shoe laces.”
现在你需要描述的运动种类是：{key}'''

            response = openai.ChatCompletion.create(
                model = 'gpt-4',
                messages = [
                    {'role': 'user', 'content': prompt}
                ]
            )

            description = response.choices[0]['message']['content']
            results[key][str(value)] = re.findall(r'"(.*?)"', description)
    
    with open('texts.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

def makeup():
    '''Checks and completes missing descriptions in the texts.json file'''

    openai.api_key = 'API_KEY'
    
    with open('texts.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    result = data.copy()
    
    for key, dicc in tqdm(data.items(), total=len(data)):
        for pattern, texts in dicc.items():
            if not texts:
                if eval(pattern):
                    prompt = f'''现在我将给你提供一些运动描述对，包括运动种类和其对应的错误模式组合，组合中从左到右代表错误模式出现的频次从高到低。
请你扮演一个专业的临床康复理疗医师，用流畅的英文，为每个这样的文本对提供3种生动的描述，字数保持在15个单词上下，不要超过20个单词。每次你只需要输出三句描述，用回车隔开，不要添加数字编号。
在描述中，请用 'this individual' 称呼动作执行者，人称代词使用 they/them。
例如，对于运动种类：系鞋带, 错误模式：腰背部前屈/弯腰拱背，你可以做出这样的描述：“This individual is tying their shoe laces, with their back bended.”
现在你需要描述的运动种类是：{key}，错误模式组合是{'，'.join([error_patterns[idx] for idx in eval(pattern)])}'''
                else:
                    prompt = f'''现在我将给你提供一些运动种类关键词。
请你扮演一个专业的临床康复理疗医师，用流畅的英文，为每个这样的关键词提供3种生动的描述，字数保持在15个单词上下，不要超过20个单词。每次你只需要输出三句描述，不需要输出任何其他内容。
在描述中，请用 'this individual' 称呼动作执行者，人称代词使用 they/them。
例如，对于运动种类：系鞋带, 你可以做出这样的描述：“This individual is tying their shoe laces.”
现在你需要描述的运动种类是：{key}'''

                response = openai.ChatCompletion.create(
                    model = 'gpt-4',
                    messages = [
                        {'role': 'user', 'content': prompt}
                    ]
                )

                description = response.choices[0]['message']['content']
                result[key][pattern] = re.findall(r'"(.*?)"', description)
    
    with open('texts-2.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some actions.')
    parser.add_argument('action', choices=['extract', 'sumup', 'generate', 'makeup'], help='Action to perform')

    args = parser.parse_args()

    if args.action == 'extract':
        extract_text()
    elif args.action == 'sumup':
        sumup()
    elif args.action == 'generate':
        generate_texts()
    elif args.action == 'makeup':
        makeup()
