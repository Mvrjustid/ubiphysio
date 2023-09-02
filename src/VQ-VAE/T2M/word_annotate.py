import spacy
import os

# 加载预训练模型
nlp = spacy.load('en_core_web_sm')

def process_files2(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            # 读取文件的每一行
            with open(os.path.join(input_dir, filename), 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                # 全部转为小写
                lower_line = line.lower().strip() # convert to lowercase and remove newline character
                
                # 进行词性标注
                doc = nlp(lower_line)
                processed_text = " ".join([f"{token.text}/{token.pos_}" for token in doc])
                
                # 将词性标注文本添加到原始文本后面
                result_text = lower_line + "#" + processed_text + "#0.0#0.0"
                new_lines.append(result_text)
            
            # 写入新的文件
            with open(os.path.join(output_dir, filename), 'w') as f:
                f.write('\n'.join(new_lines))  # join lines with newline character

# 调用函数，处理.txt文件并生成新的.txt文件
process_files2("./dataset/ubiphysio/raw_texts/", "./dataset/ubiphysio/texts/")
