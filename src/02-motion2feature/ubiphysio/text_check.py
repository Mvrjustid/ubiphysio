import os
import re

# def find_empty_files(directory):
#     empty_files = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.txt'):  # 确保处理的是文本文件
#             file_path = os.path.join(directory, filename)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 content = file.read().strip()  # 读取内容并移除前后空白字符
#                 if not content:  # 检查内容是否为空
#                     empty_files.append(filename)
#     return empty_files

# directory_path = '/datax/zhonglx/archive/texts'  # 替换为你的文件夹路径
# empty_files = find_empty_files(directory_path)

# print("以下文件内容为空：")
# for file in empty_files:
#     print(file)

#############################################################################

# def find_files_without_slash(directory):
#     files_without_slash = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.txt'):  # 确保处理的是文本文件
#             file_path = os.path.join(directory, filename)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 content = file.read()
#                 if '/' not in content:  # 检查内容中是否不包含 '/'
#                     files_without_slash.append(filename)
#     return files_without_slash

# directory_path = '/datax/zhonglx/archive/texts'  # 替换为你的文件夹路径
# files_without_slash = find_files_without_slash(directory_path)

# print("以下文件中不包含 '/' 符号：")
# for file in files_without_slash:
#     print(file)

#############################################################################

def replace_double_spaces(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 使用正则表达式替换两个连续空格为一个空格
            updated_content = re.sub(r'  +', ' ', content)

            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(updated_content)
            print(f"已更新文件：{filename}")

directory_path = '/datax/zhonglx/archive/texts'  # 替换为你的文件夹路径
replace_double_spaces(directory_path)