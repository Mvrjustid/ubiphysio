import os
import numpy as np
import pickle
import nltk
import json

# nltk.download('punkt')  # 如果还没有下载过nltk的punkt数据包，需要先下载

def load_glove_model(file):
    model = {}
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
    return model

# 加载原始GloVe模型
glove_path = "glove/glove.6B.300d.txt"
glove_model = load_glove_model(glove_path)

# 加载你的词嵌入
word_vecs = np.load('glove/our_vab_data.npy')

# 加载你的词-索引映射
with open('glove/our_vab_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)

# 加载一些路径
vecs_path = 'glove/our_vab_data.npy'
idx_path = 'glove/our_vab_idx.pkl'
words_path = 'glove/our_vab_words.pkl'

Loc_list1 = ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise', 'forward', 'backward',
            'up', 'down', 'straight', 'curve','distance','out', 'front','opposite','on', 'beside')

Body_list1 = ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg', 'waist', 'eye', 'knee', 'shoulder', 'thigh',
             'side', 'body','chest', 'muscle','toe', 'limb','heel')

Obj_List1 = ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball', 'handrail', 'baseball', 'basketball',
            'sofa', 'broom', 'couch', 'dust', 'debris', 'corner', 'spot', 'environment','shoelace','suitcase', 'pillow','ground','bed')

Act_list1 = ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat', 'throw', 'hop', 'dance', 'jump', 'turn',
            'stumble', 'stop', 'sit', 'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll',
            'rub', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate', 'spin', 'spread', 'climb',
            'untie', 'tie', 'stoop', 'loosen', 'fasten', 'sweep', 'remove', 'clean', 'ensure', 'leave', 'transition', 'rise',
            'set', 'carry', 'grab', 'bend', 'perform', 'alternate', 'demonstrate', 'wrap',
            'expand', 'contract', 'engage', 'execute', 'extend', 'maintain', 'focus', 'target', 'touch',
            'contact', 'ambulating','catwalk','lying')

Desc_list1 = ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily',
             'angrily', 'sadly','securely', 'meticulously', 'thoroughly', 'comfortably','short', 'certain', 'circular',
             'wide', 'together','parallel','deep','only', 'casual', 
             'relaxed', 'specific', 'nonchalant', 'easygoing', 'rush', 'unhurried')

VIP_dict_act = {
    'Loc_VIP': Loc_list1,
    'Body_VIP': Body_list1,
    'Obj_VIP': Obj_List1,
    'Act_VIP': Act_list1,
    'Desc_VIP': Desc_list1,
}

with open('./dataset/ubiphysio/VIP.json', 'r') as f:
    VIP_dict_full = json.load(f)

# 遍历所有VIP词，查看是否存在于词典中
def check_VIP_dict(dict):
    
    with open('glove/our_vab_words.pkl', 'rb') as f:
        idx_to_word = pickle.load(f)

    for vip_type, vip_words in dict.items():
        for word in vip_words:
            if word in idx_to_word:
                # print(f"{word} in {vip_type} is in the vocabulary.")
                continue
            else:
                print(f"{word} in {vip_type} is NOT in the vocabulary.")

# check_VIP_dict(VIP_dict)

def compare_vectors(word_list, word_to_idx, word_vecs, glove_model):
    for word in word_list:
        if word in word_to_idx and word in glove_model:
            idx = word_to_idx[word]  # 获取词的索引
            word_vector_current = word_vecs[idx]  # 获取词的向量
            word_vector_glove = glove_model[word]  # 从GloVe模型中获取相同词的向量

            # 比较两个向量
            if np.allclose(word_vector_current, word_vector_glove, atol=1e-5):
                # print(f"The vectors for '{word}' are the same.")
                continue
            else:
                print(f"The vectors for '{word}' are different!")
        elif word in glove_model:
            print(f"Word '{word}' not found in current vocabulary but found in GloVe model!!")
        else:
            print(f"Word '{word}' not found in any vocabularies!!!")

def add_words_to_vocab(new_words, glove_model, vecs_path, idx_path, words_path):
    # 加载你的词向量和词到索引的映射
    word_vecs = np.load(vecs_path)
    with open(idx_path, 'rb') as f:
        word_to_idx = pickle.load(f)
    with open(words_path, 'rb') as f:
        idx_to_word = pickle.load(f)

    # 对于每一个新词
    for word in new_words:
        # 如果新词在GloVe模型中
        if word in glove_model and word not in word_to_idx:
            # 获取新词的向量
            new_word_vec = glove_model[word]
            # 添加新词的向量到你的词向量中
            word_vecs = np.vstack([word_vecs, new_word_vec])
            # 更新你的词到索引的映射和索引到词的映射
            new_idx = len(word_to_idx)
            word_to_idx[word] = new_idx
            idx_to_word.append(word)  # 使用append()而不是直接赋值
            print(f"Word '{word}' added!")

    # 保存更新后的词向量和词到索引的映射
    np.save(vecs_path, word_vecs)
    with open(idx_path, 'wb') as f:
        pickle.dump(word_to_idx, f)
    with open(words_path, 'wb') as f:
        pickle.dump(idx_to_word, f)

def check_keywords_in_glove(text_dict):
    keywords = []
    # for key, sentences in text_dict.items(): #This is for dict. with sentences
    #     for sentence in sentences:
    #         # 使用nltk的word_tokenize函数将句子拆解为单词
    #         words = nltk.word_tokenize(sentence)
    #         # 将这些单词添加到关键词列表中
    #         keywords.extend(words)
    
    for key, words in text_dict.items(): #This is for dict. with words
        keywords.extend(words)

    word_counts = nltk.FreqDist(keywords)
    duplicate_words = [word for word, count in word_counts.items() if count > 1]
    print(f"重复的单词是: {duplicate_words}")

    # 去掉重复的词
    unique_keywords = list(set(keywords))

    # 调用你已经实现的compare_vectors()函数，输入关键词列表
    compare_vectors(unique_keywords, word_to_idx, word_vecs, glove_model)

    # 添加关键词到本地
    add_words_to_vocab(unique_keywords, glove_model, vecs_path, idx_path, words_path)

    # 查看当前的关键词
    # print(unique_keywords)

# 调用函数
check_keywords_in_glove(VIP_dict_full)



# 其他小工具
# 检查词列表的重复性
full_words_list = ['slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily',
             'angrily', 'sadly',
             'stiff', 'stiffly', 'interrupted','rigid','rigidly','continuous','continuously','broken','abnormal','abnormally',
             'sudden','suddenly','extraneous','extraneously','intended','intentionally', 'intermittently', 'rigidity', 'extraneous', 
             'stiffness', 'abruptly', 'visible', 'continuous', 'hesitation', 'ordinary', 'rigid', 'suggesting', 'unexpected', 'abnormal', 
             'unintended', 'unusual', 'needing', 'individual', 'stance', 'movements', 'distribution', 'characterized', 'jerky', 'apparent', 
             'broken', 'unbalanced', 'posture', 'periodically', 'fragmented', 'sudden', 'allocation', 'position', 'segmented', 'gait']
unique_keywords = list(set(full_words_list))
# print(unique_keywords)