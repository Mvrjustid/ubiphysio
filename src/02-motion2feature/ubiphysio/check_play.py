import os
import numpy as np
import pickle
import nltk
# nltk.download('punkt')  # 如果还没有下载过nltk的punkt数据包，需要先下载

# 查看our_vab_data.npy
# word_vecs = np.load('glove/our_vab_data.npy')
# print(np.shape(word_vecs))

# 查看our_vab_idx.pkl
# with open('glove/our_vab_idx.pkl', 'rb') as f:
    # word_to_idx = pickle.load(f)
# print(len(word_to_idx))

# 查看our_vab_words.pkl
# with open('glove/our_vab_words.pkl', 'rb') as f:
    # idx_to_word = pickle.load(f)
# print(idx_to_word)

# # 定义VIP词
VIP_dict_old = {
    'Loc_VIP': ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise', 'forward', 'back', 'backward', 'up', 'down', 'straight', 'curve'),
    'Body_VIP': ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg', 'waist', 'eye', 'knee', 'shoulder', 'thigh'),
    'Obj_VIP': ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball', 'handrail', 'baseball', 'basketball'),
    'Act_VIP': ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat', 'throw', 'hop', 'dance', 'jump', 'turn', 'stumble', 'dance', 'stop', 'sit', 'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll', 'rub', 'bend', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate', 'spin', 'spread', 'climb'),
    'Desc_VIP': ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily', 'angrily', 'sadly')
}

# 我们自己的词，其中有一些不常见的词
our_VIP_dict = {
    'Act_VIP': ('guarding', 'hesitation', 'support', 'bracing', 'rubbing', 'stimulation'),
    'Desc_VIP': ('stiff', 'interrupted', 'rigid', 'stopping', 'broken', 'abnormal', 'sudden', 'extraneous', 'massaging', 'touching', 'shaking'),
    'Loc_list': ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise', 'forward', 'back', 'backward',
            'up', 'down', 'straight', 'curve'),
    'Body_list': ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg', 'waist', 'eye', 'knee', 'shoulder', 'thigh',
             'hands', 'legs', 'arms', 'limb', 'body'),
    'Obj_List': ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball', 'handrail', 'baseball', 'basketball'),
    'Act_list': ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat', 'throw', 'hop', 'dance', 'jump', 'turn',
            'stumble', 'dance', 'stop', 'sit', 'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll',
            'rub', 'bend', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate', 'spin', 'spread', 'climb',
            'reach', 'walking', 'support', 'maintain', 'massaging', 'touching', 'shaking','appearing','pause'
            ),
    'Desc_list': ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily',
             'angrily', 'sadly',
             'stiff', 'stiffly', 'interrupted','interruptedly','rigid','rigidly','continuous','continuously','broken','abnormal','abnormally',
             'sudden','suddenly','extraneous','extraneously','intended','intentionally')
}

Loc_list = ('up', 'backward', 'straight', 'anticlockwise', 'left', 'back', 'curve', 'clockwise', 
            'down', 'counterclockwise', 'forward', 'right')

Body_list = ('hand', 'waist', 'face', 'legs', 'knee', 'arm', 'shoulder', 'mouth', 'feet', 'foot', 
             'chin', 'leg', 'eye', 'hands', 'arms', 'limb', 'body', 'thigh', 'parts')

Obj_List = ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball', 'handrail', 'baseball', 'basketball')

Act_list = ('interrupted', 'walk', 'supporting', 'spin', 'throw', 'bending', 'exhibits', 'rotate', 'supports', 
            'maintain', 'climb', 'bring', 'stimulation', 'walking', 'guarding', 'dance', 'engage', 'maintains', 
            'lean', 'raise', 'stopped', 'spread', 'leaning', 'hop', 'stimulating', 'stop', 'bracing', 'walks', 
            'displaying', 'sit', 'run', 'balance', 'jump', 'rise', 'rub', 'stride', 'appear', 'showcasing', 'sits', 
            'lower', 'guarded', 'exhibiting', 'leans', 'support', 'engaged', 'jog', 'maintaining', 'rising', 'standing', 
            'reaches', 'paused', 'seated', 'attempting', 'rubbing', 'turn', 'pick', 'exhibited', 'shaking', 'transitioning', 
            'lift', 'touching', 'movement', 'stumble', 'act', 'halted', 'massaging', 'reaching', 'rises', 'wash', 'stand', 
            'moving', 'engages', 'strides', 'portraying', 'trying', 'appearing', 'stands', 'attempt', 'bend', 'stroll', 
            'swing', 'bends', 'observed', 'kneel', 'attempts', 'flap', 'indicating', 'reach', 'transitions', 'extending', 
            'squat', 'shuffle', 'kick', 'extends', 'put', 'pause', 'showing', 'shaking', 'lowering', 'going'
            )

Desc_list = ('angry', 'posture', 'distribution', 'characterized', 'movements', 'unexpected', 'rigidity', 'suggesting', 
             'unbalanced', 'abnormally', 'apparent', 'angrily', 'quickly', 'intentionally', 'needing', 'continuously', 
             'unintended', 'happy', 'stiffly', 'interrupted', 'abnormal', 'stiffness', 'rigidly', 'unusual', 'fragmented', 
             'gait', 'slowly', 'broken', 'hesitation', 'carefully', 'intermittently', 'individual', 'ordinary', 'periodically', 
             'position', 'allocation', 'sudden', 'sadly', 'sad', 'rigid', 'visible', 'happily', 'slow', 'fast', 
             'suddenly', 'stance', 'intended', 'stiff', 'extraneous', 'abruptly', 'jerky', 'segmented', 'careful', 'continuous',
             'affected','predominantly','primarily','mainly','interspersed','brief')

VIP_dict = {
    'Loc_VIP': Loc_list,
    'Body_VIP': Body_list,
    'Obj_VIP': Obj_List,
    'Act_VIP': Act_list,
    'Desc_VIP': Desc_list,
}

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

# 选择一个词来进行比较，假设我们选择"support"
new_words = ["one", "leg", "stand", 
             "reach", "forward", 
             "sit", "arms", 
             "bend", "down", 
             "walking", 
             "stiff", "interrupted", "rigid", "movement", 
             "stopping", "partway", "continuous", "appearing", "broken", "stages", 
             "position", "limb", "supports", "maintains", "abnormal", "distribution", "weight", "done", "without", "support", 
             "sudden", "extraneous", "intended", "motion", "pause", "hesitation", 
             "massaging", "touching", "affected", "body", "part", "another", "shaking", "hands", "legs", "feet"]
new_words1 = ['hands', 'legs', 'arms', 'limb', 'body',
              'reach', 'walking', 'support', 'maintain', 'massaging', 'touching', 'shaking','appearing','pause','guarding','bracing','stimulation',
              'stiff', 'stiffly', 'interrupted','rigid','rigidly','continuous','continuously','broken','abnormal','abnormally',
              'sudden','suddenly','extraneous','intended','intentionally'
             ]
new_words2 = ['stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball', 'handrail', 'baseball', 'basketball']

# add_words_to_vocab(new_words1, glove_model, vecs_path, idx_path, words_path)
# compare_vectors(new_words2, word_to_idx, word_vecs, glove_model)

description_dict_v1 = {

    (0, 0): [ "a person stands on one leg, displaying interrupted or rigid movements.", 
              "an individual maintains a single leg stance, exhibiting signs of guarding with stiffness.", 
              "one leg supports the person, showcasing stiff and abruptly paused motion."],
    (0, 1): [ "a person attempts to stand on one leg, showing hesitation and movements broken into stages.",
              "an individual trying to maintain a one leg stand, the action appearing segmented due to hesitation.",
              "a person's one leg stance is characterized by stopped motions, portraying the behavior of hesitation."],
    (0, 2): [ "a person stands on one leg, maintaining an abnormal distribution of weight.",
              "an individual exhibits an unbalanced weight distribution while attempting a single leg stance.",
              "in a one leg stand, the person showcases a posture supporting an unusual weight allocation."],
    (0, 3): [ "a person stands on one leg, showing sudden movements extraneous to the intended motion.",
              "an individual attempts a one leg stance, displaying jerky, unintended movements.",
              "a person trying to stand on one leg, featuring sudden extraneous motion."],
    (0, 4): [ "a person maintaining a one leg stand while massaging or touching the standing leg.",
              "an individual stands on one leg, intermittently rubbing or stimulating the supporting limb.",
              "during a one leg stance, the person exhibits behavior of rubbing or shaking the standing leg."],

    (1, 0): [ "a person reaches forward, the movement appearing stiff and interrupted.", 
              "an individual extends their body forward, displaying rigidity and signs of guarding.", 
              "a person trying to reach forward showcases stiff and abruptly paused motions."],
    (1, 1): [ "a person attempting to reach forward, showing hesitation with movements broken into stages.",
              "an individual extending forward exhibits a paused motion, indicating hesitation.",
              "the action of reaching forward is segmented due to a person's visible hesitation."],
    (1, 2): [ "a person reaches forward, maintaining an abnormal distribution of weight for support.",
              "an individual leaning forward showcases unusual weight distribution suggesting the need for support.",
              "in an attempt to reach forward, the person shows signs of abnormal weight distribution for balance."],
    (1, 3): [ "a person reaching forward displays sudden, extraneous movements.",
              "as an individual leans forward, unexpected and jerky movements are exhibited.",
              "the motion of reaching forward is characterized by sudden, unintended movements."],
    (1, 4): [ "a person reaches forward while massaging or stimulating their arms or legs.",
              "as an individual extends forward, the behavior of rubbing or shaking limbs is observed.",
              "a person leans forward while intermittently touching or shaking their body parts."],

    (2, 0): [ "a person transitions from sitting to standing, showing signs of stiffness and interrupted movement.", 
              "as an individual rises from a seated position, their movements exhibit guarding and rigidity.", 
              "in the act of standing up, a person's motion seems stiff and abruptly halted, showcasing guarding behavior."],
    (2, 1): [ "a person attempts to rise from a seated position, displaying hesitation with movements broken into stages.",
              "the act of transitioning from sitting to standing is fragmented due to apparent hesitation.",
              "in the process of standing up, a person's motion is periodically halted, indicating hesitation."],
    (2, 2): [ "as a person stands up, they maintain an abnormal distribution of weight, suggesting a need for support.",
              "an individual transitioning to standing position shows signs of unusual weight distribution for balance.",
              "standing up from a seated position, the person exhibits an abnormal weight distribution indicating support."],
    (2, 3): [ "a person rising from a seated position showcases sudden, extraneous movements.",
              "an individual transitioning from sitting to standing exhibits sudden, unintended motion.",
              "as a person stands up, their movements appear jerky and out of the ordinary."],
    (2, 4): [ "while transitioning from sitting to standing, a person is seen massaging or shaking their limbs.",
              "a person stands up while intermittently rubbing or stimulating their body parts.",
              "as an individual rises to stand, they engage in rubbing or stimulating their limbs."],

    (3, 0): [ "a person transitions from standing to sitting, their movement showing signs of stiffness and interrupted motion.", 
              "an individual attempting to sit down from a standing position displays a guarded and rigid demeanor.", 
              "as a person sits down, their movements appear stiff and abruptly halted, indicating guarding."],
    (3, 1): [ "a person moving from standing to sitting exhibits hesitation, with movements broken into stages.",
              "the action of sitting down from standing is segmented due to visible hesitation.",
              "an individual displays hesitation, their movement to sit down appearing broken and paused."],
    (3, 2): [ "a person transitioning from standing to sitting showcases an abnormal weight distribution, suggesting a need for support.",
              "as an individual sits down, they maintain an unusual weight distribution for balance.",
              "in the act of sitting down, the person exhibits signs of abnormal weight distribution indicating support."],
    (3, 3): [ "a person moving from standing to sitting displays sudden, unintended movements.",
              "as an individual sits down, their movements appear jerky and extraneous.",
              "a person exhibits jerky, sudden motions while transitioning from standing to sitting."],
    (3, 4): [ "a person transitioning from standing to sitting is seen massaging or shaking their limbs.",
              "while sitting down, the individual engages in rubbing or stimulating their body parts.",
              "in the act of sitting down, a person is observed massaging or shaking their limbs."],

    (4, 0): [ "a person bends down, showcasing stiff and interrupted movement.", 
              "an individual bending down displays signs of guarding with stiffness and rigidity.", 
              "as a person bends down, their movement seems abruptly halted and rigid, indicating guarding behavior."],
    (4, 1): [ "a person bends down with hesitation, their movements broken into stages.",
              "an individual attempting to bend down shows a fragmented action due to visible hesitation.",
              "as a person bends down, their movement is periodically halted, suggesting hesitation."],
    (4, 2): [ "a person bends down, maintaining an abnormal distribution of weight as if needing support.",
              "an individual bending down showcases signs of abnormal weight distribution indicating balance support.",
              "while bending down, a person displays an unusual weight distribution, suggesting the need for support."],
    (4, 3): [ "a person bending down exhibits sudden, extraneous movements.",
              "in the action of bending down, an individual shows jerky and unintended motions.",
              "as a person bends down, their movements are characterized by sudden, extraneous motion."],
    (4, 4): [ "a person bending down is seen massaging or shaking their limbs.",
              "an individual bending down engages in rubbing or stimulating their body parts.",
              "as a person bends down, they are observed massaging or shaking their limbs."],

    (5, 0): [ "a person walks with a stiff and interrupted gait, showing signs of guarding.", 
              "an individual displays a guarded demeanor, their walking movement rigid and abruptly halted.", 
              "as a person walks, their movements seem stiff and interrupted, showcasing guarding behavior."],
    (5, 1): [ "a person walks with hesitation, their continuous strides broken into stages.",
              "an individual attempting to walk shows a fragmented gait due to visible hesitation.",
              "in the act of walking, a person's stride is periodically halted, indicating hesitation."],
    (5, 2): [ "a person walks maintaining an abnormal distribution of weight, suggesting a need for support.",
              "an individual's walking gait shows signs of unusual weight distribution, indicating a need for balance.",
              "while walking, a person displays an abnormal weight distribution, suggesting the need for support."],
    (5, 3): [ "a person walking displays sudden, unintended movements.",
              "an individual's gait features jerky and extraneous movements while walking.",
              "as a person walks, their movements are characterized by sudden, extraneous motion."],
    (5, 4): [ "a person walking is seen massaging or shaking their limbs.",
              "an individual engaged in walking exhibits rubbing or stimulating behavior on their body parts.",
              "while walking, a person is observed massaging or shaking their limbs."],

}

description_dict_v2 = {

    (0, 0): [ "a person stands on one leg, displaying interrupted or rigid movements.", 
              "an individual maintains a single leg stance, exhibiting signs of guarding with stiffness.", 
              "one leg supports the person, showcasing stiff and abruptly paused motion."],
    (0, 1): [ "a person attempts to stand on one leg, showing hesitation and movements broken into stages.",
              "an individual trying to maintain a one leg stand, the action appearing segmented due to hesitation.",
              "a person's one leg stance is characterized by stopped motions, portraying the behavior of hesitation."],
    (0, 2): [ "a person stands on one leg, maintaining an abnormal distribution of weight.",
              "an individual exhibits an unbalanced weight distribution while attempting a single leg stance.",
              "in a one leg stand, the person showcases a posture supporting an unusual weight allocation."],
    (0, 3): [ "a person stands on one leg, showing sudden movements extraneous to the intended motion.",
              "an individual attempts a one leg stance, displaying jerky, unintended movements.",
              "a person trying to stand on one leg, featuring sudden extraneous motion."],
    (0, 4): [ "a person maintaining a one leg stand while massaging or touching the standing leg.",
              "an individual stands on one leg, intermittently rubbing or stimulating the supporting limb.",
              "during a one leg stance, the person exhibits behavior of rubbing or shaking the standing leg."],
    (0, 5): [ "an individual stands on one leg.",
              "a person stands on one leg.",
              "a person attempts to stand on one leg.",
            ],

    (1, 0): [ "a person reaches forward, the movement appearing stiff and interrupted.", 
              "an individual extends their body forward, displaying rigidity and signs of guarding.", 
              "a person trying to reach forward showcases stiff and abruptly paused motions."],
    (1, 1): [ "a person attempting to reach forward, showing hesitation with movements broken into stages.",
              "an individual extending forward exhibits a paused motion, indicating hesitation.",
              "the action of reaching forward is segmented due to a person's visible hesitation."],
    (1, 2): [ "a person reaches forward, maintaining an abnormal distribution of weight for support.",
              "an individual leaning forward showcases unusual weight distribution suggesting the need for support.",
              "in an attempt to reach forward, the person shows signs of abnormal weight distribution for balance."],
    (1, 3): [ "a person reaching forward displays sudden, extraneous movements.",
              "as an individual leans forward, unexpected and jerky movements are exhibited.",
              "the motion of reaching forward is characterized by sudden, unintended movements."],
    (1, 4): [ "a person reaches forward while massaging or stimulating their arms or legs.",
              "as an individual extends forward, the behavior of rubbing or shaking limbs is observed.",
              "a person leans forward while intermittently touching or shaking their body parts."],
    (1, 5): [ "a person reaches forward.",
              "a person leans forward.",
              "an individual extending forward.",
            ],

    (2, 0): [ "a person transitions from sitting to standing, showing signs of stiffness and interrupted movement.", 
              "as an individual rises from a seated position, their movements exhibit guarding and rigidity.", 
              "in the act of standing up, a person's motion seems stiff and abruptly halted, showcasing guarding behavior."],
    (2, 1): [ "a person attempts to rise from a seated position, displaying hesitation with movements broken into stages.",
              "the act of transitioning from sitting to standing is fragmented due to apparent hesitation.",
              "in the process of standing up, a person's motion is periodically halted, indicating hesitation."],
    (2, 2): [ "as a person stands up, they maintain an abnormal distribution of weight, suggesting a need for support.",
              "an individual transitioning to standing position shows signs of unusual weight distribution for balance.",
              "standing up from a seated position, the person exhibits an abnormal weight distribution indicating support."],
    (2, 3): [ "a person rising from a seated position showcases sudden, extraneous movements.",
              "an individual transitioning from sitting to standing exhibits sudden, unintended motion.",
              "as a person stands up, their movements appear jerky and out of the ordinary."],
    (2, 4): [ "while transitioning from sitting to standing, a person is seen massaging or shaking their limbs.",
              "a person stands up while intermittently rubbing or stimulating their body parts.",
              "as an individual rises to stand, they engage in rubbing or stimulating their limbs."],
    (2, 5): [ "a person stands up.",
              "an individual rises to stand.",
              "a person transitions from sitting to standing.",
            ],

    (3, 0): [ "a person transitions from standing to sitting, their movement showing signs of stiffness and interrupted motion.", 
              "an individual attempting to sit down from a standing position displays a guarded and rigid demeanor.", 
              "as a person sits down, their movements appear stiff and abruptly halted, indicating guarding."],
    (3, 1): [ "a person moving from standing to sitting exhibits hesitation, with movements broken into stages.",
              "the action of sitting down from standing is segmented due to visible hesitation.",
              "an individual displays hesitation, their movement to sit down appearing broken and paused."],
    (3, 2): [ "a person transitioning from standing to sitting showcases an abnormal weight distribution, suggesting a need for support.",
              "as an individual sits down, they maintain an unusual weight distribution for balance.",
              "in the act of sitting down, the person exhibits signs of abnormal weight distribution indicating support."],
    (3, 3): [ "a person moving from standing to sitting displays sudden, unintended movements.",
              "as an individual sits down, their movements appear jerky and extraneous.",
              "a person exhibits jerky, sudden motions while transitioning from standing to sitting."],
    (3, 4): [ "a person transitioning from standing to sitting is seen massaging or shaking their limbs.",
              "while sitting down, the individual engages in rubbing or stimulating their body parts.",
              "in the act of sitting down, a person is observed massaging or shaking their limbs."],
    (3, 5): [ "a person transitions from standing to sitting.",
              "a person sits down.",
              "an individual attempting to sit down from a standing position.",
            ],

    (4, 0): [ "a person bends down, showcasing stiff and interrupted movement.", 
              "an individual bending down displays signs of guarding with stiffness and rigidity.", 
              "as a person bends down, their movement seems abruptly halted and rigid, indicating guarding behavior."],
    (4, 1): [ "a person bends down with hesitation, their movements broken into stages.",
              "an individual attempting to bend down shows a fragmented action due to visible hesitation.",
              "as a person bends down, their movement is periodically halted, suggesting hesitation."],
    (4, 2): [ "a person bends down, maintaining an abnormal distribution of weight as if needing support.",
              "an individual bending down showcases signs of abnormal weight distribution indicating balance support.",
              "while bending down, a person displays an unusual weight distribution, suggesting the need for support."],
    (4, 3): [ "a person bending down exhibits sudden, extraneous movements.",
              "in the action of bending down, an individual shows jerky and unintended motions.",
              "as a person bends down, their movements are characterized by sudden, extraneous motion."],
    (4, 4): [ "a person bending down is seen massaging or shaking their limbs.",
              "an individual bending down engages in rubbing or stimulating their body parts.",
              "as a person bends down, they are observed massaging or shaking their limbs."],
    (4, 5): [ "a person bends down.",
              "an individual attempting to bend down.",
              "an individual bends down.",
            ],

    (5, 0): [ "a person walks with a stiff and interrupted gait, showing signs of guarding.", 
              "an individual displays a guarded demeanor, their walking movement rigid and abruptly halted.", 
              "as a person walks, their movements seem stiff and interrupted, showcasing guarding behavior."],
    (5, 1): [ "a person walks with hesitation, their continuous strides broken into stages.",
              "an individual attempting to walk shows a fragmented gait due to visible hesitation.",
              "in the act of walking, a person's stride is periodically halted, indicating hesitation."],
    (5, 2): [ "a person walks maintaining an abnormal distribution of weight, suggesting a need for support.",
              "an individual's walking gait shows signs of unusual weight distribution, indicating a need for balance.",
              "while walking, a person displays an abnormal weight distribution, suggesting the need for support."],
    (5, 3): [ "a person walking displays sudden, unintended movements.",
              "an individual's gait features jerky and extraneous movements while walking.",
              "as a person walks, their movements are characterized by sudden, extraneous motion."],
    (5, 4): [ "a person walking is seen massaging or shaking their limbs.",
              "an individual engaged in walking exhibits rubbing or stimulating behavior on their body parts.",
              "while walking, a person is observed massaging or shaking their limbs."],
    (5, 5): [ "a person walks.",
              "an individual attempting to walk.",
              "an individual engaged in walking.",
            ],

    (5, 0, 4): [
            "a person walking primarily moves with stiffness, and occasionally rubs or massages the affected area.",
            "a person taking steps mainly exhibits rigidity, accompanied by moments of rubbing or stimulating the limbs.",
            "a person walks and mainly shows guarding or stiffness, interspersed with moments of massaging."
            ],
    (5, 0, 1): [
            "a person walking primarily shows stiffness in movements and hesitates periodically.",
            "a person taking steps mainly exhibits rigidity, punctuated by moments of hesitation.",
            "a person moving forward on foot predominantly shows guarding or stiffness, interrupted by brief pauses."
            ],
    (5, 0, 2): [
            "a person walking primarily moves with stiffness, and uses limbs for support or bracing.",
            "a person taking steps mainly exhibits rigidity, alongside supporting or bracing with limbs.",
            "a person walks and primarily shows guarding or stiffness, coupled with a supportive posture of limbs."
            ],
    
    (4, 0, 4): [
            "a person bending down mainly shows stiffness and simultaneously rubs or massages the affected area.",
            "a person stooping predominantly moves with rigidity and engages in stimulation of the limbs.",
            "a person lowering themselves primarily exhibits guarding or stiffness, accompanied by rubbing or massaging."
            ],
    (4, 0, 1): [
            "a person bending down mainly shows stiffness, interrupted by moments of hesitation.",
            "a person stooping predominantly moves with rigidity, interspersed with brief pauses.",
            "a person lowering themselves primarily exhibits guarding or stiffness, punctuated by hesitations."
            ],
    (4, 0, 2): [
            "a person bending down primarily moves with stiffness and occasionally uses limbs for support.",
            "a person stooping mainly exhibits rigidity, accompanied by moments of support or bracing with limbs.",
            "a person lowering themselves primarily shows guarding or stiffness, coupled with a supportive posture of limbs."
            ],
    (4, 0, 3): [
            "a person bending down primarily shows stiffness and occasionally exhibits jerky motions.",
            "a person stooping mainly exhibits rigidity, punctuated by sudden movements extraneous to the intended motion.",
            "a person lowering themselves primarily moves with stiffness, interspersed with jerky movements."
            ],

    (3, 0, 2): [
            "a person moving from standing to sitting primarily moves with stiffness and occasionally uses limbs for support.",
            "a person transitioning from standing to sitting mainly exhibits rigidity, alongside moments of support or bracing with limbs.",
            "a person going from a standing to a sitting position primarily shows guarding or stiffness, coupled with a supportive posture."
            ],
    (3, 0, 1): [
            "a person moving from standing to sitting primarily shows stiffness, interrupted by moments of hesitation.",
            "a person transitioning from standing to sitting predominantly moves with rigidity, interspersed with brief pauses.",
            "a person going from a standing to a sitting position mainly exhibits guarding or stiffness, punctuated by hesitations."
            ],
    (3, 0, 3): [
            "a person moving from standing to sitting primarily shows stiffness and occasionally exhibits jerky motions.",
            "a person transitioning from standing to sitting mainly exhibits rigidity, punctuated by sudden movements extraneous to the intended motion.",
            "a person going from a standing to a sitting position primarily moves with stiffness, interspersed with jerky movements."
            ],
    (3, 1, 2): [
            "a person moving from standing to sitting primarily hesitates and occasionally uses limbs for support.",
            "a person transitioning from standing to sitting mainly exhibits pauses in movement, with moments of support or bracing.",
            "a person going from a standing to a sitting position predominantly hesitates, coupled with occasional limb support."
            ],
    (3, 0, 4): [
        "a person moving from standing to sitting primarily shows stiffness and occasionally engages in rubbing or stimulation.",
        "a person transitioning from standing to sitting predominantly moves with rigidity, interspersed with massaging or shaking affected body parts.",
        "a person going from a standing to a sitting position mainly exhibits guarding or stiffness, punctuated by rubbing or stimulation."
        ],

    (2, 0, 2): [
        "a person moving from sitting to standing primarily moves with stiffness and occasionally uses limbs for support.",
        "a person transitioning from sitting to standing mainly exhibits rigidity, alongside moments of support or bracing with limbs.",
        "a person going from a sitting to a standing position primarily shows guarding or stiffness, coupled with a supportive posture."
        ],
    (2, 0, 1): [
        "a person moving from sitting to standing primarily shows stiffness, interrupted by moments of hesitation.",
        "a person transitioning from sitting to standing predominantly moves with rigidity, interspersed with brief pauses.",
        "a person going from a sitting to a standing position mainly exhibits guarding or stiffness, punctuated by hesitations."
        ],
    (2, 1, 4): [
        "a person moving from sitting to standing primarily hesitates and occasionally engages in rubbing or stimulation.",
        "a person transitioning from sitting to standing predominantly exhibits pauses in movement, interspersed with massaging or shaking affected body parts.",
        "a person going from a sitting to a standing position mainly hesitates, punctuated by rubbing or stimulation."
        ],
    (2, 0, 4): [
        "a person moving from sitting to standing primarily moves with stiffness and occasionally engages in rubbing or stimulation.",
        "a person transitioning from sitting to standing mainly exhibits rigidity, interspersed with massaging or shaking affected body parts.",
        "a person going from a sitting to a standing position mainly exhibits guarding or stiffness, punctuated by rubbing or stimulation."
        ],
    (2, 0, 3): [
        "a person moving from sitting to standing primarily moves with stiffness and occasionally makes jerky motions.",
        "a person transitioning from sitting to standing mainly exhibits rigidity, punctuated by sudden movements extraneous to the intended motion.",
        "a person going from a sitting to a standing position primarily shows guarding or stiffness, with intermittent jerky movements."
        ],

    (1, 0, 1): [
        "a person reaching forward primarily shows stiffness, interrupted by moments of hesitation.",
        "a person extending their arm forward predominantly moves with rigidity, interspersed with brief pauses.",
        "a person making a forward reaching motion mainly exhibits guarding or stiffness, punctuated by hesitations."
        ],
    (1, 0, 3): [
        "a person reaching forward primarily moves with stiffness and occasionally makes jerky motions.",
        "a person extending their arm forward predominantly moves with rigidity, interspersed with sudden movements extraneous to the intended motion.",
        "a person making a forward reaching motion mainly exhibits guarding or stiffness, with intermittent jerky movements."
        ],
    (1, 0, 4): [
        "a person reaching forward primarily moves with stiffness and occasionally engages in rubbing or stimulation.",
        "a person extending their arm forward predominantly moves with rigidity, interspersed with massaging or shaking affected body parts.",
        "a person making a forward reaching motion mainly exhibits guarding or stiffness, punctuated by rubbing or stimulation."
        ],

    (0, 0, 3): [
        "a person standing on one leg primarily shows stiffness, interrupted by moments of jerky motion.",
        "a person balancing on a single leg predominantly exhibits rigidity, interspersed with sudden movements extraneous to the intended motion.",
        "a person performing a one leg stand mainly displays guarding or stiffness, punctuated by jerky movements."
        ],
    (0, 0, 4): [
        "a person standing on one leg primarily moves with stiffness and occasionally engages in rubbing or stimulation.",
        "a person balancing on a single leg predominantly exhibits rigidity, interspersed with massaging or shaking affected body parts.",
        "a person performing a one leg stand mainly displays guarding or stiffness, punctuated by rubbing or stimulation."
        ],
    (0, 1, 3): [
        "a person standing on one leg primarily hesitates and occasionally makes jerky motions.",
        "a person balancing on a single leg predominantly shows moments of pause, interspersed with sudden movements extraneous to the intended motion.",
        "a person performing a one leg stand mainly exhibits hesitation, with intermittent jerky movements."
        ],
    (0, 0, 1): [
        "a person standing on one leg primarily shows stiffness, interrupted by moments of hesitation.",
        "a person balancing on a single leg predominantly moves with rigidity, interspersed with brief pauses.",
        "a person performing a one leg stand mainly exhibits guarding or stiffness, punctuated by hesitations."
        ],
    (0, 0, 2): [
        "a person standing on one leg primarily moves with stiffness and occasionally uses a limb for support.",
        "a person balancing on a single leg predominantly exhibits rigidity, interspersed with positions where a limb supports an abnormal distribution of weight.",
        "a person performing a one leg stand mainly displays guarding or stiffness, punctuated by support or bracing."
        ]
    
}



def check_keywords_in_glove(text_dict):
    keywords = []
    for key, sentences in text_dict.items():
        for sentence in sentences:
            # 使用nltk的word_tokenize函数将句子拆解为单词
            words = nltk.word_tokenize(sentence)
            # 将这些单词添加到关键词列表中
            keywords.extend(words)
    
    # 去掉重复的词
    unique_keywords = list(set(keywords))

    # 调用你已经实现的compare_vectors()函数，输入关键词列表
    compare_vectors(unique_keywords, word_to_idx, word_vecs, glove_model)

    # 添加关键词到本地
    # add_words_to_vocab(unique_keywords, glove_model, vecs_path, idx_path, words_path)

    # 查看当前的关键词
    # print(unique_keywords)

# 调用函数
check_keywords_in_glove(VIP_dict)

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



####################################动作数据相关########################################

# 初始化最大和最小值
# max_value = -np.inf
# min_value = np.inf

# 指定数据目录
# directory = "./dataset/EmoPain/joints/"

# 遍历数据目录中的所有文件
# for filename in os.listdir(directory):
    # 确保文件是.npy文件
    # if filename.endswith(".npy"):
        # 加载.npy文件
        # data = np.load(os.path.join(directory, filename))

        # 更新最大值和最小值
        # max_value = max(max_value, data.max())
        # min_value = min(min_value, data.min())

# 打印数值范围
# print("The range of the data is from ", min_value, " to ", max_value)

# folder_path='./dataset/EmoPain/labels/'
# data_path='./dataset/EmoPain/new_joints/'
# filename='P13-0030-aclabel.npy'
# dataname='P13-0030.npy'
# data = np.load(os.path.join(data_path, dataname))
# label = np.load(os.path.join(folder_path, filename))
# print(f'Label Shape: {label.shape}')
# print(f'Data Shape: {data.shape}')
# print(label)