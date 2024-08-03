import numpy as np
import os
import random
import spacy
from scipy import stats

# 创建描述字典，键为运动-行为的标签对，值为对应的三种描述
description_dict = {

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

def process_files1(input_dir, output_dir):
    # 根据前缀（例如：'C5-0006'）创建一个集合
    prefix_set = set(filename.rsplit('-', 1)[0] for filename in os.listdir(input_dir))
    for prefix in prefix_set:
        # 加载运动种类标签的.npy文件
        activity_data = np.load(os.path.join(input_dir, f"{prefix}-aclabel.npy"))
        # 提取运动种类的标签，取众数
        activity_label = stats.mode(activity_data)[0][0]  
        
        # 加载行为标签的.npy文件
        behavior_data = np.load(os.path.join(input_dir, f"{prefix}-label.npy"))
        # 提取行为的标签，找到1出现最多的列
        # 如果所有行和列都是0，将行为标签设置为5
        if np.all(behavior_data == 0):
            behavior_label = 5
        else:
            behavior_label = np.argmax(np.sum(behavior_data == 1, axis=0))

        # 从字典中获取对应的描述
        descriptions = description_dict.get((activity_label, behavior_label))
        if descriptions is None:
            print(f"No description found for labels ({activity_label}, {behavior_label}).")
            continue

        # 随机选择一个描述
        description = random.choice(descriptions)
            
        print(f"Activity label is '{activity_label}', behavior label is '{behavior_label}', description: '{description}' ")

        # 将描述写入.txt文件
        with open(os.path.join(output_dir, f"{prefix}.txt"), 'w') as f:
            f.write(description)

# 处理一种及以上行为出现的情况
def process_files1_v2(input_dir, output_dir):
    # 根据前缀（例如：'C5-0006'）创建一个集合
    prefix_set = set(filename.rsplit('-', 1)[0] for filename in os.listdir(input_dir))
    for prefix in prefix_set:
        # 加载运动种类标签的.npy文件
        activity_data = np.load(os.path.join(input_dir, f"{prefix}-aclabel.npy"))
        # 提取运动种类的标签，取众数
        activity_label = int(stats.mode(activity_data)[0][0])
        
        # 加载行为标签的.npy文件
        behavior_data = np.load(os.path.join(input_dir, f"{prefix}-label.npy"))
        # 按列计算1的数量
        ones_count = np.sum(behavior_data == 1, axis=0)
        non_zero_indices = np.nonzero(ones_count)[0]
        
        # 提取行为的标签
        if len(non_zero_indices) == 0:
            behavior_labels = (5,)
        elif len(non_zero_indices) == 1:
            behavior_labels = (non_zero_indices[0],)
        else:
            sorted_indices = np.argsort(ones_count)[::-1]
            behavior_labels = tuple(sorted([sorted_indices[0], sorted_indices[1]]))

        # 从字典中获取对应的描述
        combination = (activity_label,) + behavior_labels
        descriptions = description_dict.get(combination)
        if descriptions is None:
            print(f"No description found for labels {combination}.")
            continue

        # 随机选择一个描述
        description = random.choice(descriptions)
            
        print(f"Labels are {combination}, description: '{description}' ")

        # 将描述写入.txt文件
        with open(os.path.join(output_dir, f"{prefix}.txt"), 'w') as f:
            f.write(description)

#step1: 调用函数，处理.npy文件并生成.txt文件
process_files1_v2("./dataset/EmoPain/labels/", "./dataset/EmoPain/raw_texts/")

#step 2：词性标注
# 加载预训练模型
nlp = spacy.load('en_core_web_sm')

def process_files2(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            # 读取文件
            with open(os.path.join(input_dir, filename), 'r') as f:
                original_text = f.read()
            
            # 进行词性标注
            doc = nlp(original_text)
            processed_text = " ".join([f"{token.text}/{token.pos_}" for token in doc])
            
            # 将词性标注文本添加到原始文本后面
            result_text = original_text + "#" + processed_text + "#0.0#0.0"

            # 写入新的文件
            with open(os.path.join(output_dir, filename), 'w') as f:
                f.write(result_text)

# 调用函数，处理.txt文件并生成新的.txt文件
process_files2("./dataset/EmoPain/raw_texts/", "./dataset/EmoPain/texts/")