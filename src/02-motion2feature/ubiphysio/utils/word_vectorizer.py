import numpy as np
import pickle
from os.path import join as pjoin

# With more words added both at here and for the GloVe dict.
# interruptedly, extraneously两个词未被收录进GloVe6B
# 加在了最后一行

POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
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


class WordVectorizer(object):
    def __init__(self, meta_root, prefix):
        vectors = np.load(pjoin(meta_root, '%s_data.npy'%prefix))
        words = pickle.load(open(pjoin(meta_root, '%s_words.pkl'%prefix), 'rb'))
        self.word2idx = pickle.load(open(pjoin(meta_root, '%s_idx.pkl'%prefix), 'rb'))
        self.word2vec = {w: vectors[self.word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator['OTHER']] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):
        word, pos = item.split('/')
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_ohot(vip_pos)
            else:
                pos_vec = self._get_pos_ohot(pos)
        else:
            word_vec = self.word2vec['unk']
            pos_vec = self._get_pos_ohot('OTHER')
        return word_vec, pos_vec


class WordVectorizerV2(WordVectorizer):
    def __init__(self, meta_root, prefix):
        super(WordVectorizerV2, self).__init__(meta_root, prefix)
        self.idx2word = {self.word2idx[w]: w for w in self.word2idx}

    def __getitem__(self, item):
        word_vec, pose_vec = super(WordVectorizerV2, self).__getitem__(item)
        word, pos = item.split('/')
        if word in self.word2vec:
            return word_vec, pose_vec, self.word2idx[word]
        else:
            return word_vec, pose_vec, self.word2idx['unk']

    def itos(self, idx):
        if idx == len(self.idx2word):
            return "pad"
        return self.idx2word[idx]