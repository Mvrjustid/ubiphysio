import os
import json
import numpy as np
import models.vqvae as vqvae
import utils.losses as losses 
import options.option_vq as option_vq
args = option_vq.get_args_parser()
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils.utils_model as utils_model
from dataset import dataset_VQ, dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer

device = torch.device('cuda')

def encode_audio(model, audio_file, device="cuda"):
        model = model.to(device)
        # Load the audio features
        features = np.load(audio_file)
        # Convert features to torch tensor and add batch dimension
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
        # Make sure the model is in evaluation mode
        model.eval()
        with torch.no_grad():
            # Encode audio using the model
            code_idx = model.encode(features_tensor)
            # Transfer code_idx back to CPU and convert to numpy array
            code_idx = code_idx.cpu().numpy()
        return code_idx

print('---Extracting Code Indexes---')
save_dir = './output/VQVAE_win192_alpha05/'
motion_dir = './dataset/ubiphysio/new_joint_vecs'
all_file_list = './dataset/ubiphysio/all.txt'
audio_code_dir = './dataset/ubiphysio/motion_code_VQVAE_win192_alpha05/'
os.makedirs(audio_code_dir, exist_ok=True)

# Load the trained model
model_path = save_dir+"_best_model.pt"
model = torch.load(model_path)

with open(all_file_list, 'r') as f:
    file_list = [line.strip() for line in f.readlines()]
for file_name in tqdm(file_list,desc='Encoding motion files into tokens'):
    features_file = os.path.join(motion_dir, file_name + '.npy')
    code_idx = encode_audio(model, features_file, device="cuda")
    np.save(os.path.join(audio_code_dir, file_name), code_idx)