import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import librosa
from torch.utils.data import Dataset
from pathlib import Path
from torch.optim.swa_utils import AveragedModel
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import pad
from tqdm import tqdm
import logging

def create_logger(log_dir, log_file):
    """
    Create a logger with a stream handler and a file handler.

    Args:
    log_dir (str): The directory where the log file will be created.
    log_file (str): The name of the log file.
    
    Returns:
    logger (logging.Logger): The created logger.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

device = torch.device('cuda')

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=num_residual_hiddens,
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, 
                      stride=1, 
                      bias=False)
        )
    
    def forward(self, x):
        return self._block(x) + x

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

    def reset_codebook(self, indices):
        self._embedding.weight.data[indices].normal_()
        self._ema_cluster_size.data[indices].zero_()
        self._ema_w.data[indices].normal_()

    # @torch.no_grad()
    def update_codebook(self, flat_input, encodings):
        with torch.no_grad():
            # Update the EMA
            self._ema_cluster_size = self.decay * self._ema_cluster_size + \
                                    (1 - self.decay) * torch.sum(encodings, 0)
            encodings = encodings.to(torch.float32)
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w.data = self.decay * self._ema_w.data + (1 - self.decay) * dw
            # Update the embedding
            self._embedding.weight.data = self._ema_w / (self._ema_cluster_size.unsqueeze(-1) + self.epsilon)

            # print(f"词嵌入使用次数的缓存变量: {self._ema_cluster_size.cpu().tolist()}")
            
            # Codebook Reset
            use_threshold = 1.0  # This threshold needs to be tuned.
            underused = (self._ema_cluster_size < use_threshold).nonzero(as_tuple=True)[0]
            # print(f"Underused: f{underused}")

            if underused.nelement() > 0:
                self.reset_codebook(underused)
            
            # Compute the reset ratio
            reset_mask = (self._ema_cluster_size < use_threshold).float()
            num_reset = torch.sum(reset_mask)
            reset_ratio = (num_reset / self._num_embeddings).detach()
        return reset_ratio

    def init_codebook(self, x):
        with torch.no_grad():
            flat_x = x.view(-1, self._embedding_dim)
            self._embedding.weight.data = flat_x[:self._num_embeddings]  # assuming flat_x has size larger than num_embeddings
            self.init = True
    
    def quantize(self, x):
        # Calculate distances
        distances = (torch.sum(x**2, dim=-1, keepdim=True) 
                        + torch.sum(self._embedding.weight**2, dim=1)
                        - 2 * torch.matmul(x, self._embedding.weight.t()))
        # Monitor
        # print(f"Distance size before ouput is {distances.size()}")
        # print(f"Distances is {distances}")
        # _, code_idx = torch.min(distances, dim=-1)
        # print(f"Codes are {code_idx}")

        # Encoding
        return torch.argmin(distances, dim=1).unsqueeze(1)

    def forward(self, inputs):
        # input_shape = inputs.shape
        # Flatten input
        # flat_input = inputs.view(-1, self._embedding_dim)
        # print(f"The number of features is {flat_input.shape[0]}")

        N, width, T = inputs.shape
        # NCT -> NTC -> [NT, C]
        inputs = inputs.permute(0, 2, 1).contiguous()
        flat_input = inputs.view(-1, inputs.shape[-1])  
        
        encoding_indices = self.quantize(flat_input)
        
         # De-Quantize and unflatten
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(N,T,-1)

        if self.training:
            reset_ratio = self.update_codebook(flat_input, encodings)
        else:
            reset_ratio = torch.zeros(1, device=inputs.device).detach()

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        quantized=quantized.permute(0,2,1)
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, reset_ratio, encodings 

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv1d(in_channels=in_channels, 
                                 out_channels=num_hiddens//2, 
                                 kernel_size=4, 
                                 stride=4, padding=1)
        self._conv_2 = nn.Conv1d(in_channels=num_hiddens//2, 
                                 out_channels=num_hiddens, 
                                 kernel_size=4, 
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv1d(in_channels=num_hiddens, 
                                 out_channels=num_hiddens, 
                                 kernel_size=3, 
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens, 
                                             num_hiddens=num_hiddens, 
                                             num_residual_layers=num_residual_layers, 
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        # print(f"Encode 1 size: {x.size()}")
        x = self._conv_2(x)
        x = F.relu(x)
        # print(f"Encode 2 size: {x.size()}")
        x = self._conv_3(x)
        # print(f"Encode 3 size: {x.size()}")
        x = self._residual_stack(x)
        # print(f"Encode 4 size: {x.size()}")
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv1d(in_channels=in_channels, 
                                 out_channels=num_hiddens, 
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens, 
                                             num_hiddens=num_hiddens, 
                                             num_residual_layers=num_residual_layers, 
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2, 
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens//2, 
                                                out_channels=out_channels, 
                                                kernel_size=4, 
                                                stride=4, padding=0)
        
        self._conv_adjust = nn.Conv1d(in_channels=out_channels, 
                              out_channels=out_channels, 
                              kernel_size=3, stride=1, padding=3)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        # print(f"Decode 1 size: {x.size()}")
        x = self._residual_stack(x)
        # print(f"Decode 2 size: {x.size()}")
        x = self._conv_trans_1(x)
        x = F.relu(x)
        # print(f"Decode 3 size: {x.size()}")
        x = self._conv_trans_2(x)
        # print(f"Decode 4 size: {x.size()}")
        return self._conv_adjust(x)

class VQ_VAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost):
        super(VQ_VAE, self).__init__()
        
        self._encoder = Encoder(60, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        # self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, 
                                    #    commitment_cost)
        
        self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay=0.99)

        self._decoder = Decoder(60,
                                embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

    def preprocess(self, x):
        # (bs, T, D) -> (bs, D, T), for Conv1D
        x = x.permute(0,2,1).float()
        return x

    def postprocess(self, x):
        # (bs, D, T) ->  (bs, T, D), for Conv1D
        x = x.permute(0,2,1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self._encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self._vq_vae.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def forward(self, x):
        # print(f"The input size seen in model is {x.size()}")
        z = self.preprocess(x)
        z = self._encoder(z)
        # print(f"The encoder size seen in model is {z.size()}")
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, reset_ratio, _ = self._vq_vae(z)
        # print(f"The quantizer size seen in model is {quantized.size()}")
        x_recon = self._decoder(quantized)
        # print(f"The decoder size seen in model is {x_recon.size()}")
        x_out = self.postprocess(x_recon)
        return loss, x_out, perplexity, reset_ratio

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Set the size of within-model batch 
        batch_size = 256

        # Calculate distances in batches
        distances = torch.zeros(flat_input.shape[0], self._num_embeddings).to(inputs.device)
        for i in range(0, flat_input.shape[0], batch_size):
            batch = flat_input[i:i+batch_size]
            batch_distances = (torch.sum(batch**2, dim=1, keepdim=True) 
                            + torch.sum(self._embedding.weight**2, dim=1)
                            - 2 * torch.matmul(batch, self._embedding.weight.t()))
            distances[i:i+batch_size] = batch_distances
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._embedding.num_embeddings).to(inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(*input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings

def train(model, train_loader, valid_loader, optimizer, num_epochs, logger,exp):
    best_loss = float('inf')
    save_dir = './output/audio_vqvae_new/'
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info('Training...')
    logger.info(exp)

    for epoch in range(num_epochs):    
        # Train mode
        model.train()
        epoch_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_recon_loss = 0.0
        train_batch_num = len(train_loader)
        
        for waveforms in train_loader:
            waveforms = waveforms.to(device)
            optimizer.zero_grad()
            vq_loss, data_recon, perplexity, reset_ratio = model(waveforms)
            recon_loss = F.smooth_l1_loss(data_recon, waveforms)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / train_batch_num
            epoch_vq_loss += vq_loss.item() / train_batch_num
            epoch_recon_loss += recon_loss.item() / train_batch_num

        logger.info(f"Epoch {epoch+1}/{num_epochs} Train Loss: {epoch_loss} VQ Loss: {epoch_vq_loss} Recon Loss: {epoch_recon_loss}")
        logger.info(f"The codebook reset ratio is {reset_ratio.item()}")

        # Validation mode
        logger.info('Evaluating...')
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_vq_loss = 0.0
        epoch_val_recon_loss = 0.0
        valid_batch_num = len(valid_loader)
        with torch.no_grad():
            for waveforms in valid_loader:
                waveforms = waveforms.to(device)
                # print(f"The feature size seen outside for model is {waveforms.size()}")
                vq_loss, data_recon, perplexity, reset_ratio = model(waveforms)
                # Monitor the codes
                code_idx = model.encode(waveforms)
                code_idx = code_idx.cpu().numpy()
                print(f"The codes of validation files are {code_idx}")
                recon_loss = F.smooth_l1_loss(data_recon, waveforms)
                loss = recon_loss + vq_loss
                epoch_val_loss += loss.item() / valid_batch_num
                epoch_val_vq_loss += vq_loss.item() / valid_batch_num
                epoch_val_recon_loss += recon_loss.item() / valid_batch_num

        logger.info(f"Epoch {epoch+1}/{num_epochs} Validation Loss: {epoch_val_loss} VQ Loss: {epoch_val_vq_loss} Recon Loss: {epoch_val_recon_loss}")
        
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            logger.info(f"Average Loss Reduced to {best_loss}!!!")
            torch.save(model.state_dict(), os.path.join(save_dir, exp+'_best_model.pt'))

        # Save the latest model
        torch.save(model.state_dict(), os.path.join(save_dir, exp+'_last_model.pt'))

def collate_fn(batch):
    # Get the number of frames for each feature matrix in the batch
    # feature size: TxD
    # print(features.size() for features in batch)
    frame_nums = [features.shape[0] for features in batch]
    # Get the maximum number of frames
    max_frame_num = max(frame_nums)
    # Pad the features in the batch
    batch = [pad(features, (max_frame_num - features.shape[0],0)) if features.shape[0] < max_frame_num else features for features in batch]
    # Stack the feature matrices
    features = torch.stack(batch)
    return features

def process_audio(features_path):
    # Load the features file
    features = np.load(features_path)
    return torch.from_numpy(features.astype(np.float32)).T # T X D

class AudioDataset_split(Dataset):
    def __init__(self, file_list, audio_dir, extension='.npy'):
        self.file_list = file_list
        self.audio_dir = audio_dir
        self.extension = extension

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        features_file = os.path.join(self.audio_dir, self.file_list[idx] + self.extension)
        return process_audio(features_file)

def create_data_loader(file_list_path, audio_dir, batch_size=64):
    with open(file_list_path, 'r') as f:
        file_list = [line.strip() for line in f.readlines()]
    dataset = AudioDataset_split(file_list, audio_dir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return data_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# Define hyperparameters
batch_size = 32
num_epochs = 300
num_hiddens = 128
num_embeddings = 128
embedding_dim = 128
num_residual_layers = 2
num_residual_hiddens = 32
commitment_cost = 1.0
learning_rate = 2e-4
is_train = False
experiment = "bs32_epoch300_dim128_nb128_reset1_RL2_nh32_commitc100"

"0.0044 - bs32_epoch100_dim512_nb512_reset1_RL8_nh32_commitc100"

logger = create_logger('./output/audio_vqvae_new/', 'train.log')

# Create the model
model = VQ_VAE(num_hiddens, num_residual_layers, num_residual_hiddens,
               num_embeddings, embedding_dim, commitment_cost)
model = model.to(device)

# ema_model = AveragedModel(model)

# Create the optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.0)

# Create the data loader
audio_dir = './dataset/ubiphysio/audio_new'
# dataset = AudioDataset(audio_dir)
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

train_file_list = './dataset/ubiphysio/train.txt'
valid_file_list = './dataset/ubiphysio/val.txt'
train_loader = create_data_loader(train_file_list, audio_dir, batch_size)
valid_loader = create_data_loader(valid_file_list, audio_dir, batch_size)

if is_train:

# Train the model
    train(model, train_loader, valid_loader, optimizer, num_epochs,logger=logger,exp=experiment)
    
else:
    # Yes, extracting audio tokens!
    def encode_audio(model, audio_file, device="cuda"):
        model = model.to(device)
        # Load the audio features
        features = np.load(audio_file)
        print(f"The size is {np.shape(features)}")
        # Convert features to torch tensor and add batch dimension
        features_tensor=torch.from_numpy(features.astype(np.float32)).to(device).T.unsqueeze(dim=0)
        # Make sure the model is in evaluation mode
        model.eval()
        with torch.no_grad():
            # Encode audio using the model
            code_idx = model.encode(features_tensor)
            # Transfer code_idx back to CPU and convert to numpy array
            code_idx = code_idx.cpu().squeeze(0).numpy()
            print(f"The size of the code is {np.shape(code_idx)}")
            print(f"The first 10 codes of validation files are {code_idx}")
        return code_idx

    print('---Extracting Code Indexes---')
    save_dir = './output/audio_vqvae_new/'
    audio_dir = './dataset/ubiphysio/audio_new'
    all_file_list = './dataset/ubiphysio/all.txt'
    audio_code_dir = './dataset/ubiphysio/AudioCode_VQVAE_RL2_Last/'
    os.makedirs(audio_code_dir, exist_ok=True)

    # Load the trained model
    model_path = experiment+"_last_model.pt"
    model.load_state_dict(torch.load(save_dir+model_path))
    
    with open(all_file_list, 'r') as f:
        file_list = [line.strip() for line in f.readlines()]

    for file_name in tqdm(file_list,desc='Encoding audio files into tokens'):
        print(f'Now processing: {file_name}')
        features_file = os.path.join(audio_dir, file_name + '.npy')
        code_idx = encode_audio(model, features_file, device="cuda")
        # save as txt
        np.savetxt(os.path.join(audio_code_dir, file_name+'.txt'), code_idx, fmt='%d')
        # save as npy
        # np.save(os.path.join(audio_code_dir, file_name), code_idx)

# def collate_fn(batch):
#     # Set maximum length to 240000 (15 seconds at 44100 Hz)
#     max_length = 661500
#     # Pad shorter waveforms or trim longer waveforms
#     batch = [(F.pad(waveform, (0, max_length - waveform.shape[1])) if waveform.shape[1] < max_length else waveform[:,:max_length], text) for waveform, text in batch]
#     # Stack waveforms and texts
#     waveforms = torch.stack([waveform for waveform, _ in batch])
#     texts = [text for _, text in batch]
#     return waveforms, texts

# def train(model, data_loader, optimizer, num_epochs):
#     model.train()
#     for epoch in range(num_epochs):
#         for waveforms in data_loader:
#             waveforms = waveforms.to(device)
#             optimizer.zero_grad()
#             vq_loss, data_recon, perplexity = model(waveforms)
#             loss = F.mse_loss(data_recon, waveforms) + vq_loss
#             loss.backward()
#             optimizer.step()

# def create_data_loader(file_list_path, batch_size=64):
#     with open(file_list_path, 'r') as f:
#         file_list = [line.strip() for line in f.readlines()]
#     dataset = AudioDataset(file_list)
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#     return data_loader

# def process_audio(audio_path, sample_rate=12000):
#     # Load the audio file
#     waveform, sr = torchaudio.load(audio_path)
    
#     # Convert to mono
#     mono_waveform = torch.mean(waveform, dim=0, keepdim=True)
    
#     # Resample the audio
#     resampler = T.Resample(sr, sample_rate)
#     resampled_waveform = resampler(mono_waveform)
    
#     return resampled_waveform

# def train(model, train_loader, valid_loader, optimizer, num_epochs,logger):
#     model.train()
#     best_loss = float('inf')
#     save_dir = './output/audio_vqvae/'
#     os.makedirs(save_dir, exist_ok=True)
#     for epoch in range(num_epochs):
#         for loader, mode in [(train_loader, 'train'), (valid_loader, 'valid')]:
#             epoch_loss = 0.0
#             epoch_vq_loss = 0.0
#             epoch_recon_loss = 0.0
#             for waveforms in loader:
#                 # print(f"The input size outside the model is {np.shape(waveforms)}")
#                 waveforms = torch.from_numpy(waveforms.astype(np.float32)).to(device)
#                 optimizer.zero_grad()
#                 vq_loss, data_recon, perplexity, reset_ratio = model(waveforms)
#                 # print(f"The output size is {data_recon.size()}")
#                 recon_loss = F.smooth_l1_loss(data_recon, waveforms)
#                 loss = recon_loss + vq_loss
#                 if mode == 'train':
#                     loss.backward()
#                     optimizer.step()
#                 epoch_loss += loss.item()
#                 epoch_vq_loss += vq_loss.item()
#                 epoch_recon_loss += recon_loss.item()
            
#             average_epoch_loss = epoch_loss / len(loader)
#             print(f"Epoch {epoch+1}/{num_epochs} {mode.capitalize()} Loss: {average_epoch_loss} VQ Loss: {epoch_vq_loss/len(loader)} Recon Loss: {epoch_recon_loss/len(loader)}")
#             print(f"The codebook reset ratio is {reset_ratio.item()}")
            
#             if mode == 'valid' and average_epoch_loss < best_loss:
#                 best_loss = average_epoch_loss
#                 print(f"Average Loss Reduced from {best_loss} to {average_epoch_loss}!!!")
#                 torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

#         # Save the latest model
#         torch.save(model.state_dict(), os.path.join(save_dir, 'latest_model.pt'))

# def collate_fn(batch):
#     # Get the number of frames for each feature matrix in the batch
#     frame_nums = [features.shape[1] for features in batch]
#     # Get the maximum number of frames
#     max_frame_num = max(frame_nums)
#     # Pad the features in the batch
#     batch = [np.pad(features, ((0, 0), (0, max_frame_num - features.shape[1]))) if features.shape[1] < max_frame_num else features for features in batch]
#     # Stack the feature matrices
#     features = np.stack(batch)
#     return features

# def encode_audio(model, audio_path):
#     waveform = process_audio(audio_path)
#     waveform = waveform.to(device)
#     z = model._encoder(waveform)
#     z = model._pre_vq_conv(z)
#     _, indices = model._vq_vae._embedding(z.permute(0, 2, 1))
#     return indices.cpu().numpy()

# class AudioDataset(Dataset):
#     def __init__(self, audio_dir, transform=None):
#         self.audio_dir = Path(audio_dir)
#         self.audio_files = list(self.audio_dir.glob('*.wav'))
#         self.transform = transform

#     def __len__(self):
#         return len(self.audio_files)

#     def __getitem__(self, idx):
#         audio_file = self.audio_files[idx]
#         waveform = process_audio(audio_file)
        
#         if self.transform:
#             waveform = self.transform(waveform)

#         return waveform

# def process_audio(audio_path, sample_rate=12000, duration=15):
#     # Load the audio file
#     waveform, sr = librosa.load(audio_path, sr=sample_rate)

#     # If the audio is shorter than the max length, zero-pad it
#     if len(waveform) < sr * duration:
#         padding = sr * duration - len(waveform)
#         waveform = np.pad(waveform, (0, padding))
#     # If the audio is longer than the max length, take the first 'duration' seconds only
#     else:
#         waveform = waveform[:sr * duration]

#     # Extract features
#     mfcc_feature = librosa.feature.mfcc(waveform, sr=sr)
#     chroma = librosa.feature.chroma_stft(waveform, sr=sr)
#     contrast = librosa.feature.spectral_contrast(waveform, sr=sr, n_bands=5, fmin=100)
#     tonnetz = librosa.feature.tonnetz(waveform, sr=sr)

#     # Concatenate features
#     features = np.concatenate([mfcc_feature, chroma, contrast, tonnetz], axis=0)

#     scaler = StandardScaler()
#     features = scaler.fit_transform(features)

#     # print(np.shape(features)) T:44 X D:352

#     return  torch.from_numpy(features.astype(np.float32)).to(device)

# class AudioDataset_split(Dataset):
#     def __init__(self, file_list, audio_dir, extension='.wav'):
#         self.file_list = file_list
#         self.audio_dir = audio_dir
#         self.extension = extension

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         audio_file = os.path.join(self.audio_dir, self.file_list[idx] + self.extension)
#         return process_audio(audio_file)

# Use EMA to update the embedding vectors
# if self.training:
#     self._ema_cluster_size = self.decay * self._ema_cluster_size + \
#                             (1 - self.decay) * torch.sum(encodings, 0)
#     # epsilon is a small number to avoid division by zero
#     # n = torch.sum(encodings, 0)
#     # dw = torch.sum(encodings.unsqueeze(-1) * flat_input.unsqueeze(1), 0)

#     # encodings = encodings.to(torch.float32)
#     # dw = torch.zeros_like(self._ema_w).to(inputs.device)
#     # for i in range(self._num_embeddings):
#     #     selected = encodings[:, i].unsqueeze(1)
#     #     dw[i] = torch.sum(selected * flat_input, 0)
#     # self._ema_w.data = self.decay * self._ema_w.data + (1 - self.decay) * dw

#     encodings = encodings.to(torch.float32)
#     dw = torch.matmul(encodings.t(), flat_input)
#     self._ema_w.data = self.decay * self._ema_w.data + (1 - self.decay) * dw

#     # self._embedding.weight.data = self._ema_w / (self._ema_cluster_size.unsqueeze(-1) + self.epsilon)
#     with torch.no_grad():
#         self._embedding.weight.data = self._ema_w / (self._ema_cluster_size.unsqueeze(-1) + self.epsilon)
    
#     use_threshold = 0.01  # This threshold needs to be tuned.
#     underused = (self._ema_cluster_size < use_threshold).nonzero(as_tuple=True)[0]
#     if underused.nelement() > 0:
#         self.reset_codebook(underused)
#     usage = torch.mean(encodings, dim=0)
#     reset_mask = (usage < use_threshold).float()
#     num_reset = torch.sum(reset_mask)
#     reset_ratio = (num_reset / self._num_embeddings).detach()
# else:
#     reset_ratio = torch.zeros(1, device=inputs.device).detach()