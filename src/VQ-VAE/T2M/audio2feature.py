import os
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def standardize_features(features, mean_file, std_file):
    # 加载均值和标准差
    mean = np.load(mean_file)
    std = np.load(std_file)

    # 检查维度匹配
    if features.shape[1] != mean.shape[0] or features.shape[1] != std.shape[0]:
        raise ValueError("特征维度与均值/标准差维度不匹配。")

    # 对特征进行标准化
    standardized_features = (features - mean) / std

    return standardized_features

def process_audio(audio_path, sample_rate=16000, duration=15, normalize=None):
    # Load the audio file
    waveform, sr = librosa.load(audio_path, sr=sample_rate)

    # If the audio is shorter than the max length, zero-pad it
    if len(waveform) < sr * duration:
        padding = sr * duration - len(waveform)
        waveform = np.pad(waveform, (0, padding))
    # If the audio is longer than the max length, take the first 'duration' seconds only
    else:
        waveform = waveform[:sr * duration]

    # Extract features
    mfcc_feature = librosa.feature.mfcc(waveform, sr=sample_rate, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(waveform, sr=sample_rate, n_chroma=24)
    contrast = librosa.feature.spectral_contrast(waveform, sr=sample_rate, n_bands=6, fmin=50)
    tonnetz = librosa.feature.tonnetz(waveform, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(waveform)
    rmse = librosa.feature.rms(waveform)
    centroid = librosa.feature.spectral_centroid(waveform, sr=sample_rate)

    features = np.concatenate([mfcc_feature, chroma, contrast, tonnetz, zcr, rmse, centroid], axis=0)
    
    if normalize:
        features = standardize_features(features,"./dataset/ubiphysio/audio_mean.npy","./dataset/ubiphysio/audio_std.npy")

    # print(np.shape(features)) D:60 X T:1292

    return  features

def process_and_save(audio_dir, save_dir=None, sample_rate=44100, duration=15, extension='.wav'):
    save_dir = audio_dir if save_dir is None else save_dir
    os.makedirs(save_dir, exist_ok=True)
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(extension)]
    for audio_file in tqdm(audio_files, desc='Processing audio files'):  # add a progress bar here
        audio_path = os.path.join(audio_dir, audio_file)
        features = process_audio(audio_path, sample_rate, duration,normalize=True)
        save_path = os.path.join("./dataset/ubiphysio/audio_new", os.path.splitext(audio_file)[0] + '.npy')
        np.save(save_path, features)  # save features as .npy file

def compute_global_stats(audio_dir, sample_rate=44100, duration=15, extension='.wav'):
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(extension)]
    all_features = []
    for audio_file in tqdm(audio_files, desc='Computing global statistics'):
        audio_path = os.path.join(audio_dir, audio_file)
        features = process_audio(audio_path, sample_rate, duration,normalize=False)
        all_features.append(features)

    # Concatenate all the features
    all_features = np.concatenate(all_features, axis=0)

    # Compute mean and std
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)

    # Save the mean and std
    np.save(os.path.join("./dataset/ubiphysio", 'audio_mean.npy'), mean)
    np.save(os.path.join("./dataset/ubiphysio", 'audio_std.npy'), std)
    
# Compute globals
audio_dir = "./dataset/ubiphysio/audio"
compute_global_stats(audio_dir)

process_and_save("./dataset/ubiphysio/audio")  # specify your audio files directory
