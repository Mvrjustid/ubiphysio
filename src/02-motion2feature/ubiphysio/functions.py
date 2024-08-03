import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import matplotlib.animation as animation
from natsort import natsorted
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader,TensorDataset,Dataset
from sklearn.metrics import f1_score, confusion_matrix
from torchvision import transforms
from scipy.linalg import fractional_matrix_power
from scipy import stats
from scipy import interpolate

data_dir=os.path.join('..','Dataset','EmoPain/')
data_dir_health_raw=os.path.join('..','Dataset','All_Data_ShortName_health/')
data_dir_patient_raw=os.path.join('..','Dataset','All_Data_ShortName_patient/')

def downsample(data, labels1, labels2, old_fps, new_fps):
    # 计算新的数据点数量
    new_length = int(data.shape[0] * new_fps / old_fps)

    # 创建新的时间轴
    old_time = np.arange(data.shape[0])
    new_time = np.linspace(0, data.shape[0] - 1, new_length)

    # 初始化新的数据和标签
    new_data = np.zeros((new_length, data.shape[1]))
    new_labels1 = np.zeros((new_length, labels1.shape[1]))
    new_labels2 = np.zeros((new_length, labels2.shape[1]))

    # 对每一维的数据和标签进行插值
    for i in range(data.shape[1]):
        interp_func = interpolate.interp1d(old_time, data[:, i])
        new_data[:, i] = interp_func(new_time)
    
    for i in range(labels1.shape[1]):
        interp_func = interpolate.interp1d(old_time, labels1[:, i], kind='nearest')
        new_labels1[:, i] = interp_func(new_time)

    for i in range(labels2.shape[1]):
        interp_func = interpolate.interp1d(old_time, labels2[:, i], kind='nearest')
        new_labels2[:, i] = interp_func(new_time)

    return new_data, new_labels1, new_labels2

def sliding_window_split(data_x, window_length, step):
    num_frames = data_x.shape[0]
    windowed_data = []

    for start_idx in range(0, num_frames, int(step)):
        end_idx = start_idx + window_length
        if end_idx <= num_frames:
            windowed_data.append(data_x[start_idx:end_idx, :])
        else:
            # padding = np.zeros((window_length, data_x.shape[1]))
            # padding[:num_frames - start_idx, :] = data_x[start_idx:, :]
            # windowed_data.append(padding)
            break

    return np.stack(windowed_data)

def z_normalize(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    normalized_data = (data - mean_val) / std_val
    return normalized_data

def reshape_normalize_return(data):
    # input size frames 78
    num_joints=26
    # 分割 x, y, z 坐标值
    x = data[:, :num_joints]
    x = z_normalize(x)
    y = data[:, num_joints:num_joints*2]
    y = z_normalize(y)
    z = data[:, num_joints*2:num_joints*3]
    z = z_normalize(z)
    # 将 x, y, z 坐标值堆叠成形状为 (frames 78) 的张量
    data_reshaped = np.concatenate((x, y, z), axis=-1) # frames 78
    return data_reshaped

def reshape_xyz_full(data):
    # input size N 180 78, do not delete the foot
    num_joints=26
    # 分割 x, y, z 坐标值
    x = data[:, :, :num_joints]
    x = np.expand_dims(x, axis=-1) # N 180 26 1
    y = data[:, :, num_joints:num_joints*2]
    y = np.expand_dims(y, axis=-1) # N 180 26 1
    z = data[:, :, num_joints*2:num_joints*3]
    z = np.expand_dims(z, axis=-1) # N 180 26 1
    # 将 x, y, z 坐标值堆叠成形状为 (N 180 26 3) 的张量
    data_reshaped = np.concatenate((x, y, z), axis=-1) # N 180 26 3
    return data_reshaped

def reshape_xyz_raw(data):
    # input size frames 356
    num_joints=26
    # 分割 x, y, z 坐标值
    x = data[:, :num_joints]
    # x = z_normalize(x)
    x = np.expand_dims(x, axis=2) # frames 26 1
    y = data[:, num_joints:num_joints*2]
    # y = z_normalize(y)
    y = np.expand_dims(y, axis=2) # frames 26 1
    z = data[:, num_joints*2:num_joints*3]
    # z = z_normalize(z)
    z = np.expand_dims(z, axis=2) # frames 26 1
    # 将 x, y, z 坐标值堆叠成形状为 (frames 26 3) 的张量
    data_reshaped = np.concatenate((x, y, z), axis=-1) # frames 26 3

    # 在核心部位增加一个节点以适配human数据库
    node11 = data_reshaped[:, 11, :]
    node12 = data_reshaped[:, 12, :]
    extra_joint_data = 0.5 * (node11 + node12)

    # Insert the interpolated node into the data
    data_reshaped = np.insert(data_reshaped, 12, extra_joint_data, axis=1)

    indices_to_delete = [4, 9, 18, 23, 26]
    data_new = np.delete(data_reshaped, indices_to_delete, axis=1) # frames 22 3
    return data_new

def remove_transition(x,ac,y):
    # x: Tx78, ac: Tx1, y: Tx5
    remove_indices = np.where(ac == 6)[0]
    x_new = np.delete(x, remove_indices, axis=0)
    ac_new = np.delete(ac, remove_indices, axis=0)
    y_new = np.delete(y, remove_indices, axis=0)
    return x_new, ac_new, y_new

def get_id(filename):
    return int(re.findall(r'\d+', filename)[0])

def get_actlabel(data):
    # Input frames x 356
    I = np.arange(data.shape[0])
    
    olg  = 0.1 * (data[I, 110] + data[I, 111] + data[I, 112] + data[I, 113] + data[I, 114] + data[I, 115])
    rf   = 0.01 * (data[I, 117] + data[I, 118])
    sits = 0.001 * (data[I, 119] + data[I, 120] + data[I, 121] + data[I, 126] + data[I, 132])
    stsi = 0.0001 * (data[I, 122] + data[I, 123] + data[I, 124] + data[I, 127] + data[I, 133])
    bend = 0.00001 * (data[I, 128] + data[I, 129] + data[I, 131])
    walk = 0.000001 * (data[I, 130])
    
    null = 0.0000001 * np.ones(I.shape[0])
    null[(olg + rf + sits + stsi + bend + walk) != 0] = 0
    
    activity = olg + rf + sits + stsi + bend + walk + null
    
    activity[activity == 0.1] = 1 # One leg stand
    activity[activity == 0.01] = 2 # Reach forward
    activity[activity == 0.001] = 3 # Sit to stand
    activity[activity == 0.0001] = 4 # Stand to sit
    activity[activity == 0.00001] = 5 # Bend down
    activity[activity == 0.000001] = 6 # Walk
    activity[activity == 0.0000001] = 7 # Transition
    activity[activity < 1] = 7 # frames,
    activity=np.expand_dims(activity,axis=-1)
    return activity-1

def get_blabel(data):
    # Input frames x 356
    I = np.arange(data.shape[0])
    
    Rater1  = data[I, 82]  + data[I, 83]  + data[I, 84]  + data[I, 85]  + data[I, 87]  + data[I, 88]
    Rater2  = data[I, 89]  + data[I, 90]  + data[I, 91]  + data[I, 92]  + data[I, 94]  + data[I, 95]
    Rater3  = data[I, 96]  + data[I, 97]  + data[I, 98]  + data[I, 99]  + data[I, 101] + data[I, 102]
    Rater4  = data[I, 103] + data[I, 104] + data[I, 105] + data[I, 106] + data[I, 108] + data[I, 109]

    Rater1[Rater1>0]=1
    Rater2[Rater2>0]=1
    Rater3[Rater3>0]=1
    Rater4[Rater4>0]=1

    Rater1=np.expand_dims(Rater1,axis=-1)
    Rater2=np.expand_dims(Rater2,axis=-1)
    Rater3=np.expand_dims(Rater3,axis=-1)
    Rater4=np.expand_dims(Rater4,axis=-1)

    return np.concatenate([Rater1,Rater2,Rater3,Rater4],axis=-1)

def get_blabel_fullcategory(data):
    # Input frames x 356
    # Merge the opinion of all raters
    I = np.arange(data.shape[0])
    
    Behavior_1  = data[I, 89] + data[I, 82] + data[I, 96] + data[I, 103]
    Behavior_2  = data[I, 90] + data[I, 83] + data[I, 97] + data[I, 104]
    Rater2_3  = data[I, 91] + data[I, 84] + data[I, 98] + data[I, 105]
    Rater2_4  = data[I, 92] + data[I, 85] + data[I, 99] + data[I, 106]
    Rater2_5  = data[I, 94] + data[I, 87] + data[I, 101] + data[I, 108]

    Behavior_1[Behavior_1>0]=1
    Rater2_2[Rater2_2>0]=1
    Rater2_3[Rater2_3>0]=1
    Rater2_4[Rater2_4>0]=1
    Rater2_5[Rater2_5>0]=1

    Behavior_1=np.expand_dims(Behavior_1,axis=-1)
    Rater2_2=np.expand_dims(Rater2_2,axis=-1)
    Rater2_3=np.expand_dims(Rater2_3,axis=-1)
    Rater2_4=np.expand_dims(Rater2_4,axis=-1)
    Rater2_5=np.expand_dims(Rater2_5,axis=-1)
    
    return np.concatenate([Behavior_1,Rater2_2,Rater2_3,Rater2_4,Rater2_5],axis=-1)

def cleanandsave_health_raw():
    input_folder = data_dir_health_raw 
    output_folder = os.path.join('..', 'Dataset', f'EmoPain-raw/')
    files = os.listdir(input_folder)
    files = natsorted(files, key=lambda x: (x[0], get_id(x)))
    grouped_files = {}

    for filename in files:
        file_id = get_id(filename)
        if file_id not in grouped_files:
            grouped_files[file_id] = []
        grouped_files[file_id].append(filename)

    sequence_number = 1

    for file_id, file_group in grouped_files.items():
        stacked_data_x = []
        stacked_data_ac = []
        stacked_data_y = []
        for filename in file_group:
            file_path = os.path.join(input_folder, filename)
            mat_data = scipy.io.loadmat(file_path)
            data_key = list(mat_data.keys())[-1]
            data = mat_data[data_key]

            data_x = data[:, :78] # This is to get the motion data
            # data_x = reshape_normalize_return(data_x) # Normalize per coordiante and return
            data_ac = get_actlabel(data) # This is to get the activity label
            data_y = get_blabel_fullcategory(data) # This is to get the behavior label in 5 categories
            stacked_data_x.append(data_x)
            stacked_data_ac.append(data_ac)
            stacked_data_y.append(data_y)

        all_data_x = np.concatenate(stacked_data_x,axis=0)
        all_data_ac = np.concatenate(stacked_data_ac,axis=0)
        all_data_y = np.concatenate(stacked_data_y,axis=0)
        # all_data_x, all_data_ac, all_data_y = remove_transition(all_data_x, all_data_ac, all_data_y) #注释掉方便凌逍分辨运动的分段
        all_data_x, all_data_ac, all_data_y = downsample(all_data_x, all_data_ac, all_data_y, 60, 20)
        all_data_x = reshape_xyz_raw(all_data_x) #reshape from 78 to 26 x 3
        # all_data_ac = np.squeeze(all_data_ac,axis=-1) # For activity label data only
        
        group_char = file_group[0][0]
        output_filename_x = f"{group_char}{sequence_number}.mat" 
        output_filepath_x = os.path.join(output_folder, output_filename_x)
        output_filename_ac = f"{group_char}{sequence_number}aclabel.mat" 
        output_filepath_ac = os.path.join(output_folder, output_filename_ac)
        output_filename_y = f"{group_char}{sequence_number}label.mat" 
        output_filepath_y = os.path.join(output_folder, output_filename_y)
        scipy.io.savemat(output_filepath_x, {'data': all_data_x})
        scipy.io.savemat(output_filepath_ac, {'data': all_data_ac})
        scipy.io.savemat(output_filepath_y, {'data': all_data_y})

        sequence_number += 1

def cleanandsave_patient_raw():
    input_folder = data_dir_patient_raw
    output_folder = os.path.join('..', 'Dataset', f'EmoPain-raw/')
    files = os.listdir(input_folder)
    files = natsorted(files, key=lambda x: (x[0], get_id(x)))
    grouped_files = {}

    for filename in files:
        file_id = get_id(filename)
        if file_id not in grouped_files:
            grouped_files[file_id] = []
        grouped_files[file_id].append(filename)

    sequence_number = 13 

    for file_id, file_group in grouped_files.items():
        stacked_data_x = []
        stacked_data_ac = []
        stacked_data_y = []
        for filename in file_group:
            file_path = os.path.join(input_folder, filename)
            mat_data = scipy.io.loadmat(file_path)
            data_key = list(mat_data.keys())[-1]
            data = mat_data[data_key]

            data_x = data[:, :78] # This is to get the motion data
            # data_x = reshape_normalize_return(data_x) # Normalize per coordiante and return, disable only for M2T
            data_ac = get_actlabel(data) # This is to get the activity label
            # data_y = get_blabel(data) # This is to get the behavior label
            data_y = get_blabel_fullcategory(data) # This is to get the behavior label in 5 categories, only for M2T
            stacked_data_x.append(data_x)
            stacked_data_ac.append(data_ac)
            stacked_data_y.append(data_y)

        all_data_x = np.concatenate(stacked_data_x,axis=0)
        all_data_ac = np.concatenate(stacked_data_ac,axis=0)
        all_data_y = np.concatenate(stacked_data_y,axis=0)
        # all_data_x, all_data_ac, all_data_y = remove_transition(all_data_x, all_data_ac, all_data_y) #注释掉方便凌逍分辨运动的分段
        all_data_x, all_data_ac, all_data_y = downsample(all_data_x, all_data_ac, all_data_y, 60, 20)
        all_data_x = reshape_xyz_raw(all_data_x) #reshape from 78 to 26 x 3

        group_char = file_group[0][0]
        output_filename_x = f"{group_char}{sequence_number}.mat" 
        output_filepath_x = os.path.join(output_folder, output_filename_x)
        output_filename_ac = f"{group_char}{sequence_number}aclabel.mat" 
        output_filepath_ac = os.path.join(output_folder, output_filename_ac)
        output_filename_y = f"{group_char}{sequence_number}label.mat" 
        output_filepath_y = os.path.join(output_folder, output_filename_y)
        scipy.io.savemat(output_filepath_x, {'data': all_data_x})
        scipy.io.savemat(output_filepath_ac, {'data': all_data_ac})
        scipy.io.savemat(output_filepath_y, {'data': all_data_y})

        sequence_number += 1

def cleanandsave_health_window(length=None):
    input_folder = data_dir_health_raw 
    output_folder = os.path.join('..', 'Dataset', f'EmoPain-{length}/')
    files = os.listdir(input_folder)
    files = natsorted(files, key=lambda x: (x[0], get_id(x)))
    grouped_files = {}

    for filename in files:
        file_id = get_id(filename)
        if file_id not in grouped_files:
            grouped_files[file_id] = []
        grouped_files[file_id].append(filename)

    sequence_number = 1

    for file_id, file_group in grouped_files.items():
        stacked_data_x = []
        stacked_data_ac = []
        stacked_data_y = []
        for filename in file_group:
            file_path = os.path.join(input_folder, filename)
            mat_data = scipy.io.loadmat(file_path)
            data_key = list(mat_data.keys())[-1]
            data = mat_data[data_key]

            data_x = data[:, :78] # This is to get the motion data
            # data_x = reshape_normalize_return(data_x) # Normalize per coordiante and return
            data_ac = get_actlabel(data) # This is to get the activity label
            data_y = get_blabel_fullcategory(data) # This is to get the behavior label in 5 categories
            stacked_data_x.append(data_x)
            stacked_data_ac.append(data_ac)
            stacked_data_y.append(data_y)

        all_data_x = np.concatenate(stacked_data_x,axis=0)
        all_data_ac = np.concatenate(stacked_data_ac,axis=0)
        all_data_y = np.concatenate(stacked_data_y,axis=0)
        all_data_x, all_data_ac, all_data_y = remove_transition(all_data_x, all_data_ac, all_data_y)
        windowed_data_x = sliding_window_split(all_data_x, length, length)
        windowed_data_x = reshape_xyz_full(windowed_data_x) #reshape from 78 to 26 x 3
        windowed_data_ac = sliding_window_split(all_data_ac, length, length)
        windowed_data_ac = np.squeeze(windowed_data_ac,axis=-1) # For activity label data only
        windowed_data_y = sliding_window_split(all_data_y, length, length)
        
        group_char = file_group[0][0]
        output_filename_x = f"{group_char}{sequence_number}.mat" 
        output_filepath_x = os.path.join(output_folder, output_filename_x)
        output_filename_ac = f"{group_char}{sequence_number}aclabel.mat" 
        output_filepath_ac = os.path.join(output_folder, output_filename_ac)
        output_filename_y = f"{group_char}{sequence_number}label.mat" 
        output_filepath_y = os.path.join(output_folder, output_filename_y)
        scipy.io.savemat(output_filepath_x, {'data': windowed_data_x})
        scipy.io.savemat(output_filepath_ac, {'data': windowed_data_ac})
        scipy.io.savemat(output_filepath_y, {'data': windowed_data_y})

        sequence_number += 1

def cleanandsave_patient_window(length=None):
    input_folder = data_dir_patient_raw
    output_folder = os.path.join('..', 'Dataset', f'EmoPain-{length}/')
    files = os.listdir(input_folder)
    files = natsorted(files, key=lambda x: (x[0], get_id(x)))
    grouped_files = {}

    for filename in files:
        file_id = get_id(filename)
        if file_id not in grouped_files:
            grouped_files[file_id] = []
        grouped_files[file_id].append(filename)

    sequence_number = 13 

    for file_id, file_group in grouped_files.items():
        stacked_data_x = []
        stacked_data_ac = []
        stacked_data_y = []
        for filename in file_group:
            file_path = os.path.join(input_folder, filename)
            mat_data = scipy.io.loadmat(file_path)
            data_key = list(mat_data.keys())[-1]
            data = mat_data[data_key]

            data_x = data[:, :78] # This is to get the motion data
            # data_x = reshape_normalize_return(data_x) # Normalize per coordiante and return, disable only for M2T
            data_ac = get_actlabel(data) # This is to get the activity label
            # data_y = get_blabel(data) # This is to get the behavior label
            data_y = get_blabel_fullcategory(data) # This is to get the behavior label in 5 categories, only for M2T
            stacked_data_x.append(data_x)
            stacked_data_ac.append(data_ac)
            stacked_data_y.append(data_y)

        all_data_x = np.concatenate(stacked_data_x,axis=0)
        all_data_ac = np.concatenate(stacked_data_ac,axis=0)
        all_data_y = np.concatenate(stacked_data_y,axis=0)
        all_data_x, all_data_ac, all_data_y = remove_transition(all_data_x, all_data_ac, all_data_y)
        windowed_data_x = sliding_window_split(all_data_x, length, length)
        windowed_data_x = reshape_xyz_full(windowed_data_x) #reshape from 78 to 26 x 3, only for M2T
        windowed_data_ac = sliding_window_split(all_data_ac, length, length)
        windowed_data_ac = np.squeeze(windowed_data_ac,axis=-1) # For activity label data only
        windowed_data_y = sliding_window_split(all_data_y, length, length)

        group_char = file_group[0][0]
        output_filename_x = f"{group_char}{sequence_number}.mat" 
        output_filepath_x = os.path.join(output_folder, output_filename_x)
        output_filename_ac = f"{group_char}{sequence_number}aclabel.mat" 
        output_filepath_ac = os.path.join(output_folder, output_filename_ac)
        output_filename_y = f"{group_char}{sequence_number}label.mat" 
        output_filepath_y = os.path.join(output_folder, output_filename_y)
        scipy.io.savemat(output_filepath_x, {'data': windowed_data_x})
        scipy.io.savemat(output_filepath_ac, {'data': windowed_data_ac})
        scipy.io.savemat(output_filepath_y, {'data': windowed_data_y})

        sequence_number += 1

cleanandsave_health_raw()
cleanandsave_patient_raw()