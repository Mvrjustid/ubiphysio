import argparse
import numpy as np
import sys
import os
from os.path import join as pjoin


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
# custom features (B, seq, 28)
def mean_variance(data_dir, save_dir, joints_num):
    file_list = os.listdir(data_dir)
    data_list = []

    for file in file_list:
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    # root: 4
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    # ric: (joint_num - 1)*3
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
    # rot: (joint_num - 1)*6
    Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
    # vel: joint_num*3
    Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
    # foot: 4
    Std[4 + (joints_num - 1) * 9 + joints_num * 3: 4 + (joints_num - 1) * 9 + joints_num * 3 + 4] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: 4 + (joints_num - 1) * 9 + joints_num * 3 + 4].mean() / 1.0
    # custom: 28
    Std[4 + (joints_num - 1) * 9 + joints_num * 3 + 4: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3 + 4: ].mean() / 1.0

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 + 28 == Std.shape[-1]

    np.save(pjoin(save_dir, 'Mean_30.npy'), Mean)
    np.save(pjoin(save_dir, 'Std_30.npy'), Std)

    return Mean, Std


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="计算 mean & variance")
    parser.add_argument('--data_dir', type=str, default='/data4/zhonglx/archive/new_joint_vecs', help='特征数据的目录')
    parser.add_argument('--save_dir', type=str, default='/data4/zhonglx/archive/', help='输出目录')
    args = parser.parse_args()

    mean, Std = mean_variance(args.data_dir, args.save_dir, 24)
    print(mean)
    print(Std)