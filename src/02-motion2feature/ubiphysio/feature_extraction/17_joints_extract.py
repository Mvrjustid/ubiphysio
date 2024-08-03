import numpy as np
import os
from tqdm import tqdm

def process_npy_data(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 指定需要剔除的节点index
    joints_to_remove = [2, 7, 11, 13, 15, 19, 23]

    for root, _, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                data = np.load(file_path)

                # 检查数据形状是否为 T x 24 x 3
                if data.shape[1:] == (24, 3):
                    # 去除指定的节点
                    data = np.delete(data, joints_to_remove, axis=1)
                    # 保存到新的路径
                    output_path = os.path.join(output_dir, file)
                    np.save(output_path, data)
                else:
                    print(f"Warning: {file_path} doesn't have the expected shape and was skipped.")

if __name__ == '__main__':
    input_directory = '/data4/zhonglx/archive/joints'
    output_directory = '/data4/zhonglx/archive/joints_17'
    process_npy_data(input_directory, output_directory)
