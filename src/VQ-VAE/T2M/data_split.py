import os
from sklearn.model_selection import train_test_split

def split_dataset(input_dir):
    subjects = {}
    # 遍历文件夹
    for filename in os.listdir(input_dir):
        # 提取受试者标识
        subject_id = filename.split('-')[0]

        # 将文件按受试者标识进行分组
        if subject_id not in subjects:
            subjects[subject_id] = []
        subjects[subject_id].append(filename)
    
    # For 0.80-0.15-0.05, the two values are 0.2 and 0.25
    # For 0.70-0.15-0.15, thw two values are 0.3 and 0.5 
    subject_ids = list(subjects.keys())
    train_subject_ids, test_val_subject_ids = train_test_split(subject_ids, test_size=0.2, stratify=[id[0] for id in subject_ids], random_state=24)
    val_subject_ids, test_subject_ids = train_test_split(test_val_subject_ids, test_size=0.25, stratify=[id[0] for id in test_val_subject_ids], random_state=24) 

    with open('./dataset/ubiphysio/all.txt', 'w') as f:
        for sid in subject_ids:
            for filename in subjects[sid]:
                filename_without_ext, _ = os.path.splitext(filename)  # remove the file extension
                f.write(filename_without_ext+'\n')

    with open('./dataset/ubiphysio/train.txt', 'w') as f:
        for sid in train_subject_ids:
            for filename in subjects[sid]:
                filename_without_ext, _ = os.path.splitext(filename)  # remove the file extension
                f.write(filename_without_ext+'\n')

    with open('./dataset/ubiphysio/val.txt', 'w') as f:
        for sid in val_subject_ids:
            for filename in subjects[sid]:
                filename_without_ext, _ = os.path.splitext(filename)  # remove the file extension
                f.write(filename_without_ext+'\n')

    with open('./dataset/ubiphysio/test.txt', 'w') as f:
        for sid in test_subject_ids:
            for filename in subjects[sid]:
                filename_without_ext, _ = os.path.splitext(filename)  # remove the file extension
                f.write(filename_without_ext+'\n')

split_dataset('./dataset/ubiphysio/new_joint_vecs/')