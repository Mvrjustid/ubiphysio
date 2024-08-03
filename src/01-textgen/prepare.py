import fnmatch
import os
import glob
from tqdm import tqdm
import json

# Mapping from index to action names
idx_to_name = {
    0: 'transition',
    1: 'tie-shoes', 
    2: 'sweep-floor',
    3: 'stand-to-sit-sofa',
    4: 'sit-to-stand-sofa',
    5: 'stand-to-sit-chair',
    6: 'sit-to-stand-chair',
    7: 'carry-suitcase',
    8: 'carry-pillow',
    9: 'body-side-bend',
    10: 'shoulder-wrap',
    11: 'chest-fly',
    12: 'kneeling-hand-forward',
    13: 'kneeling-leg-backward',
    14: 'left-leg-lift',
    15: 'right-leg-lift',
    16: 'squat',
    17: 'stretch-right',
    18: 'stretch-left',
    19: 'bend-touch-toe',
    20: 'kneeling-hand-leg',
    21: 'heel-walking',
    22: 'heel-to-toe-walking',
    23: 'stand-to-lay',
    24: 'lay-to-stand',
    25: 'dead-bug-stretch',
    26: 'casual-walking'
}

# Reverse mapping from action names to index
name_to_idx = {}
for key, value in idx_to_name.items():
    name_to_idx[value] = key

# Load text dataset from texts.json
with open('/path/to/texts.json', 'r', encoding='utf-8') as f:
    text_dataset = json.load(f)

# Get list of folders matching the pattern
folders = glob.glob('expert_descriptions/[CP]*')

# Process each folder
for folder in tqdm(folders):
    for file in os.listdir(folder):
        if file.endswith('.json'):
            filepath = os.path.join(folder, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            pid = folder[20:]
            cnt = 100000 if 'secondpart' in file else 0
            
            set_2, set_11 = False, False
            for action_label, patterns in data:
                action_idx = name_to_idx.get(action_label)
                descriptions = text_dataset[action_label][str(patterns)]

                if action_idx == 2:
                    if set_2:
                        count = 0
                    else:
                        set_2 = True
                        matching_files = [filename for filename in os.listdir('/path/to/joints') if fnmatch.fnmatch(filename, f'{pid}-*-{action_idx}.npy')]
                        count = len(matching_files)
                elif action_idx == 11:
                    if set_11:
                        count = 0
                    else:
                        set_11 = True
                        matching_files = [filename for filename in os.listdir('/path/to/joints') if fnmatch.fnmatch(filename, f'{pid}-*-{action_idx}.npy')]
                        count = len(matching_files)
                else:
                    count = 1
                
                for i in range(count):
                    target_file = f"/path/to/raw_patterns/{pid}-{cnt:06d}-{action_idx}.txt"
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(descriptions))
                    cnt += 1
