import os
import pickle
import random

all_files = os.listdir('SYNAPSE/img_processed')
n = len(all_files)
splits = []

# for i in range(5):
#     random.shuffle(all_files)
#     test_files = all_files[:int(n * 0.15)]
#     val_files = all_files[int(n * 0.15):int(n * 0.30)]
#     train_files = all_files[int(n * 0.30):]
#     splits.append({'train': train_files,
#                    'val': val_files, 
#                    'test': test_files})

# pickle.dump(splits, open('SYNAPSE/splits.pkl', 'wb'))

split_path = './SYNAPSE/splits.pkl'
with open(split_path, "rb") as f:
    splits = pickle.load(f)
tr_keys = splits[1]['train']
val_keys = splits[1]['val']
test_keys = splits[1]['test']
print(len(tr_keys))
print(len(val_keys))
print(len(test_keys))