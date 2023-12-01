import os
import pickle
import random

all_files = os.listdir('Kvasir-SEG/images')
n = len(all_files)
splits = []

for i in range(5):
    random.shuffle(all_files)
    test_files = all_files[:int(n * 0.15)]
    val_files = all_files[int(n * 0.15):int(n * 0.30)]
    train_files = all_files[int(n * 0.30):]
    splits.append({'train': train_files,
                   'val': val_files, 
                   'test': test_files})

pickle.dump(splits, open('Kvasir-SEG/splits.pkl', 'wb'))