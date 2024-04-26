import os
from PIL import Image
from tqdm import tqdm

data_path = '../data/processed_data'
train_data_path = '../data/training_data'

os.makedirs(train_data_path, exist_ok=True)


min_img_size = (200, 200)


folders = os.listdir(data_path)
for folder in tqdm(folders):
    folder_path = os.path.join(data_path, folder)
    os.makedirs(os.path.join(train_data_path, folder), exist_ok=True)
    for img_path in os.listdir(f'{folder_path}/no_bg'):
        with Image.open(os.path.join(f'{folder_path}/no_bg', img_path)) as img:
            img = img.convert('RGB')  # Ensure image is in RGB format

        if img.size[0] > min_img_size[0] and img.size[1] > min_img_size[1]:
            os.system(f'cp {os.path.join(f"{folder_path}/no_bg", img_path)} {os.path.join(train_data_path, folder, img_path)}')

print('done')