import pandas as pd
import os
import random
import shutil
import argparse

def read_data(path):
    data_list = []
    top_folders = sorted(os.listdir(path))[:4]
    for folder in top_folders:
        for file in os.listdir(os.path.join(path, folder)):
            data_list.append({'image': os.path.join(path, folder, file), 'label': folder})
    return pd.DataFrame(data_list)

def pick_random(data, path, num_images):
    for folder in data['label'].unique():
        folder_path = os.path.join(path, folder)
        os.makedirs(folder_path, exist_ok=True)
        images = list(data[data['label'] == folder]['image'])
        sample_size = min(num_images, len(images))
        for idx, file in enumerate(random.sample(images, sample_size), start=1):
            dest = os.path.join(folder_path, f"{idx}.png")
            shutil.copy(file, dest)

def create_mixed_data(data, path):
    folder_path = os.path.join(path, 'mixed_data')
    os.makedirs(folder_path, exist_ok=True)
    
    images_picked = []
    for idx, folder in enumerate(data['label'].unique(), start=1):
        available_images = [img for img in data[data['label'] == folder]['image'].tolist() if img not in images_picked]
        if not available_images:
            raise ValueError(f"No available images for label {folder} that haven't been picked already.")
        file = random.choice(available_images)
        images_picked.append(file)
        dest = os.path.join(folder_path, f"{idx}.png")
        shutil.copy(file, dest)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process image data.')
    parser.add_argument('--input_path', required=True, help='Path to the input data directory.')
    parser.add_argument('--output_path', required=True, help='Path to the output data directory.')
    parser.add_argument('--num_images', type=int, default=15, help='Number of images to use from each folder.')

    args = parser.parse_args()
    
    data = read_data(args.input_path)
    pick_random(data, args.output_path, args.num_images)
    create_mixed_data(data, args.output_path)