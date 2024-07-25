import argparse

from tqdm import tqdm
from annotation_handler_hslu import AnnotationHandlerHslu


import glob
from pathlib import Path
import os
import json
import random
from sklearn.model_selection import train_test_split

from delete_non_visible_series import get_start_and_stop_frame


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Path to config file', default='D:\gen2', type=Path)
    parser.add_argument('--version', help='Path to config file', default='half_step', type=str)
    parser.add_argument('--length_requirement', help='Path to config file', default=16, type=int)
    args = parser.parse_args()


    local_paths = "/."


    # args.dataset = Path("D:\gen2")

    paths = [{"back_to_front":1},
            {"front_to_back":0},
            {"left_to_right":2},
            {"right_to_left":3},
            {"standing":4}]

    data = []

    min_samples = float('inf')
    label_to_folders = {}

    for path in tqdm(paths):
        for name, label in path.items():
            folders_with_data= glob.glob(args.dataset.joinpath(name+"_*").as_posix())
            if label not in label_to_folders:
                label_to_folders[label] = []

            folders_to_remove = []

            inner_tqdm = tqdm(enumerate(folders_with_data), total=len(folders_with_data), desc=f"Processing folders for {name}", leave=False)

            for folder_index,folder_name in inner_tqdm:
                annotations_file = Path(folder_name).joinpath("annotation.csv")
                if annotations_file.exists():
                    start,stop,valid = get_start_and_stop_frame(Path(folder_name))
                    if stop-start < args.length_requirement and label != 4:
                        valid =False
                    if not valid:
                        folders_to_remove.append(folder_index)           
                else:
                    folders_to_remove.append(folder_index)

            for index in sorted(folders_to_remove, reverse=True):
                del folders_with_data[index]

            label_to_folders[label].append((name, folders_with_data))
            min_samples = min(min_samples, len(folders_with_data))


    # Step 2: Limit the number of samples taken from each folder to the minimum count
    for label, folders_list in label_to_folders.items():
        for name, folders_with_data in folders_list:
            limited_folders = folders_with_data[:min_samples]
            for folder in limited_folders:
                path_to_inspect = Path(folder)
                start,stop,valid = get_start_and_stop_frame(path_to_inspect)
                if valid:
                    data.append({"id": path_to_inspect.name,
                                "label": label,
                                "start":start,
                                "stop":stop})

    # Shuffle the data to ensure random split
    random.shuffle(data)

    # Separate the data by labels
    label_to_data = {}
    for item in data:
        label = item['label']
        if label not in label_to_data:
            label_to_data[label] = []
        label_to_data[label].append(item)

    # Split data while maintaining the balance of labels
    train_data, val_data, test_data = [], [], []

    for label, items in label_to_data.items():
        train_items, test_items = train_test_split(items, test_size=0.2, random_state=42, stratify=[item['label'] for item in items])
        train_items, val_items = train_test_split(train_items, test_size=0.25, random_state=42, stratify=[item['label'] for item in train_items])  # 0.25 * 0.8 = 0.2
        train_data.extend(train_items)
        val_data.extend(val_items)
        test_data.extend(test_items)

    # Shuffle the splits again to ensure random order
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # Save the data to JSON files
    def save_to_json(data, filename):
        with open(filename, 'w') as f:
            json.dump({"Data": data}, f, indent=4)

    save_to_json(train_data, args.dataset.joinpath('annotations_train_{}.json'.format(args.version)))
    save_to_json(val_data, args.dataset.joinpath('annotations_validation_{}.json'.format(args.version)))
    save_to_json(test_data, args.dataset.joinpath('annotations_test_{}.json'.format(args.version)))


if __name__ == "__main__":
    main()