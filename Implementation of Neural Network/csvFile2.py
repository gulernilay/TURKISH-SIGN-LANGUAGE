import os
import shutil

def group_folders(source_path):
    folders = [f for f in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, f))]
    grouped_folders = {}

    for folder in folders:
        base_name = folder.split('_')[0]
        if base_name in grouped_folders:
            grouped_folders[base_name].append(folder)
        else:
            grouped_folders[base_name] = [folder]

    for base_name, folder_group in grouped_folders.items():
        if len(folder_group) > 1:
            target_folder = os.path.join(source_path, base_name + 'Big')
            os.makedirs(target_folder, exist_ok=True)
            for folder in folder_group:
                shutil.move(os.path.join(source_path, folder), os.path.join(target_folder, folder))

source_folder_path = r"E:\türkişaretdili\Bosphorus\videos3"
group_folders(source_folder_path)