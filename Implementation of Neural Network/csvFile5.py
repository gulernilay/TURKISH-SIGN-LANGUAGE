import os
import shutil
import re

def group_folders(source_path):
    all_folders = [folder for folder in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, folder))]
    base_folders = set()

    for folder in all_folders:
        match = re.match(r"(.+?)_\d+$", folder)
        if match:
            base_folders.add(match.group(1))
        else:
            base_folders.add(folder)

    for base_folder in base_folders:
        sub_folders = [folder for folder in all_folders if folder.startswith(base_folder) and (folder == base_folder or re.match(rf"^{base_folder}_\d+$", folder))]
        base_folder_path = os.path.join(source_path, base_folder)

        # Ensure the base folder exists
        if not os.path.exists(base_folder_path):
            os.makedirs(base_folder_path)

        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(source_path, sub_folder)
            for file_name in os.listdir(sub_folder_path):
                source_file = os.path.join(sub_folder_path, file_name)
                target_file = os.path.join(base_folder_path, file_name)
                file_base, file_extension = os.path.splitext(file_name)

                counter = 2
                while os.path.exists(target_file):
                    target_file = os.path.join(base_folder_path, f"{file_base}_{counter}{file_extension}")
                    counter += 1

                shutil.move(source_file, target_file)

            if sub_folder != base_folder:
                os.rmdir(sub_folder_path)

source_folder_path = 'E:\\türkişaretdili\\Bosphorus\\videos'  # Replace with your actual source folder path
group_folders(source_folder_path)