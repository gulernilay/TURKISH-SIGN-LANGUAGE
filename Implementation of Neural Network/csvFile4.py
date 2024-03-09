import os
import shutil

def group_folders(source_path):
    all_folders = [folder for folder in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, folder))]
    base_folders = {folder.rsplit('_', 1)[0] for folder in all_folders if '_' in folder}

    for base_folder in base_folders:
        sub_folders = [folder for folder in all_folders if folder.startswith(base_folder) and folder != base_folder]
        base_folder_path = os.path.join(source_path, base_folder)

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

            os.rmdir(sub_folder_path)

source_folder_path = 'E:\\türkişaretdili\\Bosphorus\\videos3'
group_folders(source_folder_path)
