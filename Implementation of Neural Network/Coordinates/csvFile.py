import pandas as pd
import os

# Read the CSV file
file_path = r"E:\TÜRK İŞARET DİLİ\Bosphorus\BosphorusSign22k_classes.csv"
df = pd.read_csv(file_path, header=None, sep=';')

# Create a dictionary mapping codes to words
code_to_word = {}
for index, row in df.iterrows():
    parts = row[0].split(',')
    print(parts)
    print("Length")
    print(len(parts))
    if len(parts) >= 3:
        print("girdi")
        code = parts[1].strip()
        print(code)
        word = parts[2].strip()
        print(word)
        code_to_word[code] = word
        print(code_to_word)
        print(f"Mapping: {code} -> {word}")

# Directory containing the folders to be renamed
videos_dir = r"E:\TÜRK İŞARET DİLİ\Bosphorus\videos"

# Iterate through the directories in the folder
for folder in os.listdir(videos_dir):
    folder_path = os.path.join(videos_dir, folder)
    if os.path.isdir(folder_path) and folder in code_to_word:
        new_folder_name = code_to_word[folder]
        new_folder_path = os.path.join(videos_dir, new_folder_name)
        os.rename(folder_path, new_folder_path)
        print(f"Renamed '{folder}' to '{new_folder_name}'")
