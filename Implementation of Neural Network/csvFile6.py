import pandas as pd

df_e = pd.read_csv('hand_landmarks_H.csv')

# Replace 'E' with 'C' in the first word of each row
df_e.iloc[:, 0] = df_e.iloc[:, 0].str.replace('^A', 'H', regex=True)

# Save the modified dataframe to a new CSV file
output_file_path_e = 'E:\\TurkishSignLanguage\\CoordinatedImages\\Training_Coordinates\\hand_landmarks_H_modified.csv'
df_e.to_csv(output_file_path_e, index=False)

print(output_file_path_e)
