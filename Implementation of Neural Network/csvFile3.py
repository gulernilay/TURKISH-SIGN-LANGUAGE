import os
import shutil

def move_images_to_parent_big_folder(parent_directory):
    # Belirtilen dizinde dolaş
    for root, dirs, files in os.walk(parent_directory):
        # Her bir dosya için kontrol et
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Mevcut dosya yolu
                current_file_path = os.path.join(root, file)
                # 'Big' ile biten klasörü bul
                if 'Big' in root:
                    # 'Big' ile biten klasör yolu
                    big_folder_path = [path for path in root.split(os.sep) if 'Big' in path][-1]
                    # 'Big' klasörün tam yolunu elde et
                    big_folder_full_path = os.path.join(parent_directory, big_folder_path)
                    # Resmi 'Big' klasörüne taşı
                    shutil.move(current_file_path, big_folder_full_path)
                else:
                    print(f"'Big' klasörü içinde olmayan bir dosya: {file}")

# Kullanımı: move_images_to_parent_big_folder fonksiyonunu, üst klasör yolu ile çağırın
source_path = r"E:\türkişaretdili\Bosphorus\videos3"  # Bu kısmı kendi dosya yolunuz ile değiştirin.
move_images_to_parent_big_folder(source_path)
