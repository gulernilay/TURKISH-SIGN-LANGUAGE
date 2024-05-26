#Tensorflow objectdetecton API kullanıldı.

import tensorflow as tf
import pandas as pd
import os
from object_detection.utils import dataset_util

# CSV dosyasını yükleme fonksiyonu
def load_csv(csv_path):
    return pd.read_csv(csv_path)

# TF örneği oluşturma fonksiyonu
def create_tf_example(group, path):
    # TODO: Grup verilerini kullanarak tf.train.Example oluştur
    pass

# Belirtilen klasördeki tüm CSV dosyaları için TFRecord oluşturma
def create_tf_records(csv_dir, image_dir, output_dir):
    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(csv_dir, csv_file)
            output_path = os.path.join(output_dir, os.path.splitext(csv_file)[0] + '.tfrecord')
            create_tf_record(csv_path, image_dir, output_path)

# CSV'den TFRecord oluşturma fonksiyonu
def create_tf_record(csv_input, image_dir, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    path = os.path.join(image_dir)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')

    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()

# CSV dosyalarının bulunduğu klasör
csv_dir = r'E:\TurkishSignLanguage\CoordinatedImages\Coordinates'
# Görüntülerin bulunduğu klasör
image_dir = r'E:\TurkishSignLanguage\CoordinatedImages\Images'
# Çıktı TFRecord dosyalarının kaydedileceği klasör
output_dir = r'E:\TurkishSignLanguage\CoordinatedImages\TFRecords'

# TFRecord dosyalarını oluştur
create_tf_records(csv_dir, image_dir, output_dir)
