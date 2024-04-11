import os
import tifffile
import numpy as np
import tensorflow as tf
from skimage.transform import resize
print(tf.__version__)
# Replace with the path to your GeoTIFF dataset

IMAGE_SIZE = 256
CANALI = 1

import matplotlib.pyplot as plt


mask_dir = "mask"
def load_and_preprocess_geotiff_images(dir):
    image_tensors = []
    n = 0

    for file in os.listdir(dir):
        if file.endswith(".tif"):
            percorso_file = os.path.join(dir, file)

            #plt.figure(figsize=(12, 4))

            # Carica l'immagine TIFF con tifffile
            image = tifffile.imread(percorso_file)

            image = np.round(image)

            image = image.astype('float32')
            print(image)

            image_tensors.append(image)
            n = n + 1
            print(f"[{n}]")
    return np.array(image_tensors)

# Load and preprocess GeoTIFF images_satellite
train_images_mask = load_and_preprocess_geotiff_images(mask_dir)

train_images_mask = train_images_mask.reshape(train_images_mask.shape[0], IMAGE_SIZE, IMAGE_SIZE, CANALI)



# Path to save TFRecord file
tfrecord_path_mask = "images_mask.tfrecord"


# Create a function to convert tensors to tf.train.Example
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )

def create_example(image_tensor):
    # Assuming the original type is uint8
    feature = {
        'image': _bytes_feature(image_tensor),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# Convert tensors to TFRecord format
with tf.io.TFRecordWriter(tfrecord_path_mask) as writer:
    for image_tensor in train_images_mask:
        tf_example = create_example(image_tensor)
        writer.write(tf_example.SerializeToString())
