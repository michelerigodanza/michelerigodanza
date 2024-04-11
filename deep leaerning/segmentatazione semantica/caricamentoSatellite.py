import os
import numpy as np
import tensorflow as tf
import tifffile
from matplotlib import pyplot as plt
from skimage.transform import resize

IMAGE_SIZE = 128
CANALI = 3
print(tf.__version__)

satellite_dir = "satellite"


def load_and_preprocess_geotiff_images(dir):
    image_tensors = []
    n = 0
    for file in os.listdir(dir):
        if file.endswith(".tif"):
            percorso_file = os.path.join(dir, file)

            # Carica l'immagine TIFF con tifffile
            image = tifffile.imread(percorso_file).astype('float32')

            #image = resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            image = image / 255

            image_tensors.append(image)

            n = n + 1
            print(f"[{n}]")

    return np.array(image_tensors)#.astype('float32')

# Load and preprocess GeoTIFF images_satellite
train_images_satellite = load_and_preprocess_geotiff_images(satellite_dir)



# Save your dataset to a .npy file
#np.save('images_Satellite.npy', train_images_satellite)

# Assuming train_images is a numpy array of tensors
# train_images_satellite = np.load("images_satellite.npy")

# Path to save TFRecord file
tfrecord_path_satellite = "images_satelliteD.tfrecord"

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
with tf.io.TFRecordWriter(tfrecord_path_satellite) as writer:
    for image_tensor in train_images_satellite:
        tf_example = create_example(image_tensor)
        writer.write(tf_example.SerializeToString())



