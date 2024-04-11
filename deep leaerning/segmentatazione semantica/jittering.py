import tensorflow as tf
import TFrecord as rc
IMG_WIDTH = 256
IMG_HEIGHT = 256

import matplotlib.pyplot as plt
def random_crop(satellite_image, mask_image):
  sat = tf.image.random_crop(satellite_image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
  mask = tf.image.random_crop(mask_image, size=[IMG_HEIGHT, IMG_WIDTH, 1])

  return sat, mask

def resize(satellite_image, mask_image, height, width):
  satellite_image = tf.image.resize(satellite_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  mask_image = tf.image.resize(mask_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return satellite_image, mask_image



#@tf.function()
def random_jitter(satellite_image, mask_image):
  # Resizing to 286x286
  satellite_image, mask_image = resize(satellite_image, mask_image, 286, 286)

  # Random cropping back to 256x256
  satellite_image, mask_image = random_crop(satellite_image, mask_image)


  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    satellite_image = tf.image.flip_left_right(satellite_image)
    mask_image = tf.image.flip_left_right(mask_image)
  else:
      satellite_image = tf.image.flip_up_down(satellite_image)
      mask_image = tf.image.flip_up_down(mask_image)


  return satellite_image, mask_image


tfrecord_path_jitter_satellite = 'images_jitter_satellite.tfrecord'
tfrecord_path_jitter_mask = 'images_jitter_mask.tfrecord'
train_dataset_satellite, train_dataset_mask = rc.importTFRecord()


writer_s = tf.io.TFRecordWriter(tfrecord_path_jitter_satellite)
writer_m = tf.io.TFRecordWriter(tfrecord_path_jitter_mask)
def serialize_tensor(tensor):
    return tf.io.serialize_tensor(tensor)
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


for satellite, mask in zip(train_dataset_satellite, train_dataset_mask):
    for s, m in zip(satellite, mask):
        s, m = random_jitter(s, m)
        serialized_s = serialize_tensor(s).numpy()  # Serializzazione di s in una stringa di byte
        serialized_m = serialize_tensor(m).numpy()  # Serializzazione di m in una stringa di byte

        # Creazione di un esempio TFRecord con i tensori serializzati
        example_s = tf.train.Example(features=tf.train.Features(feature={'image': _bytes_feature(serialized_s)}))
        example_m = tf.train.Example(features=tf.train.Features(feature={'image': _bytes_feature(serialized_m)}))

        # Scrittura dell'esempio TFRecord nei rispettivi file
        writer_s.write(example_s.SerializeToString())
        writer_m.write(example_m.SerializeToString())

# Chiusura dei writer
writer_s.close()
writer_m.close()