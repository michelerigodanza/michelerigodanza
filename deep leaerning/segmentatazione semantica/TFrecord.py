
import tensorflow as tf
import random

BUFFER_SIZE =4000
BATCH_SIZE = 53 #limite massimo per capacità di memoria su gpu

def importTFRecord():


    # Path to save TFRecord file
    tfrecord_path_satellite = "satellite.tfrecord"
    tfrecord_path_mask = "mask.tfrecord"



    def parse_tfrecord_fn(example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, feature_description)

        # Deserialize the tensor from bytes with the correct dtype
        image_tensor = tf.io.parse_tensor(example['image'], out_type=tf.float32)  # Adjust the dtype accordingly

        return image_tensor


    seed = random.randint(1, 100)
    # Create TFRecordDataset
    train_dataset_mask = (
        tf.data.TFRecordDataset(tfrecord_path_mask).shuffle(BUFFER_SIZE, seed=seed)
        .map(parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(BATCH_SIZE)
    )

    # Create TFRecordDataset
    train_dataset_satellite = (
        tf.data.TFRecordDataset(tfrecord_path_satellite).shuffle(BUFFER_SIZE, seed=seed)
        .map(parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(BATCH_SIZE)
    )


    return train_dataset_satellite, train_dataset_mask


def importTFRecord_val():




    # Path to save TFRecord file
    satellite_val = "images_satelliteB2su2-validation.tfrecord"
    mask_val = "images_maskB2su2_validation.tfrecord"

    def parse_tfrecord_fn(example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, feature_description)

        # Deserialize the tensor from bytes with the correct dtype
        image_tensor = tf.io.parse_tensor(example['image'], out_type=tf.float32)  # Adjust the dtype accordingly

        return image_tensor

    seed = random.randint(1, 100)
    # Create TFRecordDataset
    val_dataset_mask = (
        tf.data.TFRecordDataset(mask_val).shuffle(BUFFER_SIZE, seed=seed)
        .map(parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(BATCH_SIZE)
    )

    # Create TFRecordDataset
    val_dataset_satellite = (
        tf.data.TFRecordDataset(satellite_val).shuffle(BUFFER_SIZE, seed=seed)
        .map(parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(BATCH_SIZE)
    )

    return val_dataset_satellite, val_dataset_mask


