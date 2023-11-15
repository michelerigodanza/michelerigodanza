# Generatore di immagini partendo da un base geotiff
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from keras import layers
import time
from IPython import display
import rasterio
import arrow

print(tf.__version__)

# Definisci la cartella in cui si trovano i file GeoTIFF
if False:
# Replace with the path to your GeoTIFF dataset
    geo_tiff_dir = "\aree\img"

    def load_and_preprocess_geotiff_images(directory):
        image_tensors = []
        n=0
        for file in os.listdir(directory):
            if file.endswith(".tif"):
                with rasterio.open(os.path.join(directory, file)) as dataset:
                    image = dataset.read(1)  # Read the first band (adjust as needed)
                    # You may need to resize or normalize the image depending on your dataset
                    # Example: image = cv2.resize(image, (28, 28))
                    image = cv2.resize(image, (128, 128))
                    #print( image )
                    image = (image - 127.5) / 127.5  # Normalize the images to [-1, 1]
                    # print(f"normalize {image}")
                    image_tensors.append(image)
                    n=n+1;

                    print(f"[{n}/770 [70.720] 29404")
        return np.array(image_tensors)

    # Load and preprocess your GeoTIFF images
    train_images = load_and_preprocess_geotiff_images(geo_tiff_dir)




    print(train_images.shape)

    train_images = train_images.reshape(train_images.shape[0], 128, 128, 1).astype('float32')




    # Save your dataset to a .npy file
    
    np.save('images.npy', train_images)



    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    # Save your dataset to a .npy file


#########################################################################################################################

# Definisci il ciclo di allenamento
EPOCHS = 80
noise_dim = 100
num_examples_to_generate = 12

BUFFER_SIZE = 29000
BATCH_SIZE = 50

# Assuming train_images is a numpy array of tensors
train_images = np.load("images.npy")

# Path to save TFRecord file
tfrecord_path = "images.tfrecord"

# Create a function to convert tensors to tf.train.Example
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()]))

def create_example(image_tensor):
    feature = {
        'image': _bytes_feature(image_tensor),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Convert tensors to TFRecord format
with tf.io.TFRecordWriter(tfrecord_path) as writer:
    for image_tensor in train_images:
        tf_example = create_example(image_tensor)
        writer.write(tf_example.SerializeToString())

# Read TFRecord dataset
def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image_tensor = tf.io.parse_tensor(example['image'], out_type=tf.float16)
    return image_tensor

# Create TFRecordDataset
train_dataset = (
    tf.data.TFRecordDataset(tfrecord_path)
    .shuffle(BUFFER_SIZE)
    .map(parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(BATCH_SIZE)
)
for batch in train_dataset.take(1):  # take 1 batch for demonstration
    # Assuming images are in the first element of the batch
    images = batch

#plt.imshow(images[1, :, :, 0], cmap='gray')
#plt.show()




# Generator versione shape 128x128
def make_generator_model():
    #
    model = tf.keras.Sequential()

    model.add(layers.Dense(128 * 128 * 64, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((128, 128, 64)))
    assert model.output_shape == (None, 128, 128, 64)  # Note: None is the batch size

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 4, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 2, 2, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 1, 1, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())


#######################


    model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 2, 2, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 4, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())


    model.add(layers.Conv2DTranspose(1, (4, 4), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    assert model.output_shape == (None, 128, 128, 1)


    return model
# Usa il generatore (non ancora addestrato) per creare un'immagine.
with tf.device('/device:GPU:0'):
    generator = make_generator_model()

print(tf.config.experimental.list_physical_devices('GPU'))


noise = tf.random.normal([1, noise_dim])
generated_image = generator(noise, training=False)



# Il Discriminatore
# il discriminatore è un classificatore di immagini basato sulla CNN

def make_discriminator_model():
    model = tf.keras.Sequential()
    #print("discriminetore1")
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                            input_shape=[128, 128, 1]))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (4, 4), strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, (4, 4), strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    #print("discriminetore2")
    return model


# sa il discriminatore (non ancora addestrato) per classificare le immagini generate come reali o false.
# Il modello verrà addestrato per produrre valori positivi per immagini reali e valori negativi per immagini false.
with tf.device('/device:GPU:0'):
    discriminator = make_discriminator_model()

decision = discriminator(generated_image)
#print (decision)

# Definire la perdita e gli ottimizzatori

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Perdita discriminante

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# Perdita del generatore
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
"""
#Il discriminatore e gli ottimizzatori del generatore sono diversi poiché addestrerai due reti separatamente.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
"""


# Schedulazione del tasso di apprendimento
initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,  # Sostituisci con il numero di passi di addestramento desiderato
    decay_rate=0.95,  # Fattore di decadimento del tasso di apprendimento
    staircase=True)

# Ottimizzatori con schedulazione del tasso di apprendimento
#generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
#discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

# Lists to store the generator and discriminator losses during training
gen_loss_history = []
disc_loss_history = []



# Salva i checkpoint
checkpoint_dir = 'training_checkpoints_riduzione_apprendimento'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)



# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])



    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss


def train(dataset, epochs):
    total_time = 0
    for epoch in range(epochs):
        start = time.time()


        epoch_gen_loss_avg = tf.keras.metrics.Mean()
        epoch_disc_loss_avg = tf.keras.metrics.Mean()

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            epoch_gen_loss_avg.update_state(gen_loss)
            epoch_disc_loss_avg.update_state(disc_loss)
    
        gen_loss_history.append(epoch_gen_loss_avg.result())
        disc_loss_history.append(epoch_disc_loss_avg.result())

        # Save the model every 15 epochs

        if (epoch + 1) % 1 == 0:
             # Produce images for the GIF as you go
            display.clear_output(wait=True)
            generate_and_save_images(generator, epoch + 1, seed)
            checkpoint.save(file_prefix=checkpoint_prefix)


        if True:
            #Plot generator and discriminator losses
            plt.figure(figsize=(10, 5))
            plt.plot(gen_loss_history, label='Generator Loss')
            plt.plot(disc_loss_history, label='Discriminator Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('Img2/Loss/NEW_image_at_epoch_{:04d}.png'.format(epoch+1))
            #plt.show()
            plt.close()

            # Calcolo temopo
            total_time = (total_time + (time.time() - start))
            orario = arrow.now().replace(hour=0, minute=0).shift(minutes=(total_time / 60))

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start), " - total time: ",
                  orario.format('HH:mm'))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)
  plt.figure(figsize=(10, 10))
  plt.imshow(predictions[1, :, :, 0] * 127.5 + 127.5, cmap='gray')
  """
  for i in range(predictions.shape[0]):
      plt.subplot(6, 6, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
  """

  plt.savefig('Img2/NEW_image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()
  plt.close()

#Ripristina l'ultimo checkpoint.

if True:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))




#per generare immagine singola
#generate_and_save_images(generator, 1, seed)

# Allena il modello

train(train_dataset, EPOCHS)

print(" Fine training ")

#Ripristina l'ultimo checkpoint.
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
