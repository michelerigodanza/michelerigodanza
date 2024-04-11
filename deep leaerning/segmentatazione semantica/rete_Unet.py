
import TFrecord as rc
import tensorflow as tf
import time
from IPython import display
import arrow
import os
import stampa as stampa
#from validation import validate

IMAGE_SIZE = 256




# Definizione del downsample

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

# Definizione del upsample
def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

# Definisco la strutta del mio modello Unet

def Unet():
    CANALI = 3

    inputs = tf.keras.layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, CANALI])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    OUTPUT_CHANNELS = 3
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='softmax')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])


    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

model_optimizer = tf.keras.optimizers.Adam(learning_rate=4e-4, beta_1=0.5)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

def model_loss(model_output, real_mask):
    loss = loss_object(real_mask, model_output)
    return loss


modello_Unet = Unet()
modello_Unet.summary()


# Salva i checkpoint
checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model_optimizer=model_optimizer,
                                 modello_Unet=modello_Unet)



@tf.function
def train_step(satellite, mask):
    with tf.GradientTape() as tape:

        model_output = modello_Unet(satellite, training=True)
        loss = model_loss(model_output, mask)



    gradients= tape.gradient(loss, modello_Unet.trainable_variables)
    model_optimizer.apply_gradients(zip(gradients, modello_Unet.trainable_variables))

    return loss



def train(train_dataset_mask, train_dataset_satellite, epochs):
    total_time = 0
    train_losses = []
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        counter = 0

        for satellite, mask in zip(train_dataset_satellite, train_dataset_mask):

            loss = train_step(satellite, mask)
            total_loss += loss
            counter += 1

        average_loss = total_loss / counter

        train_losses.append(average_loss)



        if (epoch+1) % 5 == 0:

            display.clear_output(wait=True)
            checkpoint.save(file_prefix=checkpoint_prefix)




        stampa.plot_loss(train_losses, epoch+1)


        # Calcolo temopo
        total_time = (total_time + (time.time() - start))
        orario = arrow.now().replace(hour=0, minute=0).shift(minutes=(total_time / 60))

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start), " - total time:",
              orario.format('HH:mm'))


    # Generate after the final epoch
    display.clear_output(wait=True)




# Richiamo i miei dati come TFrecord

train_dataset_satellite, train_dataset_mask = rc.importTFRecord()

# Train

if True:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

epochs = 30

train(train_dataset_mask, train_dataset_satellite, epochs)


