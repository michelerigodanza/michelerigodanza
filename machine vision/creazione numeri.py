import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Carica il dataset MNIST
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0  # Normalizza le immagini

# Definisci il generatore
generator = Sequential([
    Dense(128, input_shape=(100,), activation='relu'),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28))
])

# Definisci il discriminatore
discriminator = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compila il generatore
generator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002))

# Compila il discriminatore
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002), metrics=['accuracy'])

# Congela il discriminatore durante l'addestramento del generatore
discriminator.trainable = False

# Definisci la GAN
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002))

# Funzione per addestrare la GAN
def train_gan(epochs=11, batch_size=128):
    batch_count = train_images.shape[0] // batch_size

    for e in range(epochs):
        for _ in range(batch_count):
            noise = np.random.normal(0, 1, size=[batch_size, 100])
            generated_images = generator.predict(noise)
            image_batch = train_images[np.random.randint(0, train_images.shape[0], size=batch_size)]

            X = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9  # Etichetta positiva per campioni reali

            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, y_dis)

            noise = np.random.normal(0, 1, size=[batch_size, 100])
            y_gen = np.ones(batch_size)  # Etichetta positiva per campioni falsi
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y_gen)

        print(f'Epoch {e+1}, D Loss: {d_loss[0]}, G Loss: {g_loss}')

        if e % 10 == 0:
            plot_generated_images(e, generator)

# Funzione per visualizzare immagini generate durante l'addestramento
def plot_generated_images(epoch, generator, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, size=[examples, 100])
    generated_images = generator.predict(noise)
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gan_generated_image_epoch_{epoch}.png')
    plt.show()

# Addestra la GAN
train_gan(epochs=100, batch_size=128)
