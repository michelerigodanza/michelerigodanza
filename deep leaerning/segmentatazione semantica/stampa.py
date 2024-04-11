import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display


def visualize_image(image, true_mask, predicted_mask):
    for image, true_mask, predicted_mask in zip(image, true_mask, predicted_mask):

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.title('Input Image')


        plt.subplot(1, 4, 2)
        plt.imshow(true_mask)
        plt.title('True Mask')


        plt.subplot(1, 4, 3)
        plt.imshow(predicted_mask)
        plt.title('Predicted Mask')


        num_classes = 3
        predicted_mask = tf.one_hot(tf.argmax(predicted_mask, axis=-1), depth=num_classes)
        plt.subplot(1, 4, 4)
        plt.imshow(predicted_mask)
        plt.title('Predicted Mask con onehot')


        plt.show()


def visualize_singolo(image, true_mask, predicted_mask, epoch):

        plt.figure(figsize=(20, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Input Image')

        plt.subplot(1, 3, 2)
        plt.imshow(true_mask)
        plt.title('True Mask')


        plt.subplot(1, 3, 3)

        num_classes = 3

        # Convertire il tensore in formato one-hot
        predicted_mask = tf.one_hot(tf.argmax(predicted_mask, axis=-1), depth=num_classes)

        plt.imshow(predicted_mask)
        plt.title('Predicted Mask')

        plt.savefig('Img2/Output_at_epoch_{:04d}.png'.format(epoch))
        plt.close()


def plot_loss_acc(train_losses, val_accuracies, epochs):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(1, (epochs + 1)), train_losses, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    if (epochs + 1) % 10 == 0:
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.scatter(range(1, epochs + 1), val_accuracies, color=color, label='Validation Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Training Loss and Validation Accuracy over Epochs')
    #plt.show()

    plt.savefig('Img2/LOSS_ACC_at_epoch_{:04d}.png'.format(epochs+1))
    plt.close()

    display.clear_output(wait=True)

    return

def plot_loss(train_losses, epochs):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(1, (epochs+1)), train_losses, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)


    fig.tight_layout()
    plt.title('Training Loss and Validation Accuracy over Epochs')
    #plt.show()

    plt.savefig('Img2/LOSS_at_epoch_{:04d}.png'.format(epochs))
    plt.close()

    display.clear_output(wait=True)

    return