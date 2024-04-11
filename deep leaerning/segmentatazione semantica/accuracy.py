import numpy as np
import TFrecord as rc
def calculate_iou(y_true, y_pred):
    y_true = np.squeeze(y_true, axis=-1)  # Rimuovi la dimensione del canale se presente
    intersection = np.sum(np.logical_and(y_true, y_pred))
    union = np.sum(np.logical_or(y_true, y_pred))
    iou = intersection / (union + 1e-7)  # aggiungiamo 1e-7 per evitare la divisione per zero
    return iou


def calculate_accuracy(modello_Unet , dataset_mask, dataset_satellite):
    iou_scores = []
    for mask, satellite in zip(dataset_mask, dataset_satellite):
        predicted_mask = modello_Unet(satellite, training=False)
        predicted_mask = np.argmax(predicted_mask, axis=-1)  # assumiamo che il modello produca output in forma di probabilità, quindi applichiamo argmax per ottenere le predizioni

        iou = calculate_iou(mask, predicted_mask)
        iou_scores.append(iou)
    accuracy = np.mean(iou_scores)
    return accuracy



