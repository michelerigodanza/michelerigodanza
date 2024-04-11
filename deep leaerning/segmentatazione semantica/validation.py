import numpy as np
import TFrecord as rc
from rete_Unet import modello_Unet, model_loss
from accuracy import calculate_accuracy

val_dataset_satellite, val_dataset_mask = rc.importTFRecord_val()
# Definisci la funzione di convalida
def validate(model, model_loss, test_dataset_mask, test_dataset_satellite):
    losses = []

    # Itera su ogni campione nel set di test
    for satellite, mask in zip(test_dataset_satellite, test_dataset_mask):
        # Esegui la previsione del modello
        predicted_mask = model(satellite, training=False)
        # Calcola la perdita
        loss = model_loss(predicted_mask, mask)
        # Aggiungi la perdita alla lista delle perdite
        losses.append(loss)

    # Calcola la perdita media su tutto il set di test
    mean_loss = np.mean(losses)

    return mean_loss

# Esegui la convalida del modello dopo il training
test_loss = validate(modello_Unet, model_loss, val_dataset_mask, val_dataset_satellite)
print("Test Loss:", test_loss)


accuracy = calculate_accuracy(modello_Unet, val_dataset_mask, val_dataset_satellite)
print("Accuracy:", accuracy)

"""
Val_Test Loss: 0.86660534
Val_Accuracy: 0.7190764709019833 

train su un dataset da 45k elementi 
e validation su un dataset da 7k elementi
"""