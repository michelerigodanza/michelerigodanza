import os
from PIL import Image
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import numpy as np
from numpy import mean

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

REBUILD_DATA = False



### Definisci le cartelle dei dati #####################################################################################
data_dir = r'C:\Users\MICHELE\PycharmProjects\pythonProject2\CV\caffe\dataset'
image_folder = 'img'
label_folder = 'masks'

# Ottieni una lista di percorsi alle immagini e alle etichette
image_paths = [os.path.join(data_dir, image_folder, filename) for filename in os.listdir(os.path.join(data_dir, image_folder))]
label_paths = [os.path.join(data_dir, label_folder, filename) for filename in os.listdir(os.path.join(data_dir, label_folder))]

# Dividi il dataset in insiemi di addestramento e convalida
train_image_paths = image_paths[:4000]
train_label_paths = label_paths[:4000]
val_image_paths = image_paths[4000:]
val_label_paths = label_paths[4000:]

# Trasformazione per le immagini
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

### Classe per il dataset ##############################################################################################
class SatelliteDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = Image.open(self.label_paths[idx]).convert('L')  # Converti in scala di grigi

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

### Crea il DataLoader per il set di addestramento #####################################################################
batch_size = 18
train_dataset = SatelliteDataset(train_image_paths, train_label_paths, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Calcola il numero di elementi nel dataset
num_elements = len(train_loader)

# Stampa il numero di elementi
print(f"Numero di elementi nel train_loader: {num_elements}")

# Calcola il numero di elementi nel dataset
num_elements = len(train_dataset)

# Stampa il numero di elementi
print(f"Numero di elementi nel train_dataset: {num_elements}")

### Definizione del modello U-Net ######################################################################################
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Definire l'architettura del tuo modello U-Net qui

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Middle

        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,  padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )



    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

### Creazione dell'istanza del modello #################################################################################

model = UNet()
import torch

# Verifica se una GPU CUDA è disponibile e imposta il dispositivo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sposta il modello sulla GPU se disponibile
model.to(device)


# Definizione della funzione di perdita e dell'ottimizzatore
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy per immagini in scala di grigi
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

### Addestramento del modello ##########################################################################################
num_epochs = 60
if REBUILD_DATA:
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            # Carica i dati sulla GPU utilizzando .to(device)
            images = images.to(device)
            labels = labels.to(device)
            ####
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    # Salva il modello
    #torch.save(model.state_dict(), 'unet_modelGrey.pth')



############################# Carica i pesi precedentemente salvati ##################################################################################################################################################################################

model.load_state_dict(torch.load(r'C:\Users\MICHELE\PycharmProjects\pythonProject2\CV\caffe\unet_modelGrey.pth'))
print('modello caricato')


# Crea il DataLoader per il set di convalida

val_dataset = SatelliteDataset(val_image_paths, val_label_paths, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_elements = len(val_loader)
# Stampa il numero di elementi
print(f"Numero di elementi in val_loader: {num_elements}")


### Valutazione del modello#############################################################################################

CONVALIDA = False
if CONVALIDA:
    model.eval()
    val_loss = 0.0
    num_samples = 0

    ####################################################################################################################

    for images, labels in val_loader:
        # Carica i dati sulla GPU utilizzando .to(device)
        images = images.to(device)
        labels = labels.to(device)

        ###
        with torch.no_grad():
            outputs = model(images)

        loss = criterion(outputs, labels)
        val_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

    average_val_loss = val_loss / num_samples
    print(f'Average Validation Loss: {average_val_loss:.4f}')

    ### Calcolo dell'accuracy ##########################################################################################
    def compute_accuracy(model, dataloader):
        correct_predictions = 0
        total_samples = 0

        model.eval()  # Imposta il modello in modalità di valutazione

        with torch.no_grad():
            for images, labels in val_loader:  # Usa il tuo dataloader di convalida
                images = images.to(device)  # Sposta le immagini sulla GPU se necessario
                labels = labels.to(device)  # Sposta le etichette sulla GPU se necessario

                outputs = model(images)

                predicted = (outputs > 0.5).float()  # Esempio di soglia per problemi di classificazione binaria

                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.numel()
        # stampa(val_loader)
        accuracy = (correct_predictions / total_samples) * 100
        print(f'Accuracy: {accuracy:.2f}%')

    compute_accuracy(model, train_loader)

    ### Calcolo IoU#####################################################################################################

    def compute_iou(outputs, labels):
        # Applica una soglia ai valori di output
        outputs = (outputs > 0.5).float()

        intersection = torch.sum(outputs * labels)
        union = torch.sum(outputs) + torch.sum(labels) - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)  # Aggiungi una piccola costante per evitare divisione per zero

        return iou.mean()


    for images, labels in val_loader:
        # Carica i dati sulla GPU utilizzando .to(device)
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)

        iou = compute_iou(outputs, labels)

    print('Iou Medio:', mean(iou.item()))


########################################################################################################################
def stampa(val_loader):
    # Visualizza alcune immagini insieme alle maschere e alle previsioni
    model.eval()
    num_images_to_display = 10
    def compute_iou(outputs, labels):
        # Applica una soglia ai valori di output
        outputs = (outputs > 0.5).float()

        intersection = torch.sum(outputs * labels)
        union = torch.sum(outputs) + torch.sum(labels) - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)  # Aggiungi una piccola costante per evitare divisione per zero

        return iou.mean()

    for i, (images, labels) in enumerate(val_loader):
        if i >= num_images_to_display:
            break

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predicted = torch.sigmoid(outputs) > 0.5  # Applica una soglia di 0.5 per la classificazione binaria

        iou = compute_iou(outputs, labels)


        images = images.cpu().numpy()  # Sposta l'immagine sulla CPU e converti in formato NumPy
        labels = labels.cpu().numpy()
        predicted = predicted.cpu().numpy()


        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(np.transpose(images[0], (1, 2, 0)))  # Visualizza l'immagine
        axes[0].set_title('Image')
        axes[1].imshow(labels[0][0], cmap='gray')  # Visualizza la maschera reale
        axes[1].set_title('Ground Truth Mask')
        axes[2].imshow(predicted[0][0], cmap='gray')  # Visualizza la previsione del modello
        axes[2].set_title(f'Predicted Mask:\n iou: {iou.item():.4f}')
        plt.show()

stampa(val_loader)


### Visualizzazione ####################################################################################################

# Carica un'immagine dal set di convalida
val_image = r'C:\Users\MICHELE\PycharmProjects\pythonProject2\CV\caffe\dataset\data\valid\64_sat.jpg'
val_image = Image.open(val_image)

# Applica la trasformazione alle immagini (stessa trasformazione usata durante l'addestramento)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

val_image = transform(val_image).unsqueeze(0)  # Aggiungi la dimensione del batch

# Sposta l'immagine sulla GPU se disponibile
val_image = val_image.to(device)

# Passa l'immagine attraverso il modello per ottenere le previsioni
model.eval()
with torch.no_grad():
    predictions = model(val_image)


# Applica una soglia alle previsioni per ottenere una maschera binaria (strade o non strade)
threshold = 0.5  # Puoi regolare la soglia a seconda del tuo modello
binary_mask = (predictions > threshold).squeeze().cpu().numpy()

# Visualizza l'immagine originale e la maschera binaria sovrapposte
plt.figure(figsize=(10, 5))

# Immagine originale
plt.subplot(1, 2, 1)
plt.imshow(val_image.squeeze().cpu().numpy().transpose(1, 2, 0))
plt.title('Immagine Originale')
plt.axis('off')

# Maschera binaria
plt.subplot(1, 2, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title('Maschera Binaria (Strade)')
plt.axis('off')

plt.show()

########################################################################################################################






