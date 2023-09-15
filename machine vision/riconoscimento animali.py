

REBUILD_DATA = False

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm



class DogsVSCats():
    IMG_SIZE=50
    CATS = "PetImages\Cat"
    DOGS = "PetImages\Dog"
    TESTING = "PetImages\Testing"
    LABELS= {CATS:0, DOGS:1}
    training_data = []
    cc=0
    dg=0

    def make_training_data(self):
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path=os.path.join(label, f)
                        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                        img=cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
                        self.training_data.append([np.array(img),np.eye(2)[self.LABELS[label]]])

                        if label == self.CATS:
                            self.cc+=1
                        elif label == self.DOGS:
                            self.dg+=1

                    except Exception as e:
                        pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print('Cats: ', self.cc)
        print('Dogs: ', self.dg)


if REBUILD_DATA:
    dvc = DogsVSCats()
    dvc.make_training_data()

""""
td=np.load("training_data.npy", allow_pickle=True)
print(len(td))

X = torch.Tensor([i[0] for i in td]).view( -1, 50, 50)
X = X / 255.0
y = torch.Tensor([i[0] for i in td])

plt.imshow(X[0], cmap="gray")
print(y[0])
plt.show()
"""

training_data=np.load("training_data.npy", allow_pickle=True)
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1,  50, 50)
        self._to_Linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_Linear , 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_Linear == None:
            self._to_Linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_Linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


net = Net()

import torch.optim as optim
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()


X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X / 255.0
Y = torch.Tensor([i[1] for i in  training_data])

VAL_PCT = 0.1
val_size = int(len(X) * VAL_PCT)
print('Val size: {}\n'.format(val_size))

train_X = X[:-val_size]
train_Y = Y[:-val_size]

test_X = X[-val_size:]
test_Y = Y[-val_size:]

BATCH_SIZE = 100
EPOCHS = 1

print('training starting...\n')
for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        batch_x = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_Y[i:i+BATCH_SIZE]

        net.zero_grad()
        outputs = net(batch_x)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_Y[i])

        net_out=net(test_X[i].view(-1, 50 , 50))[0]

        predicted_class = torch.argmax((net_out))
        if i<=10:
            plt.imshow(X[i], cmap="gray")
            if predicted_class==1:
                plt.title('DOG')
                print(predicted_class, ' DOG')
            elif predicted_class==0:
                plt.title('CAT')
                print(predicted_class, ' CAT')
            plt.imshow(X[i], cmap="gray")

            plt.show()


        if predicted_class == real_class:
            correct+=1
        total +=1

print('Accuracy: {}'.format(correct/total))















