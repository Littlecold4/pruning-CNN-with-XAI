import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import pickle

from createModel import CNN
from layer_activation_with_guided_backprop import GuidedBackprop
from functions import getContribution

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
# dataset loader
batch_size=100
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
training_epochs = 15

# import model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)
PATH = './models/MNIST_CNN.pth'
model.load_state_dict(torch.load(PATH))

conv_layers = [0,3,6]
filter_pos = []
gradcam = []

for i in conv_layers:
    filter_pos.append(list(np.arange(len(model.features[i].weight))))

for i in conv_layers:
    gradcam.append(list(np.zeros(len(model.features[i].weight))))   

# extract the gradcam
model.eval()
GBP = GuidedBackprop(model)

total_batch = len(data_loader)

print("Start gradcam extraction.")
for epoch in range(training_epochs):

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)
        
        numOfImg=0
        for i in range(3):
            for filter in filter_pos[i]:
                gradcam[i][filter]+=getContribution(GBP, X[numOfImg].cpu(), np.array(Y.cpu())[0],conv_layers[i],filter)
        numOfImg+=1

    print('[Epoch:{}]'.format(epoch+1))
print('grandcam extraction finished')

# save the gradcam
FILE = "./gradcam/gradcam_trainingset.pickle"
with open(FILE, "wb") as h:
    pickle.dump(gradcam, h)
