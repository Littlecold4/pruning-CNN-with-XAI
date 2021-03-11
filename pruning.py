import torch
import numpy as np
import pickle
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import timeit

from createModel import CNN
from functions import test_mnist, pruning, createMask

# Import the original model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

PATH = './models/MNIST_CNN.pth'
model_orig = CNN().to(device)
model_orig.load_state_dict(torch.load(PATH))

model_test = CNN().to(device)
model_test.load_state_dict(torch.load(PATH))

model_pruned = CNN().to(device)
model_pruned.load_state_dict(torch.load(PATH))

model_pruned2 = CNN().to(device)
model_pruned2.load_state_dict(torch.load(PATH))


# Variables to be used
mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
conv_layers = [0,3,6]
filter_pos = []
for i in conv_layers:
    filter_pos.append(list(np.arange(len(model_orig.features[i].weight))))

FILE = "./gradcam/gradcam_trainingset.pickle"
file = open(FILE, "rb")
gradcam = pickle.load(file)


# Original model accuracy
accuracy_orig = test_mnist(model_orig, mnist_test)


# Set all elements of mask to 1 and test the code
m_test = createMask(gradcam, 0, 0)

print("=====Set all elements of mask to 1 and test the code=====\n")
print("-mask-\n",m_test["mask"])
print()

model_test = pruning(model_test, m_test["mask"], conv_layers, filter_pos)
accuracy_test=test_mnist(model_test, mnist_test)

print("\nOriginal model accuracy:",accuracy_orig)
print("Accuracy after pruning: ",accuracy_test)
print("Pruning rate per layer(test): ", m_test["rate"])
print()


# pruning1

conv1mean = np.array(gradcam[1]).mean()
conv2mean = np.array(gradcam[2]).mean()

m_prun1 = createMask(gradcam, conv1mean, conv2mean)

print("=====pruning1(threshold is the averagy of each layer)=====\n")
print("-mask-\n",m_prun1["mask"])
print()

model_pruned = pruning(model_pruned, m_prun1["mask"], conv_layers, filter_pos)
accuracy_pruned=test_mnist(model_pruned, mnist_test)

print("\nOriginal model accuracy:",accuracy_orig)
print("Accuracy after pruning: ",accuracy_pruned)
print("Pruning rate per layer: ", m_prun1["rate"])
print()

# save pruned model
PATH_prun1 = './models/MNIST_CNN_pruned.pth'
torch.save(model_pruned.state_dict(), PATH_prun1)


# pruning2

# Reduce pruning rate

conv1std = (np.array(gradcam[1]).mean()-np.array(gradcam[1]).std()*0.3)
conv2std = (np.array(gradcam[2]).mean()-np.array(gradcam[2]).std()/2)

m_prun2 = createMask(gradcam, conv1std, conv2std)

print("=====pruning2(reduce pruning rate)=====\n")
print("-mask-\n",m_prun2["mask"])
print()

model_pruned2 = pruning(model_pruned2, m_prun2["mask"], conv_layers, filter_pos)
accuracy_pruned2=test_mnist(model_pruned2, mnist_test)

print("\nOriginal model accuracy:",accuracy_orig)

print("\nPruning rate per layer(pruning1): ", m_prun1["rate"])
print("\tAccuracy after pruning1: ",accuracy_pruned)

print("\nPruning rate per layer(pruning2): ", m_prun2["rate"])
print("\tAccuracy after pruning2 (reduce pruning rate): ",accuracy_pruned2)

# save pruned model
PATH_prun2 = './models/MNIST_CNN_pruned_adjusted.pth'
torch.save(model_pruned2.state_dict(), PATH_prun2)
