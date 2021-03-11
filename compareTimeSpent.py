import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import timeit

from createModel import CNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mnist_train = dsets.MNIST(root='MNIST_data/',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                             batch_size=100,
                                             shuffle=True,
                                             drop_last=True)

PATH = './models/MNIST_CNN.pth'
PATH_prun1 = './models/MNIST_CNN_pruned.pth'
PATH_prun2 = './models/MNIST_CNN_pruned_adjusted.pth'

model = CNN().to(device)
model.load_state_dict(torch.load(PATH))

model1 = CNN().to(device)
model1.load_state_dict(torch.load(PATH_prun1))

model2 = CNN().to(device)
model2.load_state_dict(torch.load(PATH_prun2))
    
def process_time(model):
    N=10
    with torch.no_grad():
        for epoch in range(N):
            for X, Y in data_loader:
                X = X.to(device)
                Y = Y.to(device)

                out = model(X)


t = timeit.timeit(stmt='process_time(model)',setup="from __main__ import process_time, model", number=1)

t1 = timeit.timeit(stmt='process_time(model1)',setup="from __main__ import process_time, model1", number=1)

t2 = timeit.timeit(stmt='process_time(model2)',setup="from __main__ import process_time, model2", number=1)

print("model_orig :", t)
print("model_pruned1 :", t1)
print("model_pruned2:", t2)