import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import torch.nn.init
import random

from functions import test_mnist

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32,64,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1=nn.Linear(3*3*128, 625, bias = True)
        self.relu = nn.ReLU()
        self.fc2=nn.Linear(625, 10, bias = True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x):
        out = self.features(x)
        
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for reproducibility
    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    # parameters
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100

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
    data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True)

    # model
    model = CNN().to(device)

    # define cost/loss & optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #training
    total_batch = len(data_loader)
    
    print("Start training...\n")
    for epoch in range(training_epochs):
        avg_cost = 0 #loss

        for X, Y in data_loader: #X=input, Y=label
            X = X.to(device) # not torch tensor, toch cuda tensor
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch

        print('[Epoch:{}] cost = {}'.format(epoch+1, avg_cost))

    print('Training finished')

    #save the model
    PATH = './models/MNIST_CNN.pth'
    torch.save(model.state_dict(), PATH)

    # Test the model using test sets
    accuracy = test_mnist(model, mnist_test)
    print('Accuracy:', accuracy)
