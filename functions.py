import torch
import torch.nn as nn
import numpy as np
import time

from layer_activation_with_guided_backprop import *

def preprocess_image_gray(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(1):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(np.flip(preprocessed_img,axis=0).copy())
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def getContribution (GBP, img, target_class, cnn_layer, filter_pos):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # preprocess the image
    img = img.squeeze().unsqueeze(2).numpy()
    prep_img = preprocess_image_gray(img).to(device)
    # extract gradcam
    guided_grads = GBP.generate_gradients(prep_img, target_class, cnn_layer, filter_pos)
    # expressing as a single value
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    filter_norm=np.linalg.norm(grayscale_guided_grads)
    
    return filter_norm

def createMask(gradcam, th1, th2):
    mask = np.array(gradcam)
    rate = [0,0,0]

    for i in range(3):
        mask[i] = np.array(gradcam[i])
    
    mask[1][mask[1]<th1]=0
    mask[1][mask[1]>=th1]=1

    mask[2][mask[2]<th2]=0
    mask[2][mask[2]>=th2]=1
    
    for i in range(len(mask)):
        if i!= 0:
            for j in range(len(mask[i])):
                if(mask[i][j] == 0):
                    rate[i] += 1
        rate[i] /= len(mask[i])
        rate[i] = round(rate[i], 2)
    
    m = {        
       "mask":mask,
        "rate":rate
    }
    
    return m

def pruning(model, mask, conv_layers, filter_pos):
    print("-pruning sample-")
    print("before pruning:\n",model.features[3].weight[0][0])
    with torch.no_grad():
        conv_seq_num=0
        i=0
        for conv_layer in filter_pos:
            for filters in conv_layers:
                if(conv_seq_num==0):
                    break
                model.features[conv_seq_num].weight[filters] *= mask[i][filters]
                if(mask[i][filters]==0):
                    model.features[conv_seq_num].weight[filters] = abs(model.features[conv_seq_num].weight[filters])
            conv_seq_num += 3
            i += 1
        print("after pruning:\n",model.features[3].weight[0][0])
    return model


def test_mnist(model, mnist_test):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with torch.no_grad():
        X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
        Y_test = mnist_test.test_labels.to(device)

        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        return accuracy.item()
    