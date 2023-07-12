from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import torch
from loader import udacity
from torchvision import transforms
from torchvision import utils as vutils
from matplotlib import pyplot as plt 
import numpy as np 

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated











# Data-loader
trans = transforms.Compose([transforms.Resize((224, 224)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])])

dsets_train = udacity("/Users/bryanzen/Desktop/Udacity training",
                              transforms=trans, training=True)
dsets_test = udacity("/Users/bryanzen/Desktop/Udacity training",
                              transforms=trans, training=False)

train_loader = torch.utils.data.DataLoader(dsets_train, batch_size=2,
                                           shuffle=True, num_workers=2)








# Visualizing the data
for data in train_loader:
    img, steer_angle = data['img'], data['steer_cmd']
    break
    
img = vutils.make_grid(img)
imshow(img)
print(steer_angle.numpy().reshape(-1).tolist())















def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model





























def visualize_model(model, num_images=3):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
 
MODEL CODE (all in one cell)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.conv2d( in_channels = 3, out_channels = 24 , kernel_size = 3, stride = 2 )
        self.conv2=nn.conv2d( in_channels = 24 , out_channels = 48, kernel_size = 3, stride = 2)
        
        self.fc1=nn.linear( in_feature = 48*4*4, out_features = 50)
        self.fc2=nn.linear( in_feature = 50, out_features = 10)
        self.out=nn.linear( in_feature = 10, out_features = 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size =2,stride = 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size =2,stride = 2)
        
        x = x.view(-1,48*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



-MODEL TRIALS-

TRIAL #1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d( in_channels = 3, out_channels = 24 , kernel_size = 3, stride = 2 )
        self.conv2=nn.Conv2d( in_channels = 24 , out_channels = 48, kernel_size = 3, stride = 2)
        
        self.fc1=nn.Linear( in_features = 432, out_features = 50)
        self.fc2=nn.Linear( in_features = 50, out_features = 10)
        self.out=nn.Linear( in_features = 10, out_features = 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size =2,stride = 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size =2,stride = 2)
        
        x = x.view(-1,432)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.out(x)
        return x

Batch_size = 64
Num_workers = 25

Train loss = 0.0821
Val loss = 0.0821
Best Val loss = 0.081732
-TRAIL #2-
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d( in_channels = 3, out_channels = 25 , kernel_size = 3, stride = 2 )
        self.conv2=nn.Conv2d( in_channels = 25 , out_channels = 50, kernel_size = 3, stride = 2)
        
        self.fc1=nn.Linear( in_features = 450, out_features = 50)
        self.fc2=nn.Linear( in_features = 50, out_features = 10)
        self.out=nn.Linear( in_features = 10, out_features = 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size =2,stride = 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size =2,stride = 2)
        
        x = x.view(-1,450)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.out(x)
        return x
 Batch_size = 64
Num_workers = 25

Train loss = 0.0826
Val loss = 0.0828
Best Val loss = 0.082544

class Net(nn.Module):




   def __init__(self):


       super(Net, self).__init__()


      


       self.conv1 = nn.Conv2d(3, 24, 5, stride=(2, 2))       


       self.conv2 = nn.Conv2d(24, 36, 5, stride=(2, 2))


       self.conv3 = nn.Conv2d(36, 48, 5, stride=(2, 2))


       self.conv4 = nn.Conv2d(48, 64, 3)


       self.conv5 = nn.Conv2d(64, 64, 3)


       self.pool = nn.MaxPool2d(2, 2)


       self.drop = nn.Dropout(p=0.5)


       self.fc1 = nn.Linear(64 * 3 * 13, 100)


       self.fc2 = nn.Linear(100, 50)


       self.fc3 = nn.Linear(50, 10)


       self.fc4 = nn.Linear(10, 1)







   def forward(self, x):


       x = F.elu(self.conv1(x))          


       x = F.elu(self.conv2(x))


       x = F.elu(self.conv3(x))


       x = F.elu(self.conv4(x))


       x = F.elu(self.conv5(x))


       x = self.drop(x)


       # print(x.size())


       x = x.view(-1, 64 * 3 * 13)


       x = F.elu(self.fc1(x))


       x = F.elu(self.fc2(x))


       x = F.elu(self.fc3(x))


       x = self.fc4(x)


       return x







class Model():   

(MIGHT WORK FOR BOTH MODELS)


