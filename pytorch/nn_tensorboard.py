#!/usr/bin/env python
# coding: utf-8

# # Neural Networks

# In[1]:


# MNIST 
# DataLoader, Transformations
# Multi-layer neural networks, activation functions
# Loss and optimizer
# Training loop w/ batch training
# Model evaluation
# GPU support


# In[1]:


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys


# In[ ]:


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist")


# In[20]:


device = torch.device('cuda')


# In[4]:


input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001


# In[ ]:


train_dataset = torchvision.datasets.MNIST(root='./data', 
                train=True, transform=transforms.ToTensor(),
                download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', 
                train=False, transform=transforms.ToTensor())


# In[9]:


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                 batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                 batch_size=batch_size, shuffle=False)


# In[10]:


examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)


# In[13]:


for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(samples[i][0], cmap='gray')
#     plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images', img_grid)
writer.close()
#sys.exit()


# In[15]:


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


# In[23]:


model = NeuralNet(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, samples.reshape(-1, 28*28))
writer.close()
sys.exit()


# In[25]:


total_steps = len(train_loader)

running_loss = 0.0
running_correct = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, pred = torch.max(outputs.data, 1)
        running_correct += (pred == labels).sum().item() 
        
        # Progress
        if (i+1)%100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{total_steps}, loss = {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss/100, epoch * total_steps + i)
            writer.add_scalar('accuracy', running_correct/100, epoch * total_steps + i)
            running_loss = 0.0
            running_correct = 0

# # In[29]:


# # Evaluation
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 28*28)
#         outputs = model(images)
        
#         # value, index of network pred output tensors
#         _, preds = torch.max(outputs, 1)
#         n_samples += labels.shape[0]
#         n_correct += (preds == labels).sum().item()
        
#     acc = 100.0 * n_correct / n_samples
#     print(n_correct)
#     print(n_samples)
#     print(f'accuracy = {acc}')


# # # Convolutional Neural Networks

# # In[30]:


# num_epochs = 4
# batch_size = 4
# learning_rate = 0.001


# # In[ ]:


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# )


# # In[31]:


# train_dataset = torchvision.datasets.CIFAR10(root='./data', 
#                 train=True, transform=transforms.ToTensor(),
#                 download=True)
# test_dataset = torchvision.datasets.CIFAR10(root='./data', 
#                 train=False, transform=transforms.ToTensor())


# # In[32]:


# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                  batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                  batch_size=batch_size, shuffle=False)


# # In[33]:


# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# # In[67]:


# example = iter(train_loader)
# images, labels = example.next()
# print(f'{images.shape}')
# print(f'{labels.shape}')


# # In[68]:


# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# # In[69]:


# imshow(torchvision.utils.make_grid(images))


# # In[ ]:


# conv1 = nn.Conv2d(3, 6, 5)
# pool = nn.MaxPool2d(2, 2)
# conv2 = nn.Conv2d(6, 15, 5)
# print(f'Initial shape: {images.shape}')

# x = conv1(images)
# print(f'Shape after conv1: {x.shape}')
# x = pool(images)
# print(f'Shape after conv1: {x.shape}')


# # In[50]:


# import torch.nn.functional as F

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # input channel size, output channel size, kernel size
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         # kernel size, stride length
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # Dimensions after convolutions, pooling, and flattening
#         self.fc1 = nn.Linear(16*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
    
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16*5*5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# # In[51]:


# model = ConvNet()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# # In[52]:


# total_steps = len(train_loader)


# # In[53]:


# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if (i+1)%2000 == 0:
#             print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{total_steps}, loss = {loss.item():.4f}')
# print('Finished training.')


# # In[56]:


# # Evaluation
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
    
#     n_class_correct = [0 for i in range(10)]
#     n_class_samples = [0 for i in range(10)]
    
#     for images, labels in test_loader:
#         outputs = model(images)
        
#         # value, index of network pred output tensors
#         _, preds = torch.max(outputs, 1)
#         n_samples += labels.shape[0]
#         n_correct += (preds == labels).sum().item()
        
#         for i in range(batch_size):
#             label = labels[i]
#             pred = preds[i]
#             if (label == pred):
#                 n_class_correct[label] += 1
#             n_class_samples[label] += 1
        
#     acc = 100.0 * n_correct / n_samples
#     print(f'Network accuracy: {acc}') 
    
#     for i in range(10):
#         acc = 100* n_class_correct[i] / n_class_samples[i]
#         print(f'Accuracy of {classes[i]}: {acc}%')

