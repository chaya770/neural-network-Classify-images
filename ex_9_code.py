import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sn
import pandas  as pd
import numpy as np
import seaborn as sn
import pandas  as pd
import matplotlib.pyplot as plt

# Loading and normalizing CIFAR10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#  Define a Convolution Neural Network


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# Define a Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network
# This is when things start to get interesting. We simply have to loop over our data iterator, and feed the inputs to the network and optimize.

for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Loading the test file:
"""
test = pickle.load(open('test.pickle'))
for data in test:
    image = Variable(data)
    output = net(image)
"""
#dataiter = iter(testloader)
#images, labels = dataiter.next()

#outputs = net(images)
#predicted = torch.max(outputs,1)
file = open('test.pred', 'w')







correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _,predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for j in range(len(predicted)):
            file.write('%s\n' % predicted[j].item())

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# Test the network on the test data

data_loss = []
validation_loss = []
data_accuracy = []
validation_accuracy = []
#for epoch in range(1, 10 + 1):
  #  data_accuracy.append(train(epoch,model, data_loss))
   # validation_accuracy.append(validation(validation_loss))


"""
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(data_loss, label='train set')
axarr[0].plot(validation_loss, label='validation set')
axarr[0].set_title('Avg Loss')
axarr[0].legend()

axarr[1].plot(data_accuracy, label='train set')
axarr[1].plot(validation_accuracy, label='validation set')
axarr[1].set_title('Accuracy')
axarr[1].legend()
plt.xlabel("epochs")
plt.show()
"""

# Visualizing of confusion matrix


cm1 =np.matrix( [[599 ,  5 , 74,  98 , 55 ,  14,  12,   9, 117,  17],
                 [16,738  ,12  ,65  , 9  , 26  , 7 ,  6 , 40 , 81],
                 [ 31  , 0 ,523 ,168 ,136 ,  86 , 33 , 14 ,  9 ,  0],
                 [ 10  , 1 , 31 ,652 , 90 , 175  ,19 , 15  , 5  , 2],
                 [  6   ,0,  34, 132, 717 ,  55 , 16 , 31 ,  9 ,  0],
                 [  5  , 1  ,17 ,233  ,53  ,661 , 10  ,15 ,  4  , 1],
                 [  2   ,1  ,39 ,157 ,105 ,  48 ,637 ,  3 ,  7 ,  1],
                 [  6   ,0,  14  ,97 ,103  , 96  , 5, 637  , 5   ,1],
                 [ 41  , 7 , 28  ,84 , 19  , 18 ,  6 ,  4 ,783  ,10],
                 [ 25 , 28  , 8 , 77 , 29   ,27  , 5 , 19 , 59 ,723]])


cm=np.matrix([[736 , 11,  54 , 45 , 30 , 14 , 15 ,  9 , 61 , 25],
 [ 10 ,839 ,  6 , 38 ,  3 , 13 ,  7 ,  5 , 22 , 57],
 [ 47  , 2 ,666 , 96, 45  ,65 , 51 , 17  , 7  , 4],
 [ 23   ,6 , 56, 670 , 97 ,40,  57 , 29 , 12 , 10],
 [ 16   ,2  ,52  ,80 ,700,  55  ,25 , 64  , 3  , 3],
 [ 10   ,1 , 64, 211 , 59 ,582 , 24  ,39  , 6  , 4],
 [  4  , 3  ,42 ,114 ,121  ,40 ,650 , 13   ,5   ,8],
 [ 14   ,1  ,40 , 57 , 69  ,8  ,11 ,783 ,  3  ,14],
 [ 43 , 32  ,26  ,37 , 16 , 15   ,6  , 2 ,802,  21],
 [ 34 , 23 ,  8 , 42 , 12  ,21  , 6,  21 , 25 ,808]])

df_cm = pd.DataFrame(cm, range(10),
                     range(10))
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})  # font size
plt.show()