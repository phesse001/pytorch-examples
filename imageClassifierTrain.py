import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

# Load and Normalize the CIFARR10 training and test datasets using torchvision
# Define a convolutional nn
# Define a loss function
# Train nn
# Test nn

# Note: for pretrained models can call torchvision.models.<model_name>(pretrained=true)

# Define Conv Net
import torch.nn as nn
import torch.nn.functional as F

#####################################CLASS AND HELPER FUNCTION###############################

class Net(nn.Module):
    def __init__(self):
        # I think super constructor is here to inherit attributes from nn.Module
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

# Viualize dataset

def imgshow(img):
    # unormalize
    img = img / 2 + .5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


# Get Data

def main():
    print("Number of threads " + str(torch.get_num_threads()))
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # print(transform) -> Compose(ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    trainset = torchvision.datasets.CIFAR10(root='./data_cifar', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=5,
                                              shuffle=True, num_workers=4)
    dataiter = iter(trainloader)
    images,labels = dataiter.next()

    imgshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(5)))

    net = Net()

    # Define a loss function and optimizer

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss() #the loss function
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=.9) #updates parameters based off of current gradients
    optimizer = optim.Adam(net.parameters(), lr = 0.0001)

    # Train the nn

    for epoch in range(50):
        # loop over the dataset multiple times (twice)
        running_loss = 0
        for num, data in enumerate(trainloader, 0):
            inputs,labels = data

            optimizer.zero_grad() #zero the parameter gradients, otherwise they will add up from previous passes

            # forward -> backward -> optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward() #AUTOGRAD!
            optimizer.step() #applys param updates

            running_loss += loss.item()

            # print every 2000 (1999 % 2000 == 1999)
            if num % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, num + 1, running_loss / 2000))
                running_loss = 0

    #save the model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

if __name__ == "__main__":
    main()
