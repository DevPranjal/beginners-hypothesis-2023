import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from zipfile import ZipFile
from tqdm.auto import tqdm
import os

############################################################
#### defining the network, optimizer, and loss function ####
############################################################

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

########################
#### unzipping data ####
########################

print('----> unzipping train_data.zip into unzipped_data')

with ZipFile('train_data.zip', 'r') as f:
    f.extractall('unzipped_data')

################################
#### preparing data loaders ####
################################

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4901, 0.4617, 0.4061], [0.1977, 0.1956, 0.1947])
])

data = torchvision.datasets.ImageFolder(
    root='unzipped_data',
    transform=transform
)

classes = data.classes
dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, num_workers=1)

#############################################
#### loading pretrained model, if exists ####
#############################################

if os.path.exists('model.pt'):
    print('----> loaded pretrained model')
    net.load_state_dict(torch.load('model.pt'))

###########################
#### starting training ####
###########################

num_epochs = 15

print(f'----> starting training for {num_epochs} epochs')

with tqdm(range(num_epochs), desc='training') as training_bar:
    for epoch in training_bar:
        running_loss = 0.0
        epoch_bar = tqdm(enumerate(dataloader), desc=f'epoch {epoch + 1}')
        for i, data in epoch_bar:
            inputs, labels = data
            outputs = net(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_bar.set_postfix(average_running_loss=running_loss / (i + 1))

print('----> finished training')

##############################
#### testing on train set ####
##############################

correct = 0
total = 0
with torch.no_grad():
    for data in dataloader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'----> accuracy on the dataset: {100 * correct / total:.2f}%')

####################
#### save model ####
####################

print('----> saving model to model.pt')

torch.save(net.state_dict(), 'model.pt')
