import foolbox as fb
import torch
from torch import nn
import torch.nn.functional as F
import os
from zipfile import ZipFile
import torchvision
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
args = parser.parse_args()

############################################################
#### loading pretrained network, which we have to evade ####
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
net.load_state_dict(torch.load(args.model_path))
net.eval()

########################
#### unzipping data ####
########################

print('----> unzipping train_data into unzipped_data')

with ZipFile(args.train_data, 'r') as f:
    f.extractall('unzipped_data')

data = torchvision.datasets.ImageFolder(root='unzipped_data', transform=transforms.ToTensor())
images, labels = [], []
for X, y in data:
    images.append(X)
    labels.append(torch.tensor(y))

images = torch.stack(images)
labels = torch.stack(labels)

####################################################################
#### generating adversarial/evasion examples using FGSM and PGD ####
####################################################################

preprocessing = dict(mean=[0.4901, 0.4617, 0.4061], std=[0.1977, 0.1956, 0.1947], axis=-3)
fnet = fb.PyTorchModel(net, bounds=(0, 1), preprocessing=preprocessing, device='cpu')
fgsm = fb.attacks.FGSM()
pgd = fb.attacks.PGD()

print('----> generating adversarial samples using FGSM')

_, fgsm_clipped_advs, fgsm_success = fgsm(fnet, images, labels, epsilons=0.01)

print('----> generating adversarial samples using PGD')

_, pgd_clipped_advs, pgd_success = pgd(fnet, images, labels, epsilons=0.01)

##################################################################
#### calculating accuracies on adversarial examples generated ####
##################################################################

clean_accuracy = fb.utils.accuracy(fnet, images, labels)
fgsm_robust_accuracy = (1 - fgsm_success.float().mean()).item()
pgd_robust_accuracy = (1 - pgd_success.float().mean()).item()

print(f'----> clean accuracy of the model: {clean_accuracy}')
print(f'----> robust accuracy for FGSM attack: {fgsm_robust_accuracy}')
print(f'----> robust accuracy for PGD attack: {pgd_robust_accuracy}')

############################################
#### saving adversarial images for test ####
############################################

print('----> saving adversarial images')

to_pil = transforms.ToPILImage()
fgsm_images = [to_pil(img) for img in fgsm_clipped_advs]
pgd_images = [to_pil(img) for img in pgd_clipped_advs]

os.system('rm -rf test_data_fgsm && mkdir test_data_fgsm')
os.system('rm -rf test_data_pgd && mkdir test_data_pgd')

for i, d in enumerate(fgsm_images):
    d.save(f'test_data_fgsm/test_{i}.jpg')

for i, d in enumerate(pgd_images):
    d.save(f'test_data_pgd/test_{i}.jpg')

