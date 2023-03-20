# Beginners Hypothesis 2023 (For Sophomore Recruitments)

## Problem Statement

You are a cat lover (yes you are!). So you decide to open a cat park allowing owners to bring their cats. You however need to protect the park from other animals. Hence you turn towards the ***EvEr-So-ReLiAnT*** machine learning to tackle this. You ask your experienced friend Aaryan to build an image classification model to classify cats from other animals. And it works like a charm!

Sarthak however wants to get his pet deer into the park. And he knows how to get this done (insert wicked smile). [Inspired by previous research](https://towardsdatascience.com/avoiding-detection-with-adversarial-t-shirts-bb620df2f7e6), he steals the model from Aaryan's _Mac_ and uses it learn features that will help his deer classify itself as a cat. Too easy for Sarthak!

To save the park, you need to build a robust cat classification model, one which is less prone to such _attacks_. You will be provided with a dataset with two categories: cat, not-cat. This is the same dataset which was used to train the original model.

While you develop your model, we've hired Sarthak to provide us with more such features on other animals so that we can test the robustness of your classifier. Your model will be tried and tested on these images. _Accuracy of correct classification_ will be used to select what goes into the park camera.

## Technical Details

- Such attacks on machine learning models are called **Adversarial Attacks**. For the purpose of this problem statement we use the **FGSM Attack** to generate images which could evade the machine learning model.
- The initial baseline model (developed by Aaryan) is a simple CNN, trained on [`train_data.zip`](train_data.zip). More details on training can be found in the [`train_baseline.py`](train_baseline.py). The same file is used to produce [`model.pt`](model.pt) which has already been provided in the repository. 
- As mentioned above, Sarthak evaded the machine learning model by using the FGSM attack. You are encouraged to read more about it online. Adversarial samples generated using the FGSM attack are in stored with the archive [`test_data_fgsm.zip`](test_data_fgsm.zip).
- The sample images stored in [`test_data_fgsm.zip`](test_data_fgsm.zip) will be used to test your freshly trained machine learning model.
- For a reference, some numbers on the baseline model ([`model.pt`](model.pt)) are provided below:

| Data | Accuracy Score |
| -- | -- |
| Train Data ([`train_data.zip`](train_data.zip)) | 94.19% |
| Adversarial Images using FGSM ([`test_data_fgsm.zip`](test_data_fgsm.zip)) | 43.54% |

- Hence, the goal of this challenge is to train your machine learning model in such a way that accuracy score on Adversarial Images is improved. This is technically referred to as **Adversarial Training**.

## Some Starter Guidelines

- To load the model weights provided in [`model.pt`](model.pt), we recommend you to use PyTorch. In any Python script they can be loaded as follows:

```python
from torch import nn
import torch.nn.functional as F

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
net.load_state_dict(torch.load('model.pt'))

... # use a pretrained net
```

- [This blog-post](https://adversarial-ml-tutorial.org/introduction/) provides an amazing introduction to adversarial machine learning. It also provides examples of adversarial attack and adversarial training.
- Many adversarial defense techniques have been implemented [here](https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/image#defense-methods)

## Submission

You will need to submit your predictions to the images present in [`test_data_fgsm.zip`](test_data_fgsm.zip). The images are within the zip are named as `test_0.jpg`, `test_1.jpg`, etc. Your final submission will be a `.csv` file with the following sample contents:

| filename | predicition |
| -- | -- |
| test_0.jpg | 0 |
| test_1.jpg | 0 |
| test_2.jpg | 1 |
| test_3.jpg | 0 |
| test_4.jpg | 1 |
| test_5.jpg | 1 |
| test_6.jpg | 0 |

where the predicition is 0 if predicted as cat, 1 if not-cat

---

***Do not be overwhelmed by the amount of new jargon introduced in this problem statement. We understand that the PS could be difficult to get started at first, hence do not hesitate to reach out to us in case of any queries***.

