# Beginners Hypothesis 2023 (For Sophomore Recruitments)

## Problem Statement

You are a cat lover (yes you are!). So you decide to open a cat park allowing owners to bring their cats. You however need to protect the park from other animals. Hence you turn towards the ***EvEr-So-ReLiAnT*** machine learning to tackle this. You ask your experienced friend Aaryan to build an image classification model to classify cats from other animals. And it works like a charm!

Sarthak however wants to get his pet deer into the park. And he knows how to get this done (insert wicked smile). [Inspired by previous research](https://towardsdatascience.com/avoiding-detection-with-adversarial-t-shirts-bb620df2f7e6), he steals the model from Aaryan's _Mac_ and uses it learn features that will help his deer classify itself as a cat. Too easy for Sarthak!

To save the park, you need to build a robust cat classification model, one which is less prone to such _attacks_. You will be provided with a dataset with two categories: cat, not-cat. This is the same dataset which was used to train the original model.

While you develop your model, we've hired Sarthak to provide us with more such features on other animals so that we can test the robustness of your classifier. Your model will be tried and tested on these images. _Accuracy of correct classification_ will be used to select what goes into the park camera.

## Technical Details

- Such attacks on machine learning models are called **Adversarial Attacks**. For the purpose of this problem statement we use the **FGSM Attack** to generate images which could evade the machine learning model.

- The initial baseline model (developed by Aaryan) is a simple CNN, trained on [`train_data.zip`](train_data.zip). More details on training can be found in the [`train_baseline.py`](train_baseline.py). The same file is used to produce [`model.pt`](model.pt) which has already been provided in the repository 
  ```
  > python train_baseline.py --train_data 'path/to/train/data'
  ```

- This is the baseline training. To approach the problem, you need to change the baseline training to **adversarial training**. This makes the model more robust to adversarial attacks.

- **To read more about adversarial attacks and adversarial training, refer to this [blog post](https://adversarial-ml-tutorial.org/introduction/)**. This is just the introduction, and the internet is your limit.

- As mentioned above, Sarthak evaded the machine learning model by using the FGSM attack. You are encouraged to read more about it online.
  - Adversarial samples are specific to the model weights.
  - Adversarial sample generated using FGSM attack may be effective against one model but not another though they have trained using the same technique.
  - Hence, for every model submission we recieve, we will attack the model and create new adversarial samples to test the model.
  - For transperancy, we provide the [Python script (FGSM attack implemented)](generate_evasion_examples.py) which will generate the adversarial samples which your model will be tested on:

    ```
    > python generate_adversarial_examples.py --train_data 'path/to/train_data.zip' --model 'path/to/model.pt'
    ```

    This script will also output the **clean** (on the actual train data) and **robust** (on FGSM attacked data) **accuracies** of your model.

- **Many adversarial defense techniques have been implemented [here](https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/image#defense-methods)**

- For a reference, some numbers on the baseline model ([`model.pt`](model.pt)) are provided below:

  | Data | Accuracy Score |
  | -- | -- |
  | Train Data | 94.19% (Clean Accuracy) |
  | Adversarial Images using FGSM (using [`generate_evasion_examples.py`](generate_evasion_examples.py)) | 43.54% (Robust Accuracy) |

  Your job is thus to improve the accuracy score of 43.54 % in the second row.

## Submission

You just need to submit the model weights file. Submit your file to [this Google form](https://docs.google.com/forms/d/e/1FAIpQLSeGOEURPFrh5MxYFUt1aR2GvlTEj0x7nO8VPnKorCahHzGmEA/viewform) (only IITR ID).

**Note**: Save the model's `state_dict` into the file as shown below:

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

# adversarial training ...

# save the model's state_dict
torch.save(net.state_dict(), 'model.pt')
```

## Leaderboard

Since we have to generate the test data for each submission, hosting on Kaggle is not possible. We will however update the leaderboard here as we start recieving the submission.

| Name | Clean Accuracy | Robust Accuracy |
| -- | -- | -- |
| Adarsh | 100% | 57.45% |
| Sarvagya | 70.38% | 53.79% |
| Suyash | 95.57% | 44.92% |
| DSG | 94.19% | 43.54% |


---

***Do not be overwhelmed by the amount of new jargon introduced in this problem statement. We understand that the PS could be difficult to get started at first, hence do not hesitate to reach out to us in case of any queries***.

