import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.tensor import Tensor
import torch
from sklearn.metrics import accuracy_score
from os.path import join


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(246016, 128)
        self.grapheme_root = nn.Linear(128, 168)
        self.vowel_diacritic = nn.Linear(128, 11)
        self.consonant_diacritic = nn.Linear(128, 7)

        self.soft_max = nn.Softmax(1)

        self._criterion = nn.BCELoss()
        self._optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        grapheme_root = self.grapheme_root(x)
        grapheme_root = self.soft_max(grapheme_root)

        vowel_diacritic = self.vowel_diacritic(x)
        vowel_diacritic = self.soft_max(vowel_diacritic)

        consonant_diacritic = self.consonant_diacritic(x)
        consonant_diacritic = self.soft_max(consonant_diacritic)

        return grapheme_root, vowel_diacritic, consonant_diacritic

    def train_on_batch(self, inputs, outputs):
        inputs = Tensor(inputs)
        grapheme_root = Tensor(outputs[0])
        vowel_diacritic = Tensor(outputs[1])
        consonant_diacritic = Tensor(outputs[2])

        self._optimizer.zero_grad()
        grapheme_root_pred, vowel_diacritic_pred, consonant_diacritic_pred = self(inputs)

        grapheme_root_loss = self._criterion(grapheme_root_pred, grapheme_root)
        vowel_diacritic_loss = self._criterion(vowel_diacritic_pred, vowel_diacritic)
        consonant_diacritic_loss = self._criterion(consonant_diacritic_pred, consonant_diacritic)

        loss = grapheme_root_loss + vowel_diacritic_loss + consonant_diacritic_loss

        loss.backward()
        self._optimizer.step()

        return loss.item()

    def predict(self, inputs):
        inputs = Tensor(inputs)
        grapheme_root_hat, vowel_diacritic_hat, consonant_diacritic_hat = self(inputs)

        _, grapheme_root_indices = grapheme_root_hat.max(1)
        _, vowel_diacritic_indices = vowel_diacritic_hat.max(1)
        _, consonant_diacritic_indices = consonant_diacritic_hat.max(1)

        return grapheme_root_indices, vowel_diacritic_indices, consonant_diacritic_indices

    def test_on_batch(self, inputs, outputs):
        grapheme_root = Tensor(outputs[0])
        vowel_diacritic = Tensor(outputs[1])
        consonant_diacritic = Tensor(outputs[2])

        _, grapheme_root_tru = grapheme_root.max(1)
        _, vowel_diacritic_tru = vowel_diacritic.max(1)
        _, consonant_diacritic_tru = consonant_diacritic.max(1)

        grapheme_root_hat, vowel_diacritic_hat, consonant_diacritic_hat = self.predict(inputs)

        grapheme_root = accuracy_score(grapheme_root_tru, grapheme_root_hat)
        vowel_diacritic = accuracy_score(vowel_diacritic_tru, vowel_diacritic_hat)
        consonant_diacritic = accuracy_score(consonant_diacritic_tru, consonant_diacritic_hat)

        print("accuracy: grapheme root {}, vowel diacritic {}, consonant diacritic {}".format(grapheme_root, vowel_diacritic, consonant_diacritic))

    def save_weights(self, location):
        torch.save(self.state_dict(), join(location, "cnn.model"))

    def load_weights(self, location):
        self.load_state_dict(torch.load(join(location, "cnn.model")))
