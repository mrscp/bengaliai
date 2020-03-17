import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.tensor import Tensor
import torch
from sklearn.metrics import accuracy_score
from os.path import join


class CNN(nn.Module):
    """
    CNN model with PyTorch

    Attributes
    ----------
    _device: str
        Should it run on GPU or CPU
    conv1: nn.Conv2d
        First convolution layer
    conv2: nn.Conv2d
        Second convolution layer
    dropout1: nn.Dropout2d
        First dropout layer
    dropout2: nn.Dropout2d
        Second dropout layer
    fc1: nn.Linear
        Fully connected layer

    grapheme_root_priority_1: nn.Linear
        Giving more priority for grapheme_root
    grapheme_root_priority_2: nn.Linear
        Giving more priority for grapheme_root
    grapheme_root: nn.Linear
        Output layer for grapheme_root
    vowel_diacritic: nn.Linear
        Output layer for vowel diacritic
    consonant_diacritic: nn.Linear
        Output layer for consonant diacritic
    soft_max: nn.Softmax
        Softmax activation for the output layers

    _criterion: nn.BCELoss
        Binary cross entropy loss function for the model
    _optimizer: optim.Adam
        Optimizer for the model

    Methods
    -------
    forward(self, x):
        Feed forward method for the model
    train_on_batch(self, inputs, outputs):
        Train the model with a given batch
    predict(self, inputs):
        Predict with the model based on the input given
    test_on_batch(self, inputs, outputs, verbose=0):
        Test the model based on the given input and output values
    save_weights(self, location):
        Saves model weights
    load_weights(self, location):
        Load existing model weights
    """
    def __init__(self):
        super(CNN, self).__init__()
        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(246016, 128)

        self.grapheme_root_priority_1 = nn.Linear(128, 10000)
        self.grapheme_root_priority_2 = nn.Linear(10000, 1024)
        self.grapheme_root = nn.Linear(1024, 168)

        self.vowel_diacritic = nn.Linear(128, 11)

        self.consonant_diacritic = nn.Linear(128, 7)

        self.soft_max = nn.Softmax(1)

        self._criterion = nn.BCELoss()
        self._optimizer = optim.Adam(self.parameters())

        self.to(self._device)
        print("device", self._device)

    def forward(self, x):
        """
        Feed forward method for the model
        :param x: Tensor
            Input data
        :return: (Tensor, Tensor, Tensor)
            Tuple of predicted value of grapheme_root, vowel_diacritic, consonant_diacritic
        """
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

        grapheme_root = self.grapheme_root_priority_1(x)
        grapheme_root = self.grapheme_root_priority_2(grapheme_root)
        grapheme_root = self.grapheme_root(grapheme_root)
        grapheme_root = self.soft_max(grapheme_root)

        vowel_diacritic = self.vowel_diacritic(x)
        vowel_diacritic = self.soft_max(vowel_diacritic)

        consonant_diacritic = self.consonant_diacritic(x)
        consonant_diacritic = self.soft_max(consonant_diacritic)

        return grapheme_root, vowel_diacritic, consonant_diacritic

    def train_on_batch(self, inputs, outputs):
        """
        Train the model based on the given batch of data
        :param inputs: array
            Values for input features
        :param outputs: array
            Values for output features
        :return: float
            Batch loss
        """
        inputs = Tensor(inputs)
        grapheme_root = Tensor(outputs[0])
        vowel_diacritic = Tensor(outputs[1])
        consonant_diacritic = Tensor(outputs[2])

        inputs = inputs.to(self._device)
        grapheme_root = grapheme_root.to(self._device)
        vowel_diacritic = vowel_diacritic.to(self._device)
        consonant_diacritic = consonant_diacritic.to(self._device)

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
        """
        Predits with the model based on the given input feature values
        :param inputs: array
            Input feature values
        :return: (Tensor, Tensor, Tensor)
            Indices for grapheme_root, vowel_diacritic, consonant_diacritic
        """
        inputs = Tensor(inputs)
        inputs = inputs.to(self._device)
        grapheme_root_hat, vowel_diacritic_hat, consonant_diacritic_hat = self(inputs)

        _, grapheme_root_indices = grapheme_root_hat.max(1)
        _, vowel_diacritic_indices = vowel_diacritic_hat.max(1)
        _, consonant_diacritic_indices = consonant_diacritic_hat.max(1)

        return grapheme_root_indices, vowel_diacritic_indices, consonant_diacritic_indices

    def test_on_batch(self, inputs, outputs, verbose=0):
        """

        :param inputs:
        :param outputs:
        :param verbose:
        :return:
        """
        grapheme_root = Tensor(outputs[0])
        vowel_diacritic = Tensor(outputs[1])
        consonant_diacritic = Tensor(outputs[2])

        _, grapheme_root_tru = grapheme_root.max(1)
        _, vowel_diacritic_tru = vowel_diacritic.max(1)
        _, consonant_diacritic_tru = consonant_diacritic.max(1)

        grapheme_root_hat, vowel_diacritic_hat, consonant_diacritic_hat = self.predict(inputs)
        grapheme_root_hat = grapheme_root_hat.cpu()
        vowel_diacritic_hat = vowel_diacritic_hat.cpu()
        consonant_diacritic_hat = consonant_diacritic_hat.cpu()

        if verbose == 1:
            print(grapheme_root_tru[:10])
            print(grapheme_root_hat[:10])
            print(vowel_diacritic_tru[:10])
            print(vowel_diacritic_hat[:10])
            print(consonant_diacritic_tru[:10])
            print(consonant_diacritic_hat[:10])

        grapheme_root = accuracy_score(grapheme_root_tru, grapheme_root_hat)
        vowel_diacritic = accuracy_score(vowel_diacritic_tru, vowel_diacritic_hat)
        consonant_diacritic = accuracy_score(consonant_diacritic_tru, consonant_diacritic_hat)

        print("accuracy: grapheme root {}, vowel diacritic {}, consonant diacritic {}".format(grapheme_root, vowel_diacritic, consonant_diacritic))

    def save_weights(self, location):
        torch.save(self.state_dict(), join(location, "cnn.weights"))

    def load_weights(self, location):
        self.load_state_dict(torch.load(join(location, "cnn.weights")))
        self.eval()
